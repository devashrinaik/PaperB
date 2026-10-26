"""LTX-Video 2B (frozen): YoCausal-style forward vs reversed denoising loss per latent frame.

Protocol in prereg/common.md. The null-caption T5 embedding is computed once on CPU,
cached, and the T5 encoder is never loaded on the GPU; scoring uses VAE + transformer only.
"""
import os

import numpy as np
import torch

from .gpu import run_batched, wait_for_free_memory
from .video import to_tensor

WEIGHTS = "/mnt/ssd1/devashri/CVPR/weights/LTX-Video"
NULL_EMB = "/mnt/ssd1/devashri/CVPR/weights/ltx_null_caption.pt"
SIGMAS = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS = (0, 1, 2, 3)
SIZE = 384
MAX_SEQ = 128


def null_caption_embedding():
    """T5 embedding of the empty caption, computed once on CPU and cached."""
    if os.path.exists(NULL_EMB):
        return torch.load(NULL_EMB)
    from transformers import T5EncoderModel, T5TokenizerFast
    tok = T5TokenizerFast.from_pretrained(WEIGHTS, subfolder="tokenizer")
    enc = T5EncoderModel.from_pretrained(WEIGHTS, subfolder="text_encoder", torch_dtype=torch.float32).eval()
    t = tok([""], padding="max_length", max_length=MAX_SEQ, truncation=True, add_special_tokens=True, return_tensors="pt")
    with torch.no_grad():
        emb = enc(t.input_ids)[0]
    out = dict(embeds=emb.half(), mask=t.attention_mask.bool())
    torch.save(out, NULL_EMB)
    del enc
    return out


class LTX:
    def __init__(self, device=0):
        from diffusers import AutoencoderKLLTXVideo, LTXVideoTransformer3DModel
        from diffusers.pipelines.ltx.pipeline_ltx import LTXPipeline
        self.device = torch.device(f"cuda:{device}")
        emb = null_caption_embedding()  # CPU only; T5 freed before GPU work
        wait_for_free_memory(device, 7.0)
        self.vae = AutoencoderKLLTXVideo.from_pretrained(WEIGHTS, subfolder="vae", torch_dtype=torch.float16).to(self.device).eval()
        self.tr = LTXVideoTransformer3DModel.from_pretrained(WEIGHTS, subfolder="transformer", torch_dtype=torch.float16).to(self.device).eval()
        self.emb, self.mask = emb["embeds"].to(self.device), emb["mask"].to(self.device)
        self.pack, self.norm = LTXPipeline._pack_latents, LTXPipeline._normalize_latents
        self.t_ratio = self.vae.temporal_compression_ratio
        self.s_ratio = self.vae.spatial_compression_ratio

    @torch.no_grad()
    def _latents(self, frames):
        x = to_tensor(frames, SIZE, normalize=False) * 2 - 1  # [3,F,H,W] in [-1,1]
        x = x.unsqueeze(0).to(self.device, torch.float16)
        z = self.vae.encode(x).latent_dist.mode()
        return self.norm(z, self.vae.latents_mean, self.vae.latents_std, self.vae.config.scaling_factor).float()

    @torch.no_grad()
    def _loss_items(self, items, z, fps):
        """items: list of (sigma, seed). Returns per-latent-frame MSE for each item."""
        _, C, Fl, Hl, Wl = z.shape
        zs, vs, ts = [], [], []
        for sigma, seed in items:
            g = torch.Generator(device="cpu").manual_seed(seed)
            eps = torch.randn(z.shape, generator=g).to(self.device)
            zs.append((1 - sigma) * z + sigma * eps)
            vs.append(eps - z)
            ts.append(sigma * 1000.0)
        zt = torch.cat(zs)
        B = zt.shape[0]
        pred = self.tr(
            hidden_states=self.pack(zt.half()),
            encoder_hidden_states=self.emb.expand(B, -1, -1),
            timestep=torch.tensor(ts, device=self.device),
            encoder_attention_mask=self.mask.expand(B, -1),
            num_frames=Fl, height=Hl, width=Wl,
            rope_interpolation_scale=(self.t_ratio / fps, self.s_ratio, self.s_ratio),
            return_dict=False,
        )[0].float()
        pred = pred.reshape(B, Fl, Hl, Wl, C).permute(0, 4, 1, 2, 3)  # unpack (patch size 1)
        mse = ((pred - torch.cat(vs)) ** 2).mean((1, 3, 4))  # [B, Fl]
        if not torch.isfinite(mse).all():
            raise FloatingPointError("non-finite LTX loss in fp16")
        return list(mse.cpu().numpy())

    def _direction(self, frames, fps, batch):
        z = self._latents(frames)
        items = [(s, k) for s in SIGMAS for k in SEEDS]
        res, batch = run_batched(lambda it: self._loss_items(it, z, fps), items, batch)
        per = np.stack(res).reshape(len(SIGMAS), len(SEEDS), -1)
        return per.mean((0, 1)), per.mean(1), batch

    def clip_loss(self, frames, fps=None, batch=4):
        """Native frames, trimmed to 8k+1. Latent k>=1 covers native frames 8k-7..8k;
        reversed latent k covers the same frames as forward latent K+1-k (exact).
        Latent 0 (a single frame) has no counterpart and is left undefined."""
        fps = fps or 15.0
        F = (len(frames) - 1) // self.t_ratio * self.t_ratio + 1
        frames = frames[:F]
        K = (F - 1) // self.t_ratio
        lf, lf_s, batch = self._direction(frames, fps, batch)
        lb_r, lb_r_s, batch = self._direction(frames[::-1].copy(), fps, batch)
        e_f = np.full(K + 1, np.nan, np.float32)
        e_b = np.full(K + 1, np.nan, np.float32)
        e_f[1:] = lf[1:]
        e_b[1:] = lb_r[1:][::-1]  # forward latent k' <- reversed latent K+1-k'
        centre = np.array([0.0] + [self.t_ratio * k - (self.t_ratio - 1) / 2 for k in range(1, K + 1)])  # 8k-3.5
        es_b = np.full_like(lf_s, np.nan)
        es_b[:, 1:] = lb_r_s[:, 1:][:, ::-1]
        lf_s[:, 0] = np.nan
        return dict(e_f_mse=e_f, e_b_mse=e_b, latent_centre=centre.astype(np.float32),
                    e_f_mse_by_sigma=lf_s.astype(np.float32), e_b_mse_by_sigma=es_b.astype(np.float32),
                    sigmas=np.array(SIGMAS, np.float32), n_frames_used=F, fps=fps, batch_used=batch)
