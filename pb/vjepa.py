"""Frozen V-JEPA 2 / 2.1 forward and reversed prediction errors per tubelet.

Follows V-JEPA training: context tokens from the online `encoder` on the
context tubelets only; targets are layer-normed `target_encoder` outputs on the
full window (per hierarchy level for 2.1); the predictor fills masked target
tubelets. See prereg/common.md and prereg/impl_notes.md.
"""
import logging

import numpy as np
import torch
import torch.nn.functional as F

from src.hub import backbones as hb

from .gpu import run_batched, wait_for_free_memory
from .video import to_tensor

log = logging.getLogger(__name__)
CKPT_DIR = "/mnt/ssd1/devashri/vjepa2_ckpts"

SPECS = {
    "vjepa2_vitg": dict(family="v2", arch="vit_giant", file="vitg.pt", img=256, need_gb=7.0),
    "vjepa2_vitl": dict(family="v2", arch="vit_large", file="vitl.pt", img=256, need_gb=3.0),
    "vjepa2_1_vitg": dict(family="v21", arch="vjepa2_1_vit_giant_384", file="vjepa2_1_vitg_384.pt", img=384, need_gb=7.0),
}


def _clean(sd):
    return hb._clean_backbone_key(sd)


def _build(family, arch, img):
    if family == "v2":
        return hb._make_vjepa2_model(model_name=arch, img_size=img, pretrained=False)
    return hb._make_vjepa2_1_model(
        model_name=arch, img_size=img, predictor_num_mask_tokens=8,
        n_output_distillation=4, return_all_tokens=True, pretrained=False)


class VJEPA:
    def __init__(self, name, device=0, img=None):
        spec = SPECS[name]
        self.name, self.family = name, spec["family"]
        self.img = img or spec["img"]
        self.device = torch.device(f"cuda:{device}")
        wait_for_free_memory(device, spec["need_gb"])
        ck = torch.load(f"{CKPT_DIR}/{spec['file']}", map_location="cpu", mmap=True, weights_only=False)
        enc, pred = _build(self.family, spec["arch"], self.img)
        tgt, _ = _build(self.family, spec["arch"], self.img)
        strict = self.family == "v21"  # v2 checkpoints carry an unused pos_embed (RoPE models)
        for m, key in ((enc, "encoder"), (tgt, "target_encoder"), (pred, "predictor")):
            missing, unexpected = m.load_state_dict(_clean(ck[key]), strict=strict)
            bad = [k for k in missing if "pos_embed" not in k]
            if bad:
                raise RuntimeError(f"{name}/{key}: missing weights {bad[:5]}")
        del ck
        self.enc, self.tgt, self.pred = (m.to(self.device).half().eval() for m in (enc, tgt, pred))
        self.patch, self.tubelet = 16, 2
        self.P = (self.img // self.patch) ** 2  # tokens per tubelet
        self.D = self.tgt.embed_dim

    # ---- model calls -------------------------------------------------------
    def _encode_context(self, x, ctx):
        if self.family == "v21":
            return self.enc(x, masks=ctx, training=True)
        return self.enc(x, masks=ctx)

    def _encode_target(self, x):
        if self.family == "v21":
            h = self.tgt(x, training=True)  # [B, N, 4*D], one block per hierarchy level
            return torch.cat([F.layer_norm(c, (self.D,)) for c in h.split(self.D, dim=-1)], dim=-1)
        h = self.tgt(x)
        return F.layer_norm(h, (h.size(-1),))

    def _predict(self, z, ctx, tgt):
        out = self.pred(z, ctx, tgt)
        return out[0] if isinstance(out, tuple) else out

    @torch.no_grad()
    def _window_errors(self, windows, c, out):
        """windows: list of [3,T,H,W] tensors, T=2*(c+out). Returns per-window
        (l2, l1) errors on the first target tubelet."""
        x = torch.stack(windows).to(self.device, torch.float16)
        B, P = x.shape[0], self.P
        ctx = torch.arange(0, c * P, device=self.device).unsqueeze(0).repeat(B, 1)
        tgt = torch.arange(c * P, (c + out) * P, device=self.device).unsqueeze(0).repeat(B, 1)
        with torch.autocast("cuda", dtype=torch.float16):
            h = self._encode_target(x)
            z = self._encode_context(x, ctx)
            p = self._predict(z, [ctx], [tgt])
        h_t = h[:, c * P:(c + 1) * P].float()
        p_t = p[:, :P].float()
        if p_t.shape[-1] != h_t.shape[-1]:
            raise RuntimeError(f"predictor dim {p_t.shape[-1]} != target dim {h_t.shape[-1]}")
        d = p_t - h_t
        l2 = d.norm(dim=-1).mean(1)
        l1 = d.abs().mean((1, 2))
        return list(zip(l2.cpu().numpy(), l1.cpu().numpy()))

    def _direction(self, x, c, out, batch):
        """x: [3,M,H,W] sampled frames, M even. Returns arrays length N (tubelets),
        nan where tubelet j cannot be the first target of a window."""
        N = x.shape[1] // 2
        js = list(range(c, N - out + 1))
        l2 = np.full(N, np.nan, np.float32)
        l1 = np.full(N, np.nan, np.float32)
        if not js:
            return l2, l1, batch
        wins = [x[:, 2 * (j - c):2 * (j + out)] for j in js]
        res, batch = run_batched(lambda w: self._window_errors(w, c, out), wins, batch)
        for j, (a, b) in zip(js, res):
            l2[j], l1[j] = a, b
        return l2, l1, batch

    def clip_errors(self, sampled_frames, c=4, out=1, batch=2):
        """Forward and reversed errors aligned to the same physical tubelet."""
        x = to_tensor(sampled_frames, self.img)
        N = x.shape[1] // 2
        ef2, ef1, batch = self._direction(x, c, out, batch)
        eb2_r, eb1_r, batch = self._direction(x.flip(1), c, out, batch)
        # reversed tubelet i == physical tubelet N-1-i (M even, so tubelets tile exactly)
        eb2, eb1 = eb2_r[::-1].copy(), eb1_r[::-1].copy()
        return dict(e_f_l2=ef2, e_b_l2=eb2, e_f_l1=ef1, e_b_l1=eb1, n_tubelets=N, c=c, out=out,
                    img=self.img, batch_used=batch)
