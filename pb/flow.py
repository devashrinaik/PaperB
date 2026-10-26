"""RAFT-large flow control: per native frame pair, mean flow magnitude and camera-motion proxy."""
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

from .gpu import run_batched, wait_for_free_memory
from .video import to_tensor


class Flow:
    def __init__(self, device=0, size=256):
        self.device = torch.device(f"cuda:{device}")
        wait_for_free_memory(device, 2.0)
        w = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=w).to(self.device).eval()
        self.size = size

    @torch.no_grad()
    def _pairs(self, pairs):
        a = torch.stack([p[0] for p in pairs]).to(self.device)
        b = torch.stack([p[1] for p in pairs]).to(self.device)
        fl = self.model(a, b)[-1]  # [B,2,H,W], pixels at `size`
        mag = fl.norm(dim=1).mean((1, 2))
        med = fl.flatten(2).median(dim=2).values.norm(dim=1)  # magnitude of median flow vector
        return list(zip(mag.cpu().numpy(), med.cpu().numpy()))

    def clip_flow(self, frames, batch=8):
        """frames: native uint8 [T,H,W,3]. Returns f[T-1], cam[T-1]."""
        x = to_tensor(frames, self.size, normalize=False) * 2 - 1  # RAFT expects [-1, 1]
        x = x.permute(1, 0, 2, 3)  # T,3,H,W
        pairs = [(x[t], x[t + 1]) for t in range(len(x) - 1)]
        res, _ = run_batched(self._pairs, pairs, batch)
        f = np.array([r[0] for r in res], np.float32)
        cam = np.array([r[1] for r in res], np.float32)
        return f, cam
