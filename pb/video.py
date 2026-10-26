"""Video decoding, shortcut filters, frame sampling and spatial preprocessing."""
import os

import av
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1, 1)
IMG_EXT = (".png", ".jpg", ".jpeg")


def read_video(path):
    """Return (frames uint8 [T,H,W,3], fps). `path` is a video file or a dir of images."""
    if os.path.isdir(path):
        import cv2
        names = sorted(f for f in os.listdir(path) if f.lower().endswith(IMG_EXT))
        frames = [cv2.cvtColor(cv2.imread(os.path.join(path, f)), cv2.COLOR_BGR2RGB) for f in names]
        return np.stack(frames), None
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate else None
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(s)]
    return np.stack(frames), fps


def check_shortcuts(frames, black_thr=10 / 255, cut_thr=0.5, bins=32):
    """Pre-registered filters. Returns None if clean, else a reason string."""
    x = frames.astype(np.float32) / 255
    means = x.reshape(len(x), -1).mean(1)
    if (means < black_thr).any():
        return f"black_frame@{int(np.argmax(means < black_thr))}"
    hists = np.stack([
        np.concatenate([np.histogram(f[..., ch], bins=bins, range=(0, 256))[0] for ch in range(3)])
        for f in frames
    ]).astype(np.float64)
    for t in range(len(hists) - 1):
        a, b = hists[t], hists[t + 1]
        if a.std() == 0 or b.std() == 0:
            continue
        if np.corrcoef(a, b)[0, 1] < cut_thr:
            return f"cut_or_fade@{t}"
    return None


def sample(frames, stride, offset):
    """Native frames -> sampled frames, plus native index of each sampled frame.
    The sampled length is trimmed to even so tubelets (2 frames) tile it exactly."""
    idx = np.arange(offset, len(frames), stride)
    idx = idx[: len(idx) // 2 * 2]
    return frames[idx], idx


def to_tensor(frames, size, normalize=True):
    """uint8 [T,H,W,3] -> float [3,T,size,size]: short side resized to size, centre crop."""
    x = torch.from_numpy(np.ascontiguousarray(frames)).permute(3, 0, 1, 2).float() / 255  # C,T,H,W
    _, T, H, W = x.shape
    s = size / min(H, W)
    nh, nw = max(size, round(H * s)), max(size, round(W * s))
    x = F.interpolate(x.permute(1, 0, 2, 3), size=(nh, nw), mode="bilinear", antialias=True, align_corners=False)
    top, left = (nh - size) // 2, (nw - size) // 2
    x = x[:, :, top:top + size, left:left + size].permute(1, 0, 2, 3)
    if normalize:
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x
