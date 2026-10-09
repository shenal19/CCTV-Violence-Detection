"""Phase 1: RWF-2000 Dataset and DataLoaders.

Returns each clip as a tensor of shape [C, T, H, W] = [3, 16, 112, 112], the input
R(2+1)D-18 expects.

Temporal sampling = uniform segment sampling across the WHOLE clip: the clip is split
into `num_frames` equal segments and one frame is taken per segment. For training the
frame is picked randomly within its segment (augmentation); for eval the segment centre
is used. This guarantees the whole 5s is covered, so the violent moment in a clip-level
label is never missed (the weak-label trap we discussed).
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.config import ROOT

# Kinetics normalization — the stats torchvision's R(2+1)D pretraining used.
MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
STD = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)


def sample_indices(total, num_frames, training):
    if total <= 0:
        return np.zeros(num_frames, dtype=int)
    seg = total / num_frames
    out = []
    for i in range(num_frames):
        start, end = i * seg, (i + 1) * seg
        f = np.random.uniform(start, end) if training else (start + end) / 2.0
        out.append(min(int(f), total - 1))
    return np.array(out, dtype=int)


def _resize_short_side(frames, size):
    import cv2
    T, H, W, C = frames.shape
    if H <= W:
        new_h, new_w = size, max(size, int(round(W * size / H)))
    else:
        new_h, new_w = max(size, int(round(H * size / W))), size
    out = np.empty((T, new_h, new_w, C), dtype=frames.dtype)
    for t in range(T):
        out[t] = cv2.resize(frames[t], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return out


def _crop(frames, size, training):
    T, H, W, C = frames.shape
    if training:
        top = np.random.randint(0, H - size + 1)
        left = np.random.randint(0, W - size + 1)
    else:
        top, left = (H - size) // 2, (W - size) // 2
    return frames[:, top:top + size, left:left + size, :]


def _color_jitter(x, b=0.2, c=0.2, s=0.2):
    """Brightness/contrast/saturation jitter on float frames in [0,1].
    One random factor per clip (consistent across all frames, so motion stays coherent)."""
    x = x * (1.0 + np.random.uniform(-b, b))                       # brightness
    mean = x.mean(axis=(0, 1, 2), keepdims=True)
    x = (x - mean) * (1.0 + np.random.uniform(-c, c)) + mean       # contrast
    gray = x.mean(axis=3, keepdims=True)
    x = (x - gray) * (1.0 + np.random.uniform(-s, s)) + gray       # saturation
    return np.clip(x, 0.0, 1.0)


class RWFDataset(Dataset):
    def __init__(self, split_items, cfg, training):
        self.items = split_items
        self.num_frames = cfg["data"]["num_frames"]
        self.img_size = cfg["data"]["img_size"]
        self.training = training

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from decord import VideoReader, cpu
        it = self.items[i]
        try:
            vr = VideoReader(it["path"], ctx=cpu(0))
            idx = sample_indices(len(vr), self.num_frames, self.training)
            frames = vr.get_batch(idx).asnumpy()           # (T, H, W, C) uint8
        except Exception:  # noqa: BLE001 — one bad file shouldn't kill a run
            frames = np.zeros((self.num_frames, self.img_size, self.img_size, 3), dtype=np.uint8)

        frames = _resize_short_side(frames, int(round(self.img_size * 1.14)))  # ~128 for 112
        frames = _crop(frames, self.img_size, self.training)
        if self.training and np.random.rand() < 0.5:
            frames = frames[:, :, ::-1, :]                 # horizontal flip

        x = frames.astype(np.float32) / 255.0
        if self.training:
            x = _color_jitter(x)                           # Phase 2 augmentation
        x = (x - MEAN) / STD
        x = torch.from_numpy(np.ascontiguousarray(x))      # (T, H, W, C)
        x = x.permute(3, 0, 1, 2).contiguous()             # (C, T, H, W)
        return x, int(it["label"])


def load_manifest():
    return json.loads((ROOT / "outputs" / "split_manifest.json").read_text())


def build_dataloaders(cfg, manifest=None):
    if manifest is None:
        manifest = load_manifest()
    bs = cfg["train"]["batch_size"]
    nw = cfg["data"]["num_workers"]
    train_ds = RWFDataset(manifest["train"], cfg, training=True)
    val_ds = RWFDataset(manifest["val"], cfg, training=False)
    test_ds = RWFDataset(manifest["test"], cfg, training=False)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    return train_dl, val_dl, test_dl
