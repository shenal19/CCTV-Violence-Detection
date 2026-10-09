"""Phase 3a check: build R(2+1)D-18 and run ONE forward+backward pass on the GPU.

Confirms the model runs on your card and reports peak VRAM, so we know whether
batch_size 4 fits in 4GB before starting a real training run.

    python scripts/model_smoke_test.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch                              # noqa: E402
import torch.nn as nn                     # noqa: E402

from src.config import load_config        # noqa: E402
from src.dataset import build_dataloaders  # noqa: E402
from src.model import build_model         # noqa: E402


def main():
    cfg = load_config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    train_dl, _, _ = build_dataloaders(cfg)
    x, y = next(iter(train_dl))
    y = torch.as_tensor(y)
    x, y = x.to(device), y.to(device)
    print(f"input batch: {tuple(x.shape)}")

    print("building R(2+1)D-18 (first run downloads Kinetics weights ~120MB)...")
    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"])
    crit = nn.CrossEntropyLoss()
    use_amp = cfg["train"]["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    t0 = time.time()
    opt.zero_grad()
    with torch.amp.autocast("cuda", enabled=use_amp):
        out = model(x)
        loss = crit(out, y)
    scaler.scale(loss).backward()
    scaler.step(opt)
    scaler.update()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"output: {tuple(out.shape)}   (want [{x.shape[0]}, 2])")
    print(f"loss: {loss.item():.4f}")
    print(f"one train step: {dt:.2f}s")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"peak VRAM this step: {peak:.2f} GB  (your card has ~4.0 GB)")

    ok = tuple(out.shape) == (x.shape[0], 2)
    print("\nPHASE 3a PASS" if ok else "\nunexpected output shape")


if __name__ == "__main__":
    main()
