"""Phase 1 "done when" check.

Run build_manifest first, then this. It pulls one training batch and prints its shape,
which should be [B, 3, 16, 112, 112].

    python -m src.build_manifest
    python scripts/phase1_smoke_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config           # noqa: E402
from src.dataset import build_dataloaders     # noqa: E402


def main():
    cfg = load_config()
    train_dl, val_dl, test_dl = build_dataloaders(cfg)
    print(f"batches -> train={len(train_dl)}  val={len(val_dl)}  test={len(test_dl)}")

    x, y = next(iter(train_dl))
    print(f"batch tensor shape: {tuple(x.shape)}   (want [B, 3, 16, 112, 112])")
    print(f"labels in batch:    {y.tolist()}")
    print(f"dtype={x.dtype}  value range=[{x.min():.2f}, {x.max():.2f}]")

    ok = tuple(x.shape)[1:] == (3, 16, 112, 112)
    print("\nPHASE 1 PASS" if ok else "\nShape mismatch — check config num_frames/img_size.")


if __name__ == "__main__":
    main()
