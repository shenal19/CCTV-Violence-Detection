"""Phase 2 "done when": save a grid of augmented training frames so you can eyeball them.

    python scripts/visualize_augmentation.py
Then open outputs/augmentation_preview.png

What you want to see: recognizable video frames with visible but MILD variation
(different crop, sometimes flipped, slight brightness/colour shifts) — not distorted,
blanked, or unrecognizable. Rerun it to see different random augmentations.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                         # noqa: E402
import matplotlib                          # noqa: E402
matplotlib.use("Agg")                      # no display needed, just save a file
import matplotlib.pyplot as plt            # noqa: E402

from src.config import load_config, ROOT   # noqa: E402
from src.dataset import RWFDataset, load_manifest, MEAN, STD  # noqa: E402

N_CLIPS = 3
N_FRAMES = 6
CLASS_NAMES = ["NonViolence", "Violence"]


def denorm(x):
    """(C,T,H,W) normalized tensor -> (T,H,W,C) uint8 image stack."""
    x = x.permute(1, 2, 3, 0).numpy()
    x = x * STD + MEAN
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def main():
    cfg = load_config()
    ds = RWFDataset(load_manifest()["train"], cfg, training=True)

    fig, axes = plt.subplots(N_CLIPS, N_FRAMES, figsize=(N_FRAMES * 2, N_CLIPS * 2))
    for r in range(N_CLIPS):
        x, label = ds[np.random.randint(len(ds))]
        frames = denorm(x)
        cols = np.linspace(0, frames.shape[0] - 1, N_FRAMES).astype(int)
        for c, t in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(frames[t])
            ax.axis("off")
            if c == 0:
                ax.set_title(CLASS_NAMES[label], loc="left", fontsize=10)
    fig.suptitle("Augmented training frames (rerun for different crops / flips / colour)")
    fig.tight_layout()

    out = ROOT / "outputs" / "augmentation_preview.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    print(f"Saved {out}")
    print("Open it — frames should be recognizable video with mild, varied augmentation.")


if __name__ == "__main__":
    main()
