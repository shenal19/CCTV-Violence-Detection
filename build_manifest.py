"""Phase 1: build the train/val/test manifest from the RWF-2000 official split.

Mapping (keeps the official split clean and comparable to published papers):
  official train/  ->  our train + our val   (val carved from train only, fixed seed)
  official val/    ->  our test               (held out, used only for final metrics)

Usage (from the project root):
    python -m src.build_manifest
    python -m src.build_manifest --validate     # also decode-check every clip (slower)
"""
import argparse
import json
import random
from pathlib import Path

from src.config import load_config, ROOT

# Accepts either Fight/NonFight or Violence/NonViolence naming.
LABEL_MAP = {
    "fight": 1, "violence": 1, "violent": 1,
    "nonfight": 0, "nonviolence": 0, "nonviolent": 0,
}
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def label_from_folder(name: str):
    return LABEL_MAP.get(name.strip().lower().replace(" ", "").replace("_", ""))


def scan_split(split_dir: Path):
    items = []
    if not split_dir.exists():
        print(f"  ! missing split folder: {split_dir}")
        return items
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        label = label_from_folder(class_dir.name)
        if label is None:
            print(f"  ! skipping unrecognized class folder: {class_dir.name}")
            continue
        for f in sorted(class_dir.iterdir()):
            if f.suffix.lower() in VIDEO_EXTS:
                items.append({"path": str(f.resolve()), "label": label})
    return items


def validate(items, min_frames):
    """Open each clip and keep only those that decode and have enough frames."""
    from decord import VideoReader
    good, bad = [], []
    for it in items:
        try:
            vr = VideoReader(it["path"])
            n = len(vr)
            if n >= min_frames:
                it["n_frames"] = n
                good.append(it)
            else:
                bad.append((it["path"], f"only {n} frames"))
        except Exception as e:  # noqa: BLE001
            bad.append((it["path"], str(e)))
    return good, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="decode-check every clip (slower, needs decord)")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(cfg["paths"]["rwf2000_root"])
    num_frames = cfg["data"]["num_frames"]
    val_fraction = cfg["data"]["val_fraction"]
    seed = cfg["project"]["seed"]

    print(f"RWF-2000 root: {root}")
    train_official = scan_split(root / "train")
    test_items = scan_split(root / "val")  # official val == our test set
    print(f"  official train clips: {len(train_official)}")
    print(f"  official val/test clips: {len(test_items)}")

    if args.validate:
        print("Validating clips by decoding (this takes a few minutes)...")
        train_official, bad_tr = validate(train_official, num_frames)
        test_items, bad_te = validate(test_items, num_frames)
        bad = bad_tr + bad_te
        if bad:
            log = ROOT / "outputs" / "logs" / "bad_clips.log"
            log.write_text("\n".join(f"{p}\t{r}" for p, r in bad))
            print(f"  skipped {len(bad)} unreadable/short clips -> {log}")

    # Carve val out of the official TRAIN only, stratified by label, fixed seed.
    rng = random.Random(seed)
    by_label = {0: [], 1: []}
    for it in train_official:
        by_label[it["label"]].append(it)
    train_items, val_items = [], []
    for label, lst in by_label.items():
        rng.shuffle(lst)
        n_val = int(len(lst) * val_fraction)
        val_items.extend(lst[:n_val])
        train_items.extend(lst[n_val:])
    rng.shuffle(train_items)
    rng.shuffle(val_items)

    manifest = {
        "meta": {
            "rwf2000_root": str(root),
            "seed": seed,
            "val_fraction": val_fraction,
            "validated": args.validate,
            "counts": {"train": len(train_items), "val": len(val_items), "test": len(test_items)},
        },
        "train": train_items,
        "val": val_items,
        "test": test_items,
    }
    out = ROOT / "outputs" / "split_manifest.json"
    out.write_text(json.dumps(manifest, indent=2))

    print(f"\nWrote {out}")
    for name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        pos = sum(i["label"] for i in items)
        print(f"  {name:5s}: {len(items):4d} clips  ({pos} violent / {len(items) - pos} non-violent)")
    if len(train_items) == 0:
        print("\n! No clips found. Check that paths.rwf2000_root in config.yaml points to the"
              " folder that directly contains train/ and val/.")


if __name__ == "__main__":
    main()
