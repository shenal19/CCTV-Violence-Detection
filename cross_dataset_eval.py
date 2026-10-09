"""Phase 4: cross-dataset generalization test.

Runs the RWF-2000-trained best.pt on a DIFFERENT violence dataset it was never
trained on (Hockey Fights or RLVS), and reports metrics. The drop vs the RWF-2000
test number is the honest generalization gap for the report. The model is NOT
retrained or tuned here.

Two layouts supported:
  (a) subdirectories  -> --violence-subdir / --nonviolence-subdir
  (b) filename prefix -> --violence-prefix  / --nonviolence-prefix   (e.g. Hockey: fi / no)

Examples:
  # Hockey Fights (flat folder, files like fi1_xvid.avi / no1_xvid.avi):
  python scripts/cross_dataset_eval.py --root data/HockeyFights --violence-prefix fi --nonviolence-prefix no
  # RLVS (folders Violence/ and NonViolence/):
  python scripts/cross_dataset_eval.py --root data/RLVS --violence-subdir Violence --nonviolence-subdir NonViolence
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                   # noqa: E402
import torch                                                         # noqa: E402
from sklearn.metrics import (                                        # noqa: E402
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.config import ROOT, load_config                            # noqa: E402
from src.dataset import _resize_short_side, _crop, MEAN, STD        # noqa: E402
from src.model import build_model                                   # noqa: E402

POS = 1
VIDEO_EXTS = {".avi", ".mp4", ".mov", ".mpeg", ".mpg", ".mkv", ".m4v"}


def resolve_device(cfg):
    want = str(cfg.get("project", {}).get("device", "auto")).lower()
    return "cpu" if want == "cpu" else ("cuda" if torch.cuda.is_available() else "cpu")


def phase_indices(total, nf, phase):
    if total <= 0:
        return np.zeros(nf, dtype=int)
    seg = total / nf
    return np.minimum(((np.arange(nf) + phase) * seg).astype(int), total - 1)


def list_videos(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in VIDEO_EXTS)


def build_items(root, vsub, nvsub, vpre, npre):
    root = Path(root)
    items = []
    if vsub and nvsub:
        for p in list_videos(root / vsub):
            items.append({"path": str(p), "label": 1})
        for p in list_videos(root / nvsub):
            items.append({"path": str(p), "label": 0})
    elif vpre and npre:
        for p in list_videos(root):
            name = p.name.lower()
            if name.startswith(vpre.lower()):
                items.append({"path": str(p), "label": 1})
            elif name.startswith(npre.lower()):
                items.append({"path": str(p), "label": 0})
    else:
        raise SystemExit("Provide --violence-subdir/--nonviolence-subdir OR "
                         "--violence-prefix/--nonviolence-prefix")
    return items


def clip_probs(model, items, cfg, device, use_amp):
    from decord import VideoReader, cpu
    nf, sz = cfg["data"]["num_frames"], cfg["data"]["img_size"]
    nviews = cfg["eval"]["num_clips"]
    phases = (np.arange(nviews) + 0.5) / nviews
    short = int(round(sz * 1.14))
    y_true, y_prob, fails = [], [], 0
    model.eval()
    with torch.no_grad():
        for k, it in enumerate(items):
            try:
                vr = VideoReader(it["path"], ctx=cpu(0))
                total = len(vr)
                views = []
                for ph in phases:
                    frames = vr.get_batch(phase_indices(total, nf, ph)).asnumpy()
                    frames = _resize_short_side(frames, short)
                    frames = _crop(frames, sz, training=False)
                    x = frames.astype(np.float32) / 255.0
                    x = (x - MEAN) / STD
                    views.append(torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2))
                batch = torch.stack(views).to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(batch)
                p = torch.softmax(logits.float(), dim=1)[:, POS].mean().item()
            except Exception:  # noqa: BLE001
                fails += 1
                p = 0.0
            y_prob.append(p)
            y_true.append(it["label"])
            if (k + 1) % 50 == 0:
                print(f"  ...{k + 1}/{len(items)} clips")
    if fails:
        print(f"  WARNING: {fails} clip(s) failed to decode (scored 0.0)")
    return np.array(y_true), np.array(y_prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--violence-subdir")
    ap.add_argument("--nonviolence-subdir")
    ap.add_argument("--violence-prefix")
    ap.add_argument("--nonviolence-prefix")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    cfg = load_config()
    device = resolve_device(cfg)
    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"

    ckpt_path = ROOT / cfg["paths"]["checkpoints_root"] / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    items = build_items(args.root, args.violence_subdir, args.nonviolence_subdir,
                        args.violence_prefix, args.nonviolence_prefix)
    if not items:
        raise SystemExit(f"No videos found under {args.root} with the given layout.")
    n_v = sum(1 for it in items if it["label"] == 1)
    print(f"cross-dataset: {len(items)} clips ({n_v} violence, {len(items) - n_v} non-violence) "
          f"from {args.root}")
    print(f"device: {device}   views/clip: {cfg['eval']['num_clips']}   threshold: {args.threshold}")

    y_true, y_prob = clip_probs(model, items, cfg, device, use_amp)
    pred = (y_prob >= args.threshold).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true.tolist())) > 1 else float("nan")

    print("\n" + "=" * 60)
    print(f"CROSS-DATASET RESULTS  ({Path(args.root).name})")
    print("=" * 60)
    print(f"  acc {accuracy_score(y_true, pred):.4f}  "
          f"P {precision_score(y_true, pred, pos_label=POS, zero_division=0):.4f}  "
          f"R {recall_score(y_true, pred, pos_label=POS, zero_division=0):.4f}  "
          f"F1 {f1_score(y_true, pred, pos_label=POS, zero_division=0):.4f}  AUC {auc:.4f}")
    print(f"  confusion [rows=true 0/1, cols=pred 0/1]: {cm.tolist()}  "
          f"(missed {fn}, false alarms {fp})")
    print("\nRWF-2000 test reference: acc 0.8375, F1 0.8387, AUC 0.9269.")
    print("The gap below those numbers is your generalization drop -> report it honestly.")


if __name__ == "__main__":
    main()
