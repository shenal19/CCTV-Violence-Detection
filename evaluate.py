"""Phase 4: evaluate best.pt on the RWF-2000 test split with multi-clip averaging.

For each clip we sample cfg.eval.num_clips evenly-spaced whole-clip views, run
each, and average the softmax P(Violence) (test-time temporal averaging). We tune
the decision threshold on the validation split, then report TEST metrics at both
0.5 and the tuned threshold, including the explicit false-positive rate. Results
are saved to outputs/eval_metrics.json.

    python scripts/evaluate.py
"""
import json
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
from src.dataset import load_manifest, _resize_short_side, _crop, MEAN, STD  # noqa: E402
from src.model import build_model                                   # noqa: E402

POS = 1  # Violence


def resolve_device(cfg):
    want = str(cfg.get("project", {}).get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def phase_indices(total, num_frames, phase):
    """16 frame indices spanning the whole clip, offset by `phase` within each segment."""
    if total <= 0:
        return np.zeros(num_frames, dtype=int)
    seg = total / num_frames
    idx = ((np.arange(num_frames) + phase) * seg).astype(int)
    return np.minimum(idx, total - 1)


def clip_probs(model, items, cfg, device, use_amp):
    """Per-clip averaged P(Violence) via multi-clip temporal averaging."""
    from decord import VideoReader, cpu
    nf = cfg["data"]["num_frames"]
    sz = cfg["data"]["img_size"]
    n_views = cfg["eval"]["num_clips"]
    phases = (np.arange(n_views) + 0.5) / n_views   # e.g. 5 views -> 0.1,0.3,0.5,0.7,0.9
    short = int(round(sz * 1.14))

    y_true, y_prob, failures = [], [], 0
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
                    x = torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2)
                    views.append(x)
                batch = torch.stack(views).to(device)               # (V, C, T, H, W)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(batch)
                p = torch.softmax(logits.float(), dim=1)[:, POS].mean().item()
            except Exception:  # noqa: BLE001
                failures += 1
                p = 0.0
            y_prob.append(p)
            y_true.append(int(it["label"]))
            if (k + 1) % 50 == 0:
                print(f"  ...{k + 1}/{len(items)} clips")
    if failures:
        print(f"  WARNING: {failures} clip(s) failed to decode (scored 0.0)")
    return np.array(y_true), np.array(y_prob)


def metrics_at(y_true, y_prob, t, auc):
    pred = (y_prob >= t).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": round(float(t), 3),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, pos_label=POS, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, pos_label=POS, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, pos_label=POS, zero_division=0)), 4),
        "auc": round(float(auc), 4),
        "false_positive_rate": round(float(fp / (fp + tn)), 4) if (fp + tn) else 0.0,
        "confusion": cm.tolist(),
        "missed_fights": int(fn),
        "false_alarms": int(fp),
    }


def tune_threshold(y_true, y_prob):
    """Pick the threshold that maximizes F1 on the validation split."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.05):
        f = f1_score(y_true, (y_prob >= t).astype(int), pos_label=POS, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return round(float(best_t), 2)


def show(title, m):
    print(f"\n{title}  (threshold {m['threshold']})")
    print(f"  acc {m['accuracy']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
          f"F1 {m['f1']:.4f}  AUC {m['auc']:.4f}  FPR {m['false_positive_rate']:.4f}")
    print(f"  confusion [rows=true 0/1, cols=pred 0/1]: {m['confusion']}  "
          f"(missed fights {m['missed_fights']}, false alarms {m['false_alarms']})")


def main():
    cfg = load_config()
    device = resolve_device(cfg)
    use_amp = bool(cfg["train"]["amp"]) and device == "cuda"
    manifest = load_manifest()

    ckpt_path = ROOT / cfg["paths"]["checkpoints_root"] / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"loaded {ckpt_path.name} (best checkpoint from epoch {ckpt.get('epoch', '?')})")
    print(f"device: {device}   views/clip: {cfg['eval']['num_clips']}")

    print("\nscoring validation split (for threshold tuning)...")
    val_true, val_prob = clip_probs(model, manifest["val"], cfg, device, use_amp)
    print("scoring test split...")
    test_true, test_prob = clip_probs(model, manifest["test"], cfg, device, use_amp)

    t_tuned = tune_threshold(val_true, val_prob)
    val_auc = roc_auc_score(val_true, val_prob)
    test_auc = roc_auc_score(test_true, test_prob)

    test_default = metrics_at(test_true, test_prob, 0.5, test_auc)
    test_tuned = metrics_at(test_true, test_prob, t_tuned, test_auc)
    val_tuned = metrics_at(val_true, val_prob, t_tuned, val_auc)

    print("\n" + "=" * 62)
    print(f"RWF-2000 TEST RESULTS  (multi-clip avg, {cfg['eval']['num_clips']} views/clip)")
    print("=" * 62)
    show("test @ default 0.5", test_default)
    show("test @ tuned threshold", test_tuned)
    print(f"\n(threshold {t_tuned} tuned on validation; val F1 there {val_tuned['f1']:.4f})")

    out = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "num_views": cfg["eval"]["num_clips"],
        "tuned_threshold": t_tuned,
        "test_at_0.5": test_default,
        "test_at_tuned": test_tuned,
        "val_at_tuned": val_tuned,
    }
    out_path = ROOT / cfg["paths"]["outputs_root"] / "eval_metrics.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    main()
