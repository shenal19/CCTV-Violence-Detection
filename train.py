"""Phase 3b: train R(2+1)D-18 on RWF-2000 (official split).

What it does:
  - AMP mixed-precision training (safe on the 4GB card).
  - Freezes the backbone for cfg.train.freeze_backbone_epochs so the new head
    warms up first, then unfreezes and fine-tunes the whole network.
  - Validates every epoch: precision / recall / F1 / AUC for the positive class
    (Violence = index 1), plus a confusion matrix, via scikit-learn.
  - Early-stops on validation F1 (patience = cfg.train.early_stopping_patience).
  - Saves the best model to outputs/checkpoints/best.pt and a CSV to outputs/logs/.

Fast sanity pass (a couple of epochs, a handful of batches):
    python scripts/train.py --epochs 2 --max-batches 10
Full run:
    python scripts/train.py
"""
import argparse
import csv
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import torch.nn as nn                                                 # noqa: E402
from sklearn.metrics import (                                         # noqa: E402
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)

from src.config import ROOT, load_config                             # noqa: E402
from src.dataset import build_dataloaders, load_manifest             # noqa: E402
from src.model import build_model, set_backbone_trainable            # noqa: E402

POS = 1  # positive class = Violence (config class_names = ["NonViolence", "Violence"])


def resolve_device(cfg):
    want = str(cfg.get("project", {}).get("device", "auto")).lower()
    if want == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def label_counts(manifest, split):
    labels = [int(it["label"]) for it in manifest[split]]
    return {0: labels.count(0), 1: labels.count(1)}


@torch.no_grad()
def run_validation(model, val_dl, device, use_amp):
    model.eval()
    probs, preds, true = [], [], []
    for x, y in val_dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
        p = torch.softmax(logits.float(), dim=1)[:, POS]
        probs.append(p.cpu().numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
        true.append(y.numpy())
    y_true = np.concatenate(true)
    y_pred = np.concatenate(preds)
    y_prob = np.concatenate(probs)
    m = {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POS, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POS, zero_division=0),
    }
    try:
        m["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        m["auc"] = float("nan")  # only one class present in this val batch set
    m["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override cfg.train.epochs")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="cap train batches per epoch (fast sanity runs)")
    args = ap.parse_args()

    cfg = load_config()
    seed_everything(cfg["project"]["seed"])
    torch.backends.cudnn.benchmark = True  # fixed input size -> faster after warmup

    device = resolve_device(cfg)
    tcfg = cfg["train"]
    epochs = args.epochs if args.epochs is not None else tcfg["epochs"]
    freeze_epochs = tcfg["freeze_backbone_epochs"]
    patience = tcfg["early_stopping_patience"]
    use_amp = bool(tcfg["amp"]) and device == "cuda"

    ckpt_dir = ROOT / cfg["paths"]["checkpoints_root"]
    log_dir = ROOT / cfg["paths"]["logs_root"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    log_path = log_dir / "train_log.csv"

    manifest = load_manifest()
    print(f"device: {device}   amp: {use_amp}   epochs: {epochs}")
    print(f"train labels {label_counts(manifest, 'train')}   "
          f"val labels {label_counts(manifest, 'val')}   (0=NonViolence, 1=Violence)")

    train_dl, val_dl, _ = build_dataloaders(cfg, manifest)
    print(f"train batches/epoch: {len(train_dl)}   val batches: {len(val_dl)}")

    model = build_model(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    crit = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    backbone_frozen = freeze_epochs > 0
    set_backbone_trainable(model, not backbone_frozen)
    if backbone_frozen:
        print(f"backbone FROZEN for first {freeze_epochs} epoch(s) (head warmup)")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_acc", "val_precision",
                                "val_recall", "val_f1", "val_auc", "backbone"])

    best_f1, no_improve = -1.0, 0

    for epoch in range(1, epochs + 1):
        if backbone_frozen and epoch > freeze_epochs:
            set_backbone_trainable(model, True)
            backbone_frozen = False
            print(f"[epoch {epoch}] backbone UNFROZEN - fine-tuning full network")

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running, seen, t0 = 0.0, 0, time.time()
        for bi, (x, y) in enumerate(train_dl):
            if args.max_batches is not None and bi >= args.max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = crit(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(seen, 1)
        m = run_validation(model, val_dl, device, use_amp)
        dt = time.time() - t0
        bb = "frozen" if backbone_frozen else "full"
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3) if device == "cuda" else 0.0

        print(f"epoch {epoch:2d}/{epochs}  loss {train_loss:.4f}  "
              f"val_f1 {m['f1']:.4f}  P {m['precision']:.4f}  R {m['recall']:.4f}  "
              f"AUC {m['auc']:.4f}  acc {m['acc']:.4f}  "
              f"[{bb}]  {dt:.0f}s  peakVRAM {peak:.2f}GB")
        print(f"           confusion [rows=true 0/1, cols=pred 0/1] "
              f"(cm[1,0]=missed fights, cm[0,1]=false alarms):\n{m['cm']}")

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.4f}", f"{m['acc']:.4f}",
                                    f"{m['precision']:.4f}", f"{m['recall']:.4f}",
                                    f"{m['f1']:.4f}", f"{m['auc']:.4f}", bb])

        if m["f1"] > best_f1:
            best_f1, no_improve = m["f1"], 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_metrics": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                for k, v in m.items()},
                "config": cfg,
                "class_names": cfg["data"]["class_names"],
                "pos_index": POS,
            }, best_path)
            print(f"           [BEST] val_f1 {best_f1:.4f} - saved {best_path.name}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"early stopping: no val_f1 gain in {patience} epochs")
                break

    print(f"\ndone. best val_f1 {best_f1:.4f}  ->  {best_path}")


if __name__ == "__main__":
    main()
