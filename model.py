"""R(2+1)D-18 builder for binary violence detection.

Loads torchvision's r2plus1d_18 with Kinetics-400 pretrained weights and
swaps the final layer for a num_classes head. Input: (B, 3, 16, 112, 112).
Output: (B, num_classes) raw logits — use with CrossEntropyLoss.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


def _cfg_get(cfg: dict, key: str, default=None):
    """Look up a key at the top level or inside common config sections."""
    if key in cfg:
        return cfg[key]
    for section in ("model", "train", "data"):
        sub = cfg.get(section)
        if isinstance(sub, dict) and key in sub:
            return sub[key]
    return default


def build_model(cfg: dict) -> nn.Module:
    """Build R(2+1)D-18 with Kinetics weights and a fresh classification head."""
    num_classes = int(_cfg_get(cfg, "num_classes", 2))

    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model = r2plus1d_18(weights=weights)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze/unfreeze the backbone while always keeping the fc head trainable.

    Used in Phase 3b: freeze for `freeze_backbone_epochs`, then unfreeze.
    """
    for name, param in model.named_parameters():
        param.requires_grad = True if name.startswith("fc.") else trainable