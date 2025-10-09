# src/models/resnet18_multitask.py
from __future__ import annotations
import torch
import torch.nn as nn

# Handle both old/new torchvision APIs for pretrained weights
try:
    from torchvision.models import resnet18, ResNet18_Weights

    _HAS_WEIGHTS_ENUM = True
except Exception:
    from torchvision.models import resnet18  # type: ignore

    _HAS_WEIGHTS_ENUM = False


class ResNet18MultiTask(nn.Module):
    def __init__(self, num_classes_grade: int = 5, pretrained: bool = True):
        super().__init__()
        if _HAS_WEIGHTS_ENUM:
            base = resnet18(
                weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            )
        else:
            base = resnet18(pretrained=pretrained)

        # Keep everything up to the global avgpool
        self.features = nn.Sequential(*list(base.children())[:-1])
        in_feats = base.fc.in_features

        # Two heads: (1) 5-class grade, (2) binary edema (logit)
        self.head_grade = nn.Linear(in_feats, num_classes_grade)
        self.head_edema = nn.Linear(in_feats, 1)

    def forward(self, x: torch.Tensor):
        x = self.features(x)  # [B, 512, 1, 1]
        x = torch.flatten(x, 1)  # [B, 512]
        logits_grade = self.head_grade(x)
        logit_edema = self.head_edema(x).squeeze(1)  # [B]
        return logits_grade, logit_edema
