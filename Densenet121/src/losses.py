import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean", label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # standard CE with optional label smoothing to get log-probs
        logp = F.log_softmax(logits, dim=1)
        if self.label_smoothing > 0:
            # smooth CE
            n_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logp)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)
            ce = -(true_dist * logp).sum(dim=1)
        else:
            ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")

        pt = torch.exp(-ce)  # pt = softmax prob of the true class
        loss = (1 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss