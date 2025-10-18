import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction="mean", label_smoothing=0.0):
        """
        gamma: focusing parameter (higher = more focus on hard samples)
        weight: optional per-class weights (for imbalance)
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: redistributes confidence slightly to other classes
        """
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, logits, target):
        # Step 1: log-softmax for numerical stability
        logp = F.log_softmax(logits, dim=1)

        # Step 2: optional label smoothing
        if self.label_smoothing > 0:
            n_classes = logits.size(1)
            with torch.no_grad():
                true_dist = torch.zeros_like(logp)
                true_dist.fill_(self.label_smoothing / (n_classes - 1))
                true_dist.scatter_(1, target.unsqueeze(1), 1 - self.label_smoothing)
            ce = -(true_dist * logp).sum(dim=1)  # smoothed cross-entropy
        else:
            # Standard CE without smoothing
            ce = F.nll_loss(logp, target, weight=self.weight, reduction="none")

        # Step 3: compute pt = probability of the true class
        pt = torch.exp(-ce)

        # Step 4: scale by focal factor (1 - pt)^gamma
        loss = (1 - pt) ** self.gamma * ce

        # Step 5: reduce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss