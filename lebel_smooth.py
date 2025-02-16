import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing CrossEntropy Loss.

    Args:
        smoothing (float): Smoothing factor for labels (default: 0.1).
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'mean'
    """
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        assert 0 <= smoothing < 1
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs (torch.Tensor): Logits of the model (before softmax). Shape (N, C) where N is batch size, C is number of classes.
            targets (torch.Tensor): Ground truth labels. Shape (N) or (N,).

        Returns:
            torch.Tensor: Loss value.
        """
        log_probs = F.log_softmax(inputs, dim=-1)
        n_classes = inputs.size(-1)

        # Convert targets to one-hot if they are class indices
        if targets.dtype in [torch.int64, torch.int32, torch.long]:
            targets_one_hot = torch.zeros_like(inputs).scatter_(-1, targets.unsqueeze(-1), 1)
        else:
            targets_one_hot = targets # Assume targets are already one-hot or probabilities

        smooth_targets = (1 - self.smoothing) * targets_one_hot + self.smoothing / n_classes

        loss = (-smooth_targets * log_probs).sum(dim=-1)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")