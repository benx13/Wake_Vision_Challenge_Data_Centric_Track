import torch
import torch.nn as nn
import torch.nn.functional as F

class SCELoss(nn.Module):
    """
    Symmetric Cross Entropy Loss
    https://arxiv.org/abs/1908.06112
    """
    def __init__(self, alpha=1.0, beta=1.0, num_classes=2, smoothing=0.0):
        super(SCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.cross_entropy = nn.CrossEntropyLoss()
        self.register_buffer('uniform_labels', torch.ones(num_classes) / num_classes)

    def forward(self, pred, labels):
        # Move tensors to the same device as pred
        device = pred.device
        
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        
        # Create one-hot labels on GPU directly
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(device)
        
        if self.smoothing > 0.0:
            label_one_hot = label_one_hot * (1 - self.smoothing) + self.smoothing * self.uniform_labels.to(device)
            
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss