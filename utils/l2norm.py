import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float):
        super(L2Norm, self).__init__()
        self.n_channels: int = n_channels
        self.gamma: Optional[float] = scale or None
        self.eps: float = 1e-10
        self.weight: nn.Parameter = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x: torch.Tensor):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out
