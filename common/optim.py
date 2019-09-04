from dataclasses import dataclass
from typing import List
import torch
from torch.optim import AdamW, Optimizer
from torchcontrib.optim import SWA


@dataclass
class ParamOptim:
    params: List[torch.Tensor]
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad: float = None
    optimizer: Optimizer = AdamW

    def __post_init__(self):
        base_opt = self.optimizer(self.params, lr=self.lr, eps=self.eps)
        self.optim = SWA(base_opt)

    def set_lr(self, lr):
        for pg in self.optim.param_groups:
            pg['lr'] = lr
        return lr

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.optim.step()
        return loss
