from dataclasses import dataclass
from typing import List
import torch


@dataclass
class ParamOptim:
    params: List[torch.Tensor]
    lr: float = 1e-3
    eps: float = 1e-8
    clip_grad: float = None
    anneal: bool = True

    def __post_init__(self):
        self.optim = torch.optim.Adam(self.params, lr=self.lr, eps=self.eps)

    def set_lr(self, lr):
        for pg in self.optim.param_groups:
            pg['lr'] = lr
        return lr

    def update(self, progress):
        if self.anneal:
            self.set_lr(self.lr * (1 - progress))

    def step(self, loss):
        self.optim.zero_grad()
        loss.backward()
        if self.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(self.params, self.clip_grad)
        self.optim.step()
        return loss
