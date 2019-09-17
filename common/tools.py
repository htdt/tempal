from typing import List, Dict
import torch
import torch.nn as nn


def lerp_nn(source: nn.Module, target: nn.Module, tau: float):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.copy_(t.data * (1. - tau) + s.data * tau)


def flat_grads(params):
    x = [p.grad.data.flatten() for p in params if p.grad is not None]
    return torch.cat(x) if len(x) else None


def log_grads(model, outp: Dict[str, List[float]]):
    for name, net in dict(model.named_children()).items():
        fg = flat_grads(net.parameters())
        if fg is not None:
            outp[f'grad/{name}/max'].append(fg.max().item())
            outp[f'grad/{name}/std'].append(fg.std().item())


def onehot(x, num):
    r = [1] * (len(x.shape) - 1) + [num]
    return torch.zeros_like(x).float().repeat(*r).scatter(-1, x, 1)


class Identity(torch.nn.Module):
    def forward(self, x): return x


class Flatten(nn.Module):
    def forward(self, x): return x.view(x.size(0), -1)


def init_ortho(module, gain=1):
    if isinstance(gain, str):
        gain = nn.init.calculate_gain(gain)
    nn.init.orthogonal_(module.weight.data, gain=gain)
    nn.init.constant_(module.bias.data, 0)
    return module


def init_ortho_multi(module):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param)
