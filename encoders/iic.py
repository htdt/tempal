import sys
from dataclasses import dataclass
from collections import deque
import torch
from torch import nn
from common.optim import ParamOptim
from common.tools import Flatten, init_ortho
from encoders.base import BaseEncoder


@dataclass
class IIC(BaseEncoder):
    def __post_init__(self):
        num_heads = 6
        self.encoder = Encoder(self.emb_size, num_heads).to(self.device)
        self.encoder.train()
        self.optim = ParamOptim(lr=self.lr, params=self.encoder.parameters())
        self.head_loss = [deque(maxlen=256) for _ in range(num_heads)]

    def _step(self, x1, x2):
        heads = zip(self.encoder.forward_heads(x1),
                    self.encoder.forward_heads(x2))
        losses = {}
        for i, (x1, x2) in enumerate(heads):
            losses[f'head_{i}'] = loss = IID_loss(x1, x2)
            self.head_loss[i].append(loss.item())
        loss_mean = sum(losses.values()) / len(losses)
        losses['mean'] = self.optim.step(loss_mean)
        return losses

    def select_head(self):
        x = torch.tensor(self.head_loss).mean(-1)
        x[0] = 1  # 0 head is auxiliary
        self.encoder.head_main = x.argmin().item()
        return self.encoder.head_main


class Encoder(nn.Module):
    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.emb_size = emb_size

        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
        self.base = nn.Sequential(
            init_ortho(nn.Conv2d(1, 32, 8, 4), 'relu'),
            nn.ReLU(),
            init_ortho(nn.Conv2d(32, 64, 4, 2), 'relu'),
            nn.ReLU(),
            init_ortho(nn.Conv2d(64, 64, 3, 1), 'relu'),
            nn.ReLU(),
            Flatten())

        self.heads = [nn.Sequential(
            nn.Linear(64 * 7 * 7, emb_size * (5 if i == 0 else 1)),
            nn.Softmax(-1))
            for i in range(num_heads)]

        for i, h in enumerate(self.heads):
            setattr(self, f'heads_{i}', h)

        self.head_main = None

    def forward_heads(self, x):
        x = self.base(x.float() / 255)
        return map(lambda h: h(x), self.heads)

    def forward(self, x):
        if self.head_main is None:
            return 0

        x = self.base(x.float() / 255)
        return self.heads[self.head_main](x)


# source
# https://github.com/xu-ji/IIC/blob/master/code/utils/cluster/IID_losses.py

def IID_loss(x_out, x_tf_out, lamb=1.0, EPS=sys.float_info.epsilon):
    # has had softmax applied
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j[(p_i_j < EPS).data] = EPS
    p_j[(p_j < EPS).data] = EPS
    p_i[(p_i < EPS).data] = EPS

    loss = - p_i_j * (torch.log(p_i_j)
                      - lamb * torch.log(p_j)
                      - lamb * torch.log(p_i))

    loss = loss.sum()
    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j
