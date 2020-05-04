import random
import sys
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from torch.nn.functional import cross_entropy, normalize
from torchvision.models import resnet18

from common.optim import ParamOptim
from common.tools import init_ortho


@dataclass
class IIC:
    emb_size: int
    emb_size_aux: int
    n_step: int
    batch_size: int
    lr: float
    epochs: int
    device: str = "cuda"
    rolls: bool = False

    def __post_init__(self):
        self.encoder = Encoder(self.emb_size, self.emb_size_aux).to(self.device)
        self.encoder.train()
        self.optim = ParamOptim(lr=self.lr, params=self.encoder.parameters())
        self.target_nce = torch.arange(self.batch_size).cuda()

    def update(self, obs):
        obs = obs[:, :, -1:]  # last frame out of 4
        num_step = self.epochs * obs.shape[0] * obs.shape[1]

        def shift(x):
            return x + random.randrange(1, self.n_step + 1)

        idx0 = random.choices(range(obs.shape[0] - self.n_step), k=num_step)
        idx1 = list(map(shift, idx0))
        idx_env = random.choices(range(obs.shape[1]), k=num_step)

        losses_iic, losses_nce, nce_acc, clusters = [], [], [], []
        for i in range(num_step // self.batch_size):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x0 = obs[idx0[s], idx_env[s]]
            x1 = obs[idx1[s], idx_env[s]]

            x0h0, x0h1 = self.encoder(x0, True)
            x1h0, x1h1 = self.encoder(x1, True)
            loss_iic = IID_loss(x0h0, x1h0)

            x0h1, x1h1 = normalize(x0h1, p=2, dim=1), normalize(x1h1, p=2, dim=1)
            logits = [x0h1 @ x1h1.t(), x0h1 @ x0h1.t(), x1h1 @ x1h1.t()]
            logits = torch.cat(logits, dim=1)
            loss_nce = cross_entropy(logits, self.target_nce)

            self.optim.step(loss_iic + loss_nce)
            losses_iic.append(loss_iic.item())
            losses_nce.append(loss_nce.item())
            nce_acc.append(
                (logits[:, : self.batch_size].argmax(-1) == self.target_nce)
                .float()
                .mean()
                .item()
            )
            clusters.append(len(x0h0.argmax(-1).unique()))

        return {
            "loss/iic": np.mean(losses_iic),
            "loss/nce": np.mean(losses_nce),
            "nce_acc": np.mean(nce_acc),
            "clusters": np.mean(clusters),
        }


class Encoder(nn.Module):
    def __init__(self, size, size_aux):
        super().__init__()

        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
        # self.base = nn.Sequential(
        #     init_ortho(nn.Conv2d(1, 32, 8, 4), "relu"),
        #     nn.ReLU(),
        #     init_ortho(nn.Conv2d(32, 64, 4, 2), "relu"),
        #     nn.ReLU(),
        #     init_ortho(nn.Conv2d(64, 64, 3, 1), "relu"),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        # size_out = 64 * 7 * 7
        self.base = resnet18()
        self.base.fc = nn.Identity()
        self.base.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)
        size_out = 512

        self.head = nn.Sequential(nn.Linear(size_out, size), nn.Softmax(-1))
        self.head_aux = nn.Linear(size_out, size_aux)

    def forward(self, x, aux=False):
        x = self.base(x.float() / 255)
        if aux:
            return self.head(x), self.head_aux(x)
        else:
            return self.head(x)


def IID_loss(x_out, x_tf_out, lamb=1.0):
    EPS = sys.float_info.epsilon
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)
    assert p_i_j.size() == (k, k)

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
    # but should be same, symmetric

    # avoid NaN losses. Effect will get cancelled out by p_i_j tiny anyway
    p_i_j = torch.where(p_i_j < EPS, torch.ones_like(p_i_j) * EPS, p_i_j)
    p_j = torch.where(p_j < EPS, torch.ones_like(p_j) * EPS, p_j)
    p_i = torch.where(p_i < EPS, torch.ones_like(p_i) * EPS, p_i)

    loss = -p_i_j * (torch.log(p_i_j) - lamb * torch.log(p_j) - lamb * torch.log(p_i))

    loss = loss.sum()

    if lamb != 1.0:
        loss_no_lamb = -p_i_j * (torch.log(p_i_j) - torch.log(p_j) - torch.log(p_i))
        loss_no_lamb = loss_no_lamb.sum()
    else:
        loss_no_lamb = loss

    return loss


def compute_joint(x_out, x_tf_out):
    # produces variable that requires grad (since args require grad)

    bn, k = x_out.size()
    assert x_tf_out.size(0) == bn and x_tf_out.size(1) == k

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.0  # symmetrise
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j
