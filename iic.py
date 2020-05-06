import random
import sys
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.functional import cross_entropy, normalize
from torch.distributions import Categorical
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
    aug: bool = True

    def __post_init__(self):
        self.encoder = Encoder(self.emb_size, self.emb_size_aux, True)
        self.encoder = self.encoder.train().cuda()
        self.optim = ParamOptim(lr=self.lr, params=self.encoder.parameters())
        self.target_nce = torch.arange(self.batch_size).cuda()

    def update(self, obs, need_stat):
        def shift(x):
            return x + random.randrange(1, self.n_step + 1)

        idx0 = random.choices(range(obs.shape[0] - self.n_step), k=self.batch_size)
        idx0_shift = list(map(shift, idx0))
        idx1 = random.choices(range(obs.shape[1]), k=self.batch_size)
        idx2 = random.choices(range(obs.shape[2]), k=self.batch_size)

        x0, x1 = obs[idx0, idx1, idx2], obs[idx0_shift, idx1, idx2]
        if self.aug:
            for n in range(self.batch_size):
                for x in [x0, x1]:
                    shifts = random.randrange(-9, 10), random.randrange(-9, 10)
                    x[n] = torch.roll(x[n], shifts=shifts, dims=(-2, -1))

        x0h0, x0h1 = self.encoder(x0, True)
        x1h0, x1h1 = self.encoder(x1, True)

        loss_iic = IID_loss(x0h0, x1h0, lamb=1.1)

        x0h1, x1h1 = normalize(x0h1, p=2, dim=1), normalize(x1h1, p=2, dim=1)
        logits = [x0h1 @ x1h1.t(), x0h1 @ x0h1.t(), x1h1 @ x1h1.t()]
        logits = torch.cat(logits, dim=1)
        loss_nce = cross_entropy(logits, self.target_nce)

        self.optim.step(loss_iic + loss_nce)

        if not need_stat:
            return {}

        clusters = x0h0.detach().argmax(-1).unique(return_counts=True)
        ent = Categorical(probs=clusters[1].float()).entropy()
        nce_acc = (
            (logits[:, : self.batch_size].argmax(-1) == self.target_nce)
            .float()
            .mean()
            .item()
        )
        iic_acc = (x0h0.argmax(-1) == x1h0.argmax(-1)).float().mean().item()
        return {
            "loss_iic": loss_iic.item(),
            "loss_nce": loss_nce.item(),
            "clusters": len(clusters[0]),
            "ent_iic": ent.item(),
            "nce_acc": nce_acc,
            "iic_acc": iic_acc,
        }


class Encoder(nn.Module):
    def __init__(self, size, size_aux, mini=False):
        super().__init__()

        if mini:
            # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
            self.base = nn.Sequential(
                init_ortho(nn.Conv2d(1, 32, 8, 4), "relu"),
                nn.ReLU(),
                init_ortho(nn.Conv2d(32, 64, 4, 2), "relu"),
                nn.ReLU(),
                init_ortho(nn.Conv2d(64, 64, 3, 1), "relu"),
                nn.ReLU(),
                nn.Flatten(),
                init_ortho(nn.Linear(64 * 7 * 7, 512), "relu"),
                nn.ReLU(),
            )
        else:
            self.base = resnet18()
            self.base.fc = nn.Identity()
            self.base.conv1 = nn.Conv2d(1, 64, 3, 1, 1, bias=False)

        self.head = nn.Sequential(nn.Linear(512, size), nn.Softmax(-1))
        self.head_aux = nn.Linear(512, size_aux)

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
