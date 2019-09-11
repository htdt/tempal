from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import random
from common.optim import ParamOptim
from common.tools import Flatten, init_ortho
from iic import IID_loss


@dataclass
class Hybrid:
    emb_size: int
    n_step: int = 2
    device: str = 'cpu'
    batch_size: int = 64
    lr: float = 5e-4
    epochs: int = 1

    def __post_init__(self):
        self.encoder = Conv(emb_size=self.emb_size).to(self.device)
        self.classifier = nn.Linear(64, 64).to(self.device)
        self.encoder.train()
        self.classifier.train()
        self.target = torch.arange(self.batch_size).to(self.device)

        params = list(self.encoder.parameters()) +\
            list(self.classifier.parameters())
        self.optim = ParamOptim(lr=self.lr, params=params)

    def update(self, obs):
        obs = obs[:, :, -1:]  # use one last layer out of 4
        losses_stdim, losses_iic = [], []
        num_step = self.epochs * obs.shape[0] * obs.shape[1]

        def shift(x): return x + random.randrange(1, self.n_step + 1)
        idx1 = random.choices(range(obs.shape[0] - self.n_step), k=num_step)
        idx2 = list(map(shift, idx1))
        idx_env = random.choices(range(obs.shape[1]), k=num_step)

        for i in range(num_step // self.batch_size):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)

            x1_loc, x1 = self.encoder.blocks(obs[idx1[s], idx_env[s]])
            x2_loc, x2 = self.encoder.blocks(obs[idx2[s], idx_env[s]])

            loss1 = self._get_st_dim(x1_loc, x2_loc)
            loss2 = IID_loss(x1, x2)
            self.optim.step(loss1 + loss2)
            losses_stdim.append(loss1.item())
            losses_iic.append(loss2.item())

        return {
            'loss/iic': sum(losses_iic) / len(losses_iic),
            'loss/st_dim': sum(losses_stdim) / len(losses_stdim)
        }

    def _get_st_dim(self, x1_loc, x2_loc):
        sy, sx = x1_loc.shape[2:]
        loss = 0
        for y in range(sy):
            for x in range(sx):
                positive = x2_loc[:, :, y, x]
                predictions = self.classifier(x1_loc[:, :, y, x])
                logits = torch.matmul(predictions, positive.t())
                loss += F.cross_entropy(logits, self.target)
        return loss / (sx * sy)


class Conv(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        self.block1 = nn.Sequential(
            init_ortho(nn.Conv2d(1, 32, 8, 4), 'relu'),
            nn.ReLU(),
            init_ortho(nn.Conv2d(32, 64, 4, 2), 'relu'))

        self.block2 = nn.Sequential(
            nn.ReLU(),
            init_ortho(nn.Conv2d(64, 64, 3, 1), 'relu'),
            nn.ReLU(),
            Flatten(),
            init_ortho(nn.Linear(64 * 7 * 7, emb_size)),
            nn.Softmax(-1)
            )

    def blocks(self, x):
        x = x.float() / 255
        b1 = self.block1(x)
        return b1, self.block2(b1)

    def forward(self, x):
        x = x.float() / 255
        return self.block2(self.block1(x))
