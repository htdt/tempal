import random
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from common.tools import init_ortho
from common.optim import ParamOptim


def xent_loss(x0, x1, tau):
    bsize = x0.shape[0]
    target = torch.arange(bsize).cuda()
    eye_mask = torch.eye(bsize).cuda() * 1e9
    logits00 = x0 @ x0.t() / tau - eye_mask
    logits01 = x0 @ x1.t() / tau
    loss = F.cross_entropy(torch.cat([logits01, logits00], dim=1), target)
    acc = (logits01.argmax(-1) == target).float().mean().item()
    return loss, acc


@dataclass
class Xent:
    emb_size: int
    temporal_shift: int
    spatial_shift: int
    batch_size: int
    optimizer: dict
    tau: float
    rollouts_in_batch: int

    def __post_init__(self):
        self.encoder = Encoder(self.emb_size)
        self.encoder = self.encoder.train().cuda()
        params = list(self.encoder.parameters())
        self.optim = ParamOptim(params=params, **self.optimizer)

    def update(self, obs):
        def temporal(x):
            return x + random.randrange(1, self.temporal_shift + 1)

        def spatial():
            return random.randrange(-self.spatial_shift, self.spatial_shift + 1)

        idx0 = random.choices(
            range(obs.shape[0] - self.temporal_shift), k=self.batch_size
        )
        idx0_shift = list(map(temporal, idx0))
        if obs.shape[1] <= self.rollouts_in_batch:
            idx1_min = 0
            idx1_max = obs.shape[1]
        else:
            idx1_min = random.randrange(obs.shape[1] - self.rollouts_in_batch + 1)
            idx1_max = idx1_min + self.rollouts_in_batch
        idx1 = random.choices(range(idx1_min, idx1_max), k=self.batch_size)
        idx2 = random.choices(range(obs.shape[2]), k=self.batch_size)

        x0, x1 = obs[idx0, idx1, idx2], obs[idx0_shift, idx1, idx2]
        if self.spatial_shift > 0:
            for n in range(self.batch_size):
                for x in [x0, x1]:
                    shifts = spatial(), spatial()
                    x[n] = torch.roll(x[n], shifts=shifts, dims=(-2, -1))

        x0, x1 = x0.float() / 255, x1.float() / 255
        y0, y1 = self.encoder(x0), self.encoder(x1)
        loss0, acc = xent_loss(y0, y1, self.tau)
        loss1, _ = xent_loss(y1, y0, self.tau)
        loss = loss0 + loss1
        self.optim.step(loss)
        return {"xent/loss": loss.item(), "xent/acc": acc}


class Encoder(nn.Module):
    def __init__(self, size):
        super().__init__()

        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
        self.net = nn.Sequential(
            init_ortho(nn.Conv2d(1, 32, 8, 4), "relu"),
            nn.ReLU(True),
            init_ortho(nn.Conv2d(32, 64, 4, 2), "relu"),
            nn.ReLU(True),
            init_ortho(nn.Conv2d(64, 32, 3, 1), "relu"),
            nn.ReLU(True),
            nn.Flatten(),
            init_ortho(nn.Linear(32 * 7 * 7, size)),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1)
