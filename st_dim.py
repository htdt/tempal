from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
import random
from common.optim import ParamOptim
from common.tools import Flatten, init_ortho


@dataclass
class STDIM:
    device: str = 'cpu'
    batch_size: int = 64
    emb_size: int = 64
    lr: float = 5e-4

    def __post_init__(self):
        self.encoder = Conv(emb_size=self.emb_size).to(self.device)
        self.classifier1 = nn.Linear(self.emb_size, 64).to(self.device)
        self.classifier2 = nn.Linear(64, 64).to(self.device)
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.target = torch.arange(self.batch_size).to(self.device)

        params = list(self.encoder.parameters()) +\
            list(self.classifier1.parameters()) +\
            list(self.classifier2.parameters())
        self.optim = ParamOptim(lr=self.lr, params=params)

    def update(self, obs, epochs=1):
        obs = obs[:, :, -1:]  # use one last layer out of 4
        losses = []
        num_step = epochs * obs.shape[0] * obs.shape[1]

        idx1 = random.choices(range(obs.shape[0] - 1), k=num_step)
        idx2 = list(map(lambda x: x + 1, idx1))
        idx_env = random.choices(range(obs.shape[1]), k=num_step)

        for i in range(num_step // self.batch_size):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x1 = obs[idx1[s], idx_env[s]]
            x2 = obs[idx2[s], idx_env[s]]
            loss = self.optim.step(self._get_loss(x1, x2))
            losses.append(loss.item())

        return {
            'loss/st_dim': sum(losses) / len(losses)
        }

    def _get_loss(self, x1, x2):
        x1_loc, x1_glob = self.encoder(x1)
        x2_loc = self.encoder(x2, only_block1=True)
        sy, sx = x1_loc.shape[2:]
        loss = 0
        for y in range(sy):
            for x in range(sx):
                positive = x2_loc[:, :, y, x]

                predictions = self.classifier1(x1_glob)
                logits = torch.matmul(predictions, positive.t())
                loss += F.cross_entropy(logits, self.target)

                predictions = self.classifier2(x1_loc[:, :, y, x])
                logits = torch.matmul(predictions, positive.t())
                loss += F.cross_entropy(logits, self.target)
        return loss / (sx * sy)


class Conv(nn.Module):
    def __init__(self, emb_size):
        super().__init__()

        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7 ->
        # 64 * 7 * 7 = 3136
        self.block1 = nn.Sequential(
            init_ortho(nn.Conv2d(1, 32, 8, 4), 'relu'),
            nn.ReLU(),
            init_ortho(nn.Conv2d(32, 64, 4, 2), 'relu'))

        self.block2 = nn.Sequential(
            nn.ReLU(),
            init_ortho(nn.Conv2d(64, 64, 3, 1), 'relu'),
            nn.ReLU(),
            Flatten(),
            init_ortho(nn.Linear(64 * 7 * 7, emb_size)))

    def forward(self, x, only_block1=False):
        x = x.float() / 255
        b1 = self.block1(x)
        if only_block1:
            return b1
        b2 = self.block2(b1)
        return b1, b2
