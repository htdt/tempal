from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from common.optim import ParamOptim
from common.tools import Flatten, init_ortho
from encoders.iic import IID_loss
from encoders.base import BaseEncoder


@dataclass
class Hybrid(BaseEncoder):
    def __post_init__(self):
        self.encoder = Conv(emb_size=self.emb_size).to(self.device)
        self.classifier = nn.Linear(64, 64).to(self.device)
        self.encoder.train()
        self.classifier.train()
        self.target = torch.arange(self.batch_size).to(self.device)

        params = list(self.encoder.parameters()) +\
            list(self.classifier.parameters())
        self.optim = ParamOptim(lr=self.lr, params=params)

    def _step(self, x1, x2):
        x1_loc, x1 = self.encoder.blocks(x1)
        x2_loc, x2 = self.encoder.blocks(x2)
        loss1 = self._get_st_dim(x1_loc, x2_loc)
        loss2 = IID_loss(x1, x2)
        return self.optim.step(loss1 + loss2)

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
