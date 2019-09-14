from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from common.optim import ParamOptim
from common.tools import Flatten, init_ortho
from encoders.base import BaseEncoder


@dataclass
class STDIM(BaseEncoder):
    def __post_init__(self):
        num_layer = 64
        self.encoder = Conv(self.emb_size, num_layer).to(self.device)
        self.classifier1 = nn.Linear(self.emb_size, num_layer).to(self.device)
        self.classifier2 = nn.Linear(num_layer, num_layer).to(self.device)
        self.encoder.train()
        self.classifier1.train()
        self.classifier2.train()
        self.target = torch.arange(self.batch_size).to(self.device)

        params = list(self.encoder.parameters()) +\
            list(self.classifier1.parameters()) +\
            list(self.classifier2.parameters())
        self.optim = ParamOptim(lr=self.lr, params=params)

    def _step(self, x1, x2):
        x1_loc, x1_glob = self.encoder.forward_blocks(x1)
        x2_loc = self.encoder.forward_block1(x2)
        sy, sx = x1_loc.shape[2:]
        loss_loc, loss_glob = 0, 0
        for y in range(sy):
            for x in range(sx):
                positive = x2_loc[:, :, y, x]

                predictions = self.classifier1(x1_glob)
                logits = torch.matmul(predictions, positive.t())
                loss_glob += F.cross_entropy(logits, self.target)

                predictions = self.classifier2(x1_loc[:, :, y, x])
                logits = torch.matmul(predictions, positive.t())
                loss_loc += F.cross_entropy(logits, self.target)
        loss_loc /= (sx * sy)
        loss_glob /= (sx * sy)
        return {
            'sum': self.optim.step(loss_glob + loss_loc),
            'glob': loss_glob,
            'loc': loss_loc,
        }


class Conv(nn.Module):
    def __init__(self, emb_size, num_layer=64):
        super().__init__()
        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
        self.block1 = nn.Sequential(
            init_ortho(nn.Conv2d(1, num_layer // 2, 8, 4), 'relu'),
            nn.ReLU(),
            init_ortho(nn.Conv2d(num_layer // 2, num_layer, 4, 2), 'relu'))

        self.block2 = nn.Sequential(
            nn.ReLU(),
            init_ortho(nn.Conv2d(num_layer, num_layer, 3, 1), 'relu'),
            nn.ReLU(),
            Flatten(),
            init_ortho(nn.Linear(num_layer * 7 * 7, emb_size)))

    def forward_block1(self, x):
        x = x.float() / 255
        return self.block1(x)

    def forward_blocks(self, x):
        x = x.float() / 255
        b1 = self.block1(x)
        return b1, self.block2(b1)

    def forward(self, x):
        x = x.float() / 255
        return self.block2(self.block1(x))
