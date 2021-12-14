import torch
import torch.nn as nn
from torch.distributions import Categorical
from common.tools import init_ortho


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        emb_size: int,
        history_size: int,
        emb_fc_size: int,
        input_size: int = 4,
        hidden_size: int = 512,
    ):
        super(ActorCritic, self).__init__()
        self.output_size = output_size
        self.device = device

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU(True))

        # 84 x 84 -> 20 x 20 -> 9 x 9 -> 7 x 7
        self.body = nn.Sequential(
            with_relu(nn.Conv2d(input_size, 32, 8, 4)),
            with_relu(nn.Conv2d(32, 64, 4, 2)),
            with_relu(nn.Conv2d(64, 64, 3, 1)),
            nn.Flatten(),
            init_ortho(nn.Linear(64 * 7 * 7, hidden_size)),
            nn.ReLU(True),
        )

        self.emb_fc = nn.Sequential(
            nn.Flatten(),
            init_ortho(nn.Linear(emb_size * history_size, emb_fc_size)),
            nn.ReLU(True),
        )

        hidden_size = 0
        self.pi = init_ortho(nn.Linear(hidden_size + emb_fc_size, output_size), 0.01)
        self.val = init_ortho(nn.Linear(hidden_size + emb_fc_size, 1))

    def forward(self, x, x_emb):
        # x = self.body(x.float() / 255)
        x = self.emb_fc(x_emb)
        # x = torch.cat([x, x_emb], -1)
        return Categorical(logits=self.pi(x)), self.val(x)
