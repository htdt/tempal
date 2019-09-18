import torch
import torch.nn as nn
from torch.distributions import Categorical
from common.tools import Flatten, init_ortho


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        emb_size: int = 32,
        history_size: int = 64,
        emb_hidden_size: int = None,
        input_size: int = 4,
        hidden_size: int = 512,
    ):
        super(ActorCritic, self).__init__()
        self.output_size = output_size
        self.device = device

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU())

        self.conv = nn.Sequential(
            with_relu(nn.Conv2d(input_size, 32, 8, 4)),
            with_relu(nn.Conv2d(32, 64, 4, 2)),
            with_relu(nn.Conv2d(64, 64, 3, 1)),
            Flatten())
        conv_output = 64 * 7 * 7

        if emb_hidden_size is not None:
            self.emb_fc = nn.Sequential(
                Flatten(),
                with_relu(nn.Linear(emb_size * history_size, emb_hidden_size))
            )
            self.emb_output = emb_hidden_size
        else:
            self.emb_fc = Flatten()
            self.emb_output = emb_size * history_size

        self.fc = with_relu(
            nn.Linear(conv_output + self.emb_output, hidden_size))
        self.pi = init_ortho(nn.Linear(hidden_size, output_size), .01)
        self.val = init_ortho(nn.Linear(hidden_size, 1))

    def forward(self, x, x_emb):
        x = x.float() / 255.
        x = self.conv(x)

        if (x_emb == 0).all():
            x_emb = torch.zeros(x.shape[0], self.emb_output).to(self.device)
        else:
            x_emb = self.emb_fc(x_emb)

        x = torch.cat([x, x_emb], -1)
        x = self.fc(x)
        return Categorical(logits=self.pi(x)), self.val(x)
