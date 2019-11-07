import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.nn.functional import softmax
from common.tools import Flatten, init_ortho, init_ortho_multi


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        emb_size: int,
        history_size: int = 32,
        input_size: int = 4,
        hidden_size: int = 512,
        mode: str = 'rnn',
    ):
        super(ActorCritic, self).__init__()
        assert mode in ['rnn', 'concat', 'no_i']
        self.mode = mode
        self.output_size = output_size
        self.device = device

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU())

        self.conv1 = nn.Sequential(
            with_relu(nn.Conv2d(input_size, 32, 8, 4)),
            with_relu(nn.Conv2d(32, 64, 4, 2)),
            with_relu(nn.Conv2d(64, 64, 3, 1)),
            Flatten())

        self.conv2 = nn.Sequential(
            with_relu(nn.Conv2d(1, 32, 8, 4)),
            with_relu(nn.Conv2d(32, 64, 4, 2)),
            with_relu(nn.Conv2d(64, 64, 3, 1)),
            Flatten(),
            init_ortho(nn.Linear(64 * 7 * 7, emb_size))
        )

        conv_output = 64 * 7 * 7

        self.rnn = nn.GRU(emb_size, emb_size * 2, batch_first=True)
        init_ortho_multi(self.rnn)

        if mode == 'rnn':
            fc_size = conv_output + emb_size * 2
        elif mode == 'concat':
            fc_size = emb_size * history_size
        elif mode == 'no_i':
            fc_size = emb_size * 2

        self.fc = with_relu(nn.Linear(fc_size, hidden_size))
        self.pi = init_ortho(nn.Linear(hidden_size, output_size), .01)
        self.val = init_ortho(nn.Linear(hidden_size, 1))

    def forward(self, x, x_emb):
        x_emb = x_emb.float() / 255.
        bsize, frames = x_emb.shape[:2]
        x_emb = x_emb.view(bsize * frames, 1, 84, 84)
        x_emb = self.conv2(x_emb)
        x_emb = x_emb.view(bsize, frames, -1)

        if self.mode == 'concat':
            x_emb = softmax(x_emb, -1)
            x_emb = x_emb.view(bsize, -1)
        else:
            x_emb = self.rnn(x_emb)[0][:, -1]

        if self.mode != 'no_i':
            x = x.float() / 255.
            x = self.conv1(x)
            x = torch.cat([x, x_emb], -1)
        x = self.fc(x)
        return Categorical(logits=self.pi(x)), self.val(x)
