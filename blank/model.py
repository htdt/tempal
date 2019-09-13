import torch.nn as nn
from torch.distributions import Categorical
from common.tools import Flatten, init_ortho


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        input_size: int = 4,
        hidden_size: int = 512,
    ):
        super(ActorCritic, self).__init__()
        self.device = device

        def with_relu(m):
            return nn.Sequential(init_ortho(m, 'relu'), nn.ReLU())

        self.net = nn.Sequential(
            with_relu(nn.Conv2d(input_size, 32, 8, 4)),
            with_relu(nn.Conv2d(32, 64, 4, 2)),
            with_relu(nn.Conv2d(64, 64, 3, 1)),
            Flatten(),
            with_relu(nn.Linear(64 * 7 * 7, hidden_size)))

        self.pi = init_ortho(nn.Linear(hidden_size, output_size), .01)
        self.val = init_ortho(nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = x.float() / 255.
        x = self.net(x)
        return Categorical(logits=self.pi(x)), self.val(x)
