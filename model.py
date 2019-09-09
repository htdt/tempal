import torch
import torch.nn as nn
from torch.distributions import Categorical
from common.cfg import find_checkpoint
from common.tools import Flatten, init_ortho


class ActorCritic(nn.Module):
    def __init__(
        self,
        output_size: int,
        device: str,
        emb_size: int,
        emb_stack: int,
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

        self.fc_emb = nn.Sequential(
            Flatten(), with_relu(nn.Linear(emb_size * emb_stack, 512)))

        input_size = 64 * 7 * 7 + 512
        self.fc = with_relu(nn.Linear(input_size, hidden_size))
        self.pi = init_ortho(nn.Linear(hidden_size, output_size), .01)
        self.val = init_ortho(nn.Linear(hidden_size, 1))

    def forward(self, x, x_emb):
        x = x.float() / 255.
        x = self.conv(x)

        x_emb = x_emb.permute(1, 0, 2).contiguous()
        x_emb = self.fc_emb(x_emb)

        x = torch.cat([x, x_emb], -1)
        x = self.fc(x)
        return Categorical(logits=self.pi(x)), self.val(x)


def init_model(cfg, env, device, resume):
    model = ActorCritic(
        output_size=env.action_space.n,
        device=device,
    ).train()
    model.to(device=device)

    if resume:
        n_start, fname = find_checkpoint(cfg)
        if fname:
            model.load_state_dict(torch.load(fname, map_location=device))
    else:
        n_start = 0

    return model, n_start
