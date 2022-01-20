import torch.nn as nn
from torch.distributions import Categorical
from common.tools import init_ortho


class ActorCriticHistory(nn.Module):
    def __init__(self, num_obs, obs_size, obs_hidden, history_fc, num_action, **kwargs):
        super().__init__()

        def net(out, gain):
            return nn.Sequential(
                init_ortho(nn.Conv1d(num_obs, obs_hidden, 1), "relu"),
                nn.ReLU(True),
                nn.Flatten(),
                init_ortho(nn.Linear(obs_hidden * obs_size, history_fc), "relu"),
                nn.ReLU(True),
                init_ortho(nn.Linear(history_fc, out), gain),
            )

        self.pi = net(num_action, 0.01)
        self.val = net(1, 1)

    def forward(self, x, x_emb):
        return Categorical(logits=self.pi(x_emb)), self.val(x_emb)


class ActorCriticInstant(nn.Module):
    def __init__(self, instant_fc, num_action, **kwargs):
        super().__init__()
        self.encoder = nn.Sequential(
            init_ortho(nn.Conv2d(4, 32, 8, 4), "relu"),
            nn.ReLU(True),
            init_ortho(nn.Conv2d(32, 64, 4, 2), "relu"),
            nn.ReLU(True),
            init_ortho(nn.Conv2d(64, 64, 3, 1), "relu"),
            nn.ReLU(True),
            nn.Flatten(),
            init_ortho(nn.Linear(64 * 7 * 7, instant_fc), "relu"),
            nn.ReLU(True),
        )
        self.val = init_ortho(nn.Linear(instant_fc, 1))
        self.pi = init_ortho(nn.Linear(instant_fc, num_action), 0.01)

    def forward(self, x, x_emb):
        x = self.encoder(x)
        return Categorical(logits=self.pi(x)), self.val(x)


class ActorCritic(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.history = ActorCriticHistory(**kwargs)
        self.instant = ActorCriticInstant(**kwargs)

    def forward(self, x, x_emb):
        pi_h = self.history.pi(x_emb)
        val_h = self.history.val(x_emb)
        x = self.instant.encoder(x)
        pi_i = self.instant.pi(x)
        val_i = self.instant.val(x)
        return Categorical(logits=(pi_h + pi_i) / 2), (val_h + val_i) / 2
