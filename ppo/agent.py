from collections import defaultdict
from dataclasses import dataclass
import random
import numpy as np
import torch

from common.optim import ParamOptim
from ppo.model import ActorCritic


@dataclass
class Agent:
    model: ActorCritic
    encoder: torch.nn.Module
    optimizer: dict
    pi_clip: float
    epochs: int
    batch_size: int
    val_loss_k: float
    ent_k: float
    gamma: float
    gae_lambda: float
    anneal: bool = False

    def __post_init__(self):
        self.optim = ParamOptim(**self.optimizer, params=self.model.parameters())

    def _gae(self, rollout, last_val):
        m = rollout["masks"] * self.gamma
        r, v = rollout["rewards"], rollout["vals"]
        adv = torch.empty_like(v)
        n_steps = adv.shape[0]
        gae = 0
        for i in reversed(range(n_steps)):
            next_val = last_val if i == n_steps - 1 else v[i + 1]
            delta = r[i] - v[i] + next_val * m[i]
            adv[i] = gae = delta + self.gae_lambda * m[i] * gae

        returns = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def model_enc(self, obs):
        bs, steps, width, height = obs.shape
        x = obs.float() / 255
        if self.encoder is None:
            x_emb = None
        else:
            with torch.no_grad():
                x_emb = x.view(bs * steps, 1, width, height)
                x_emb = self.encoder(x_emb)
                x_emb = x_emb.view(bs, steps, x_emb.shape[-1])
        return self.model(x[:, -4:], x_emb)

    def update(self, rollout, progress=0):
        clip = self.pi_clip * (1 - progress * self.anneal)
        num_step, num_env = rollout["log_probs"].shape[:2]
        with torch.no_grad():
            next_val = self.model_enc(rollout["obs"][-1])[1]
        adv, returns = self._gae(rollout, next_val)

        logs = defaultdict(list)
        num_samples = self.epochs * num_step * num_env
        idx1 = random.choices(range(num_step), k=num_samples)
        idx2 = random.choices(range(num_env), k=num_samples)

        for n_iter in range(num_samples // self.batch_size):
            s = slice(n_iter * self.batch_size, (n_iter + 1) * self.batch_size)
            idx = idx1[s], idx2[s]

            dist, vals = self.model_enc(rollout["obs"][idx])
            act = rollout["actions"][idx].squeeze(-1)
            log_probs = dist.log_prob(act).unsqueeze(-1)
            ent = dist.entropy().mean()

            old_lp = rollout["log_probs"][idx]
            ratio = torch.exp(log_probs - old_lp)
            surr1 = adv[idx] * ratio
            surr2 = adv[idx] * torch.clamp(ratio, 1 - clip, 1 + clip)
            act_loss = -torch.min(surr1, surr2).mean()
            val_loss = 0.5 * (vals - returns[idx]).pow(2).mean()

            self.optim.step(-self.ent_k * ent + act_loss + self.val_loss_k * val_loss)

            logs["ent"].append(ent.item())
            logs["clipfrac"].append((torch.abs(ratio - 1) > clip).float().mean().item())
            logs["loss/actor"].append(act_loss.item())
            logs["loss/critic"].append(val_loss.item())

        return {k: np.mean(v) for k, v in logs.items()}
