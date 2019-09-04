from collections import defaultdict
from dataclasses import dataclass
import random
import torch

from common.optim import ParamOptim
from common.tools import log_grads
from model import ActorCritic


@dataclass
class Agent:
    model: ActorCritic
    optim: ParamOptim
    pi_clip: float
    epochs: int
    batch_size: int
    val_loss_k: float
    ent_k: float
    gamma: float
    gae_lambda: float

    def _gae(self, rollout, next_val):
        m = rollout['masks'] * self.gamma
        r, v = rollout['rewards'], rollout['vals']
        adv, returns = torch.empty_like(v), torch.empty_like(v)
        gae = 0
        for i in reversed(range(adv.shape[0])):
            if i == adv.shape[0] - 1:
                next_return = next_val
            else:
                next_val = v[i + 1]
                next_return = returns[i + 1]

            delta = r[i] - v[i] + next_val * m[i]
            adv[i] = gae = delta + self.gae_lambda * m[i] * gae
            returns[i] = r[i] + next_return * m[i]

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, returns

    def update(self, rollout):
        num_step, num_env = rollout['log_probs'].shape[:2]
        with torch.no_grad():
            next_val = self.model(rollout['obs'][-1])[1]
        adv, returns = self._gae(rollout, next_val)

        logs, grads = defaultdict(list), defaultdict(list)
        for _ in range(self.epochs * num_step * num_env // self.batch_size):
            idx1d = random.sample(range(num_step * num_env), self.batch_size)
            idx = tuple(zip(*[(i % num_step, i // num_step) for i in idx1d]))

            dist, vals = self.model(rollout['obs'][idx])
            act = rollout['actions'][idx].squeeze(-1)
            log_probs = dist.log_prob(act).unsqueeze(-1)
            ent = dist.entropy().mean()

            old_lp = rollout['log_probs'][idx]
            ratio = torch.exp(log_probs - old_lp)
            surr1 = adv[idx] * ratio
            surr2 = adv[idx] * \
                torch.clamp(ratio, 1 - self.pi_clip, 1 + self.pi_clip)
            act_loss = -torch.min(surr1, surr2).mean()
            val_loss = .5 * (vals - returns[idx]).pow(2).mean()

            self.optim.step(-self.ent_k * ent + act_loss +
                            self.val_loss_k * val_loss)

            log_grads(self.model, grads)
            logs['ent'].append(ent)
            logs['clipfrac'].append(
                (torch.abs(ratio - 1) > self.pi_clip).float().mean())
            logs['loss/actor'].append(act_loss)
            logs['loss/critic'].append(val_loss)

        for name, val in grads.items():
            if '/max' in name:
                grads[name] = max(val)
            elif '/std' in name:
                grads[name] = sum(val) / (len(val) ** .5)
        return {
            'ent': torch.stack(logs['ent']).mean(),
            'clip/frac': torch.stack(logs['clipfrac']).mean(),
            'loss/actor': torch.stack(logs['loss/actor']).mean(),
            'loss/critic': torch.stack(logs['loss/critic']).mean(),
            **grads,
        }
