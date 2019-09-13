from collections import defaultdict
from dataclasses import dataclass
import random
import torch

from common.optim import ParamOptim
from common.tools import log_grads
from ppo.model import ActorCritic


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

    def _gae(self, rollout, last_val):
        m = rollout['masks'] * self.gamma
        r, v = rollout['rewards'], rollout['vals']
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

    def update(self, rollout, progress=0):
        clip = self.pi_clip * (1 - progress)
        num_step, num_env = rollout['log_probs'].shape[:2]
        with torch.no_grad():
            next_val = self.model(rollout['obs'][-1],
                rollout['obs_emb'][-1])[1]
        adv, returns = self._gae(rollout, next_val)

        logs, grads = defaultdict(list), defaultdict(list)

        num_samples = self.epochs * num_step * num_env
        idx1 = random.choices(range(num_step), k=num_samples)
        idx2 = random.choices(range(num_env), k=num_samples)

        for n_iter in range(num_samples // self.batch_size):
            s = slice(n_iter * self.batch_size, (n_iter + 1) * self.batch_size)
            idx = idx1[s], idx2[s]

            dist, vals = self.model(
                rollout['obs'][idx], rollout['obs_emb'][idx])
            act = rollout['actions'][idx].squeeze(-1)
            log_probs = dist.log_prob(act).unsqueeze(-1)
            ent = dist.entropy().mean()

            old_lp = rollout['log_probs'][idx]
            ratio = torch.exp(log_probs - old_lp)
            surr1 = adv[idx] * ratio
            surr2 = adv[idx] * torch.clamp(ratio, 1 - clip, 1 + clip)
            act_loss = -torch.min(surr1, surr2).mean()
            val_loss = .5 * (vals - returns[idx]).pow(2).mean()

            self.optim.step(-self.ent_k * ent + act_loss +
                            self.val_loss_k * val_loss)

            log_grads(self.model, grads)
            logs['ent'].append(ent)
            logs['clipfrac'].append(
                (torch.abs(ratio - 1) > clip).float().mean())
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
