from dataclasses import dataclass
import random
from collections import defaultdict


@dataclass
class BaseEncoder:
    emb_size: int
    n_step: int
    batch_size: int
    lr: float
    epochs: int
    device: str = 'cpu'

    def update(self, obs):
        obs = obs[:, :, -1:]  # last frame out of 4
        num_step = self.epochs * obs.shape[0] * obs.shape[1]

        def shift(x): return x + random.randrange(1, self.n_step + 1)
        idx1 = random.choices(range(obs.shape[0] - self.n_step), k=num_step)
        idx2 = list(map(shift, idx1))
        idx_env = random.choices(range(obs.shape[1]), k=num_step)

        losses = defaultdict(list)
        for i in range(num_step // self.batch_size):
            s = slice(i * self.batch_size, (i + 1) * self.batch_size)
            x1 = obs[idx1[s], idx_env[s]]
            x2 = obs[idx2[s], idx_env[s]]

            loss = self._step(x1, x2)
            for k, v in loss.items():
                losses[k].append(v.item())

        return {f'encoder/{k}': sum(v) / len(v)
                for k, v in losses.items()}

    def _step(self, x1, x2):
        raise NotImplementedError
