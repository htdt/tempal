from dataclasses import dataclass
import torch
import numpy as np
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from ppo.model import ActorCritic
from encoders.base import BaseEncoder


@dataclass
class EnvRunner:
    envs: ShmemVecEnv
    model: ActorCritic
    rollout_size: int
    device: str

    encoder: BaseEncoder
    emb_size: int
    history_size: int

    ep_reward = []
    ep_len = []

    def get_logs(self):
        if len(self.ep_reward) >= self.envs.num_envs:
            res = {
                '_episode/reward': np.mean(self.ep_reward),
                '_episode/len': np.mean(self.ep_len),
            }
            self.ep_reward.clear(), self.ep_len.clear()
            return res
        else:
            return {}

    def __iter__(self):
        r, n = self.rollout_size, self.envs.num_envs

        def tensor(shape=(r, n, 1), dtype=torch.float):
            return torch.empty(*shape, dtype=dtype, device=self.device)

        obs_shape = self.envs.observation_space.shape
        obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float
        obs = tensor((r + 1, n, *obs_shape), dtype=obs_dtype)

        obs_emb = torch.zeros(r + 1, n, self.history_size, self.emb_size,
                              device=self.device)

        rewards = tensor()
        vals = tensor()
        log_probs = tensor()
        actions = tensor(dtype=torch.long)
        masks = tensor()

        step = 0
        obs[0] = self.envs.reset()
        with torch.no_grad():
            obs_emb[0, :, -1] = self.encoder(obs[0, :, -1:])

        while True:
            with torch.no_grad():
                dist, vals[step] = self.model(obs[step], obs_emb[step])
                a = dist.sample()
                actions[step] = a.unsqueeze(-1)
                log_probs[step] = dist.log_prob(a).unsqueeze(-1)

            obs[step + 1], rewards[step], terms, infos =\
                self.envs.step(actions[step])
            masks[step] = ~terms

            obs_emb[step + 1, :, :-1].copy_(obs_emb[step, :, 1:])
            obs_emb[step + 1] *= masks[step, ..., None]
            with torch.no_grad():
                obs_emb[step + 1, :, -1] = self.encoder(obs[step + 1, :, -1:])

            for info in infos:
                if 'episode' in info.keys():
                    self.ep_reward.append(info['episode']['r'])
                    self.ep_len.append(info['episode']['l'])

            step = (step + 1) % self.rollout_size
            if step == 0:
                yield {'obs': obs,
                       'obs_emb': obs_emb,
                       'rewards': rewards,
                       'vals': vals,
                       'log_probs': log_probs,
                       'actions': actions,
                       'masks': masks,
                       }
                obs[0].copy_(obs[-1])
                obs_emb[0].copy_(obs_emb[-1])
