from dataclasses import dataclass
import torch
import numpy as np
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from ppo.model import ActorCritic


@dataclass
class EnvRunner:
    envs: ShmemVecEnv
    model: ActorCritic
    rollout_size: int
    device: str

    encoder: torch.nn.Module
    emb_size: int
    input_size: int

    rnd = True
    ep_reward = []
    ep_len = []

    def get_logs(self):
        if len(self.ep_reward) >= self.envs.num_envs:
            res = {
                "_episode/reward": np.mean(self.ep_reward),
                "_episode/len": np.mean(self.ep_len),
            }
            self.ep_reward.clear(), self.ep_len.clear()
            return res
        else:
            return {}

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
        with torch.no_grad():
            return self.model(x[:, -4:], x_emb)

    def __iter__(self):
        r, n = self.rollout_size, self.envs.num_envs

        def tensor(shape=(r, n, 1), dtype=torch.float):
            return torch.empty(*shape, dtype=dtype, device=self.device)

        obs_shape = self.envs.observation_space.shape
        obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float
        obs = tensor((r + 1, n, *obs_shape), dtype=obs_dtype)

        rewards = tensor()
        vals = tensor()
        log_probs = tensor()
        actions = tensor(dtype=torch.long)
        masks = tensor()

        obs[0] = self.envs.reset()
        step = 0

        while True:
            if self.rnd:
                a = [self.envs.action_space.sample() for _ in range(self.envs.num_envs)]
                actions[step] = torch.tensor(a).unsqueeze(-1)
            else:
                dist, vals[step] = self.model_enc(obs[step])
                a = dist.sample()
                actions[step] = a.unsqueeze(-1)
                log_probs[step] = dist.log_prob(a).unsqueeze(-1)

            obs[step + 1], rewards[step], terms, infos = self.envs.step(actions[step])
            masks[step] = ~terms

            for info in infos:
                if "episode" in info.keys():
                    self.ep_reward.append(info["episode"]["r"])
                    self.ep_len.append(info["episode"]["l"])

            step = (step + 1) % self.rollout_size
            if step == 0:
                yield {
                    "obs": obs,
                    "rewards": rewards,
                    "vals": vals,
                    "log_probs": log_probs,
                    "actions": actions,
                    "masks": masks,
                }
                obs[0].copy_(obs[-1])
