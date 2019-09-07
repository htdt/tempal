from dataclasses import dataclass
import torch
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv


@dataclass
class EnvRunnerRand:
    envs: ShmemVecEnv
    rollout_size: int
    device: str

    def __iter__(self):
        obs_shape = self.envs.observation_space.shape
        obs_dtype = torch.uint8 if len(obs_shape) == 3 else torch.float
        obs = torch.empty(
            self.rollout_size, self.envs.num_envs, *obs_shape,
            dtype=obs_dtype,
            device=self.device)

        step = 1
        obs[0] = self.envs.reset()

        while True:
            actions = [self.envs.action_space.sample()
                       for _ in range(self.envs.num_envs)]
            actions = torch.tensor(actions).unsqueeze(-1)
            obs[step] = self.envs.step(actions)[0]
            step = (step + 1) % self.rollout_size
            if step == 0:
                yield obs
