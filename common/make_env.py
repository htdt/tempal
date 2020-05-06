import numpy as np
import random
import torch
import gym
from gym.spaces.box import Box

from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env import VecEnvWrapper


def make_vec_envs(name, num, seed=0):
    def make_env(rank):
        def _thunk():
            env = gym.make(name)
            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(name, max_episode_steps=10000)

            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            if is_atari:
                env = wrap_deepmind(env, frame_stack=False)
            return env
        return _thunk

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    envs = [make_env(i) for i in range(num)]
    envs = DummyVecEnv(envs) if num == 1 else ShmemVecEnv(envs, context='fork')
    envs = FrameStack(VecPyTorch(envs))
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, env):
        super(VecPyTorch, self).__init__(env)
        obs = self.observation_space.shape
        self.observation_space = Box(0, 255, [obs[2], obs[0], obs[1]],
                                     dtype=self.observation_space.dtype)

    def reset(self):
        return torch.from_numpy(self.venv.reset()).permute(0, 3, 1, 2)

    def step_async(self, actions):
        assert len(actions.shape) == 2
        self.venv.step_async(actions.squeeze(1).cpu().numpy())

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).permute(0, 3, 1, 2)
        reward = torch.from_numpy(reward).unsqueeze(dim=1)
        done = torch.tensor(done.tolist()).unsqueeze(dim=1)
        return obs, reward, done, info


class FrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack=4):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape, dtype=torch.uint8)

        observation_space = gym.spaces.Box(
            low=low, high=high, dtype=venv.observation_space.dtype
        )
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rews, news, infos = self.venv.step_wait()
        self.stacked_obs = torch.roll(self.stacked_obs, -1, 1)
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs, rews, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape, dtype=torch.uint8)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0 :] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
