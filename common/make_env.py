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
                env = wrap_deepmind(env, frame_stack=True)
            return env
        return _thunk

    envs = [make_env(i) for i in range(num)]
    envs = DummyVecEnv(envs) if num == 1 else ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
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
