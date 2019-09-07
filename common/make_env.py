import torch
import gym
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.atari_wrappers import EpisodicLifeEnv, ClipRewardEnv, FireResetEnv
import cv2
import numpy as np


class Grayscale(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        prev_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(prev_shape[0], prev_shape[1], 1),
            dtype=np.uint8,
        )

    def observation(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, -1)
        return obs

class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0], [
                obs_shape[self.op[0]], obs_shape[self.op[1]],
                obs_shape[self.op[2]]
            ],
            dtype=self.observation_space.dtype)

    def observation(self, ob):
        return ob.transpose(self.op[0], self.op[1], self.op[2])

def make_vec_envs(name, num, seed=0):
    def make_env(rank):
        def _thunk():
            env = gym.make(name)
            is_atari = hasattr(gym.envs, 'atari') and isinstance(
                env.unwrapped, gym.envs.atari.atari_env.AtariEnv)
            if is_atari:
                env = make_atari(name)

            env.seed(seed + rank)
            env = bench.Monitor(env, None)
            if is_atari:
                env = EpisodicLifeEnv(env)
                if 'FIRE' in env.unwrapped.get_action_meanings():
                    env = FireResetEnv(env)
                env = ClipRewardEnv(env)
                env = Grayscale(env)
                env = TransposeImage(env, op=[2, 0, 1])
                # env = wrap_deepmind(env, frame_stack=True)
            return env
        return _thunk

    envs = [make_env(i) for i in range(num)]
    envs = DummyVecEnv(envs) if num == 1 else ShmemVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
    return envs


class VecPyTorch(VecEnvWrapper):
    def reset(self):
        return torch.from_numpy(self.venv.reset())

    def step_async(self, actions):
        assert len(actions.shape) == 2
        self.venv.step_async(actions.squeeze(1).cpu().numpy())

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs)
        reward = torch.from_numpy(reward).unsqueeze(dim=1)
        done = torch.tensor(done.tolist()).unsqueeze(dim=1)
        return obs, reward, done, info
