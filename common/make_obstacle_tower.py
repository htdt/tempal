import numpy as np
import gym
from gym.spaces.box import Box
from gym.spaces.discrete import Discrete

from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.atari_wrappers import FrameStack
from common.make_env import VecPyTorch
try:
    from obstacle_tower_env import ObstacleTowerEnv
except ModuleNotFoundError:
    ObstacleTowerEnv = None

import cv2
cv2.ocl.setUseOpenCL(False)

HUMAN_ACTIONS = (0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33)


class OTWrapper(gym.Wrapper):
    def __init__(self, env):
        super(OTWrapper, self).__init__(env)
        self.observation_space = Box(low=0, high=255, dtype=np.uint8,
                                     shape=(84, 84, 1))
        self.action_space = Discrete(len(HUMAN_ACTIONS))

    def _greyscale(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = np.expand_dims(obs, -1)
        return obs

    def step(self, action):
        action_env = HUMAN_ACTIONS[action]
        obs, reward, done, info = self.env.step(action_env)
        obs = self._greyscale(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs = self._greyscale(obs)
        return obs


def make_obstacle_tower(num, seed=0, show=False):
    assert ObstacleTowerEnv is not None,\
        'install https://github.com/Unity-Technologies/obstacle-tower-env'

    def make_env(rank):
        def _thunk():
            env = ObstacleTowerEnv('../ObstacleTower/obstacletower',
                                   retro=True, worker_id=rank,
                                   realtime_mode=show,
                                   config={'total-floors': 20})
            env.seed(seed + rank % 8)
            env = bench.Monitor(env, None, allow_early_resets=True)
            env = OTWrapper(env)
            env = FrameStack(env, 4)
            return env
        return _thunk
    envs = [make_env(i) for i in range(num)]
    envs = SubprocVecEnv(envs, context='fork')
    envs = VecPyTorch(envs)
    return envs
