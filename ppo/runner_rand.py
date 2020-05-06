from tqdm import trange
import torch


def random_rollout(size, rollout_size, envs, device="cpu"):
    num_rollouts = int(size / rollout_size / envs.num_envs)
    obs_shape = list(envs.observation_space.shape)
    obs_shape[0] = 1
    obs = torch.empty(
        rollout_size,
        num_rollouts,
        envs.num_envs,
        *obs_shape,
        dtype=torch.uint8,
        device=device
    )

    for r in trange(num_rollouts):
        if r == 0:
            obs[0, 0] = envs.reset()[:, -1:]
        else:
            obs[0, r].copy_(obs[-1, r - 1])

        for step in range(1, rollout_size):
            actions = [envs.action_space.sample() for _ in range(envs.num_envs)]
            actions = torch.tensor(actions).unsqueeze(-1)
            obs_last = envs.step(actions)[0]
            obs[step, r] = obs_last[:, -1:]

    return obs, obs_last
