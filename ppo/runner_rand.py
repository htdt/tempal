import torch


def random_rollout(num_steps, envs, device='cpu'):
    env_steps = num_steps // envs.num_envs
    obs_shape = envs.observation_space.shape
    obs = torch.empty(env_steps, envs.num_envs, *obs_shape,
                      dtype=torch.uint8, device=device)
    obs[0] = envs.reset()
    for step in range(1, env_steps):
        actions = [envs.action_space.sample()
                   for _ in range(envs.num_envs)]
        actions = torch.tensor(actions).unsqueeze(-1)
        obs[step] = envs.step(actions)[0]
    return obs
