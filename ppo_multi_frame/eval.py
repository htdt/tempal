import torch
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from ppo_multi_frame.model import ActorCritic


def eval_model(
    model: ActorCritic,
    env: ShmemVecEnv,
    history_size: int,
    emb_size: int,
    device: str,
    num_ep=100
):
    model.eval()

    obs_emb = torch.zeros(env.num_envs, history_size, 1, 84, 84,
                          device=device, dtype=torch.uint8)
    obs = env.reset().to(device=device)
    obs_emb[:, -1] = obs[:, -1:]

    ep_reward = []
    while True:
        with torch.no_grad():
            a = model(obs, obs_emb)[0].sample().unsqueeze(1)
        obs, rewards, terms, infos = env.step(a)
        obs = obs.to(device=device)

        obs_emb[:, :-1].copy_(obs_emb[:, 1:])
        obs_emb *= (~terms)[..., None, None, None
                            ].to(device=device, dtype=torch.uint8)
        obs_emb[:, -1] = obs[:, -1:]

        for info in infos:
            if 'episode' in info.keys():
                ep_reward.append(info['episode']['r'])
                if len(ep_reward) == num_ep:
                    return torch.tensor(ep_reward)
