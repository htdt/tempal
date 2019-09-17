import torch
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from ppo.model import ActorCritic
from encoders.iic import Encoder


def eval_model(
    model: ActorCritic,
    env: ShmemVecEnv,
    encoder: Encoder,
    history_size: int,
    emb_size: int,
    device: str,
    num_ep=100
):
    model.eval()
    encoder.eval()

    obs_emb = torch.zeros(env.num_envs, history_size,
                          emb_size).to(device=device)
    obs = env.reset().to(device=device)
    with torch.no_grad():
        obs_emb[:, -1] = encoder(obs[:, -1:])

    ep_reward = []
    while True:
        with torch.no_grad():
            a = model(obs, obs_emb)[0].sample().unsqueeze(1)
        obs, rewards, terms, infos = env.step(a)
        obs = obs.to(device=device)
        obs_emb[:, :-1].copy_(obs_emb[:, 1:])
        obs_emb *= (~terms)[..., None].to(device=device, dtype=torch.float)
        with torch.no_grad():
            obs_emb[:, -1] = encoder(obs[:, -1:])

        for info in infos:
            if 'episode' in info.keys():
                ep_reward.append(info['episode']['r'])
                if len(ep_reward) == num_ep:
                    return torch.tensor(ep_reward)
