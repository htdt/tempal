import torch
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from ppo.model import ActorCritic


def eval_model(
    model: ActorCritic,
    env: ShmemVecEnv,
    encoder: torch.nn.Module,
    device="cuda",
    num_ep=100
):
    model.eval()
    if encoder is not None:
        encoder.eval()
    obs = env.reset()
    ep_reward = []

    def model_enc(obs):
        bs, steps, width, height = obs.shape
        x = obs.float() / 255
        with torch.no_grad():
            if encoder is None:
                x_emb = None
            else:
                x_emb = x.view(bs * steps, 1, width, height)
                x_emb = encoder(x_emb)
                x_emb = x_emb.view(bs, steps, x_emb.shape[-1])
            return model(x[:, -4:], x_emb)

    while True:
        with torch.no_grad():
            obs = obs.to(device=device)
            a = model_enc(obs)[0].sample().unsqueeze(1)
        obs, rewards, terms, infos = env.step(a)

        for info in infos:
            if 'episode' in info.keys():
                ep_reward.append(info['episode']['r'])
                if len(ep_reward) == num_ep:
                    return torch.tensor(ep_reward)
