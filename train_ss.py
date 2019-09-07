import sys
import random
import torch
from torch import nn
from tqdm import trange, tqdm
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, RandomSampler

from common.make_env import make_vec_envs
from runner_rand import EnvRunnerRand
from st_dim import STDIM


def shift(x):
    return x + random.randrange(1, 4) * random.randrange(-1, 2, 2)

def gen_val(obs, num_samples=1024):
    obs = obs.permute(1, 0, 2, 3, 4).contiguous().view(-1, *obs.shape[-3:])
    idx1 = random.sample(range(3, obs.shape[0] - 3), num_samples)
    idx2 = list(map(shift, idx1))
    x1 = obs[idx1].float() / 255
    x2 = obs[idx2].float() / 255
    return x1, x2


def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')

    envs = make_vec_envs(name='MontezumaRevengeNoFrameskip-v4', num=8)
    st_dim = STDIM(feature_size=256, device=device)
    rollout_size = 1000
    runner = EnvRunnerRand(rollout_size=rollout_size, envs=envs, device=device)
    params = list(st_dim.encoder.parameters()) +\
        list(st_dim.classifier1.parameters()) +\
        list(st_dim.classifier2.parameters())

    opt = torch.optim.Adam(params, lr=3e-4)
    batch_size = 64

    for n_iter, obs in zip(trange(300), runner):
        if n_iter == 0:
            obs_val1, obs_val2 = gen_val(obs)
            continue

        obs = obs.permute(1, 0, 2, 3, 4).contiguous()\
            .view(-1, *obs.shape[-3:]).float() / 255

        losses = []
        sampler = BatchSampler(
            SubsetRandomSampler(range(3, len(obs) - 3)),
            batch_size, drop_last=True)

        for idx1 in sampler:
            idx2 = list(map(shift, idx1))
            loss = st_dim.get_loss(obs[idx1], obs[idx2])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        with torch.no_grad():
            val_loss = 0
            for i in range(0, len(obs_val1), batch_size):
                val_loss += st_dim.get_loss(
                    obs_val1[i: i + batch_size],
                    obs_val2[i: i + batch_size]
                )
        loss_t = sum(losses) / len(losses)
        loss_v = val_loss.item() / (len(obs_val1) // batch_size)
        print(f'train: {loss_t:.3f} val: {loss_v:.3f}')

    torch.save(st_dim.encoder.state_dict(), 'encoder.pt')


if __name__ == '__main__':
    train()
