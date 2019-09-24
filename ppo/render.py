import argparse
import time
import torch
from common.make_env import make_vec_envs
from common.make_obstacle_tower import make_obstacle_tower
from common.cfg import load_cfg
from ppo.model import ActorCritic
from encoders.iic import Encoder


def render(args):
    cfg = load_cfg(args.cfg)
    if args.env == 'OT':
        env = make_obstacle_tower(1, args.seed, True)
    else:
        env = make_vec_envs(args.env + 'NoFrameskip-v4', 1)

    emb = cfg['embedding']
    encoder = Encoder(emb['size'], 6)
    model = ActorCritic(
        output_size=env.action_space.n,
        emb_size=emb['size'],
        history_size=emb['history_size'],
        emb_hidden_size=emb.get('hidden_size'),
        device='cpu',
    )
    model.eval()
    encoder.eval()

    dump = torch.load(args.load, map_location='cpu')
    model.load_state_dict(dump[0])
    encoder.load_state_dict(dump[1])
    encoder.head_main = args.head

    obs_emb = torch.zeros(1, emb['history_size'], emb['size'])
    obs = env.reset()
    with torch.no_grad():
        obs_emb[0, -1] = encoder(obs[:, -1:])

    clusters = []
    for n_iter in range(args.steps):
        with torch.no_grad():
            a = model(obs, obs_emb)[0].sample().unsqueeze(1)

        obs, r, terms, infos = env.step(a)
        obs_emb[0, :-1].copy_(obs_emb[0, 1:])
        obs_emb *= (~terms).float()
        with torch.no_grad():
            obs_emb[0, -1] = encoder(obs[:, -1:])
        c = obs_emb[0, -1].argmax().item()
        clusters.append(c)
        print(c)

        if args.env != 'OT':
            env.render()
        time.sleep(1/30)
    # print(clusters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default='plain')
    parser.add_argument('--env', type=str, default='MsPacman')
    parser.add_argument('--load', type=str, required=True)
    parser.add_argument('--head', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=1000)
    render(parser.parse_args())
