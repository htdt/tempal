import sys
import time
import torch
from common.make_env import make_vec_envs
from common.cfg import load_cfg, find_checkpoint
from ppo.model import ActorCritic
from encoders.iic import Encoder


def render(cfg_name, steps):
    cfg = load_cfg(cfg_name)
    cfg['env']['num'] = 1
    env = make_vec_envs(**cfg['env'])

    emb = cfg['embedding']
    encoder = Encoder(emb['size'], 6)
    model = ActorCritic(
        output_size=env.action_space.n,
        emb_size=emb['size'],
        emb_stack_in=emb['stack_in'],
        emb_stack_out=emb['stack_out'],
        use_rnn=emb['use_rnn'],
        device='cpu',
    )
    model.eval()
    encoder.eval()

    n_start, fname = find_checkpoint(cfg)
    dump = torch.load(fname, map_location='cpu')
    model.load_state_dict(dump[0])
    encoder.load_state_dict(dump[1])
    encoder.head_main = 5

    obs_emb = torch.zeros(1, emb['stack_in'], emb['size'])
    obs = env.reset()
    with torch.no_grad():
        obs_emb[0, -1] = encoder(obs[:, -1:])

    for n_iter in range(steps):
        with torch.no_grad():
            a = model(obs, obs_emb)[0].sample().unsqueeze(1)

        obs, r, terms, infos = env.step(a)
        obs_emb[0, :-1].copy_(obs_emb[0, 1:])
        obs_emb *= (~terms).float()
        with torch.no_grad():
            obs_emb[0, -1] = encoder(obs[:, -1:])
        print(obs_emb[0, -1].argmax().item())

        env.render()
        time.sleep(1/30)


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'required: config name, steps'
    render(sys.argv[1], int(sys.argv[2]))
