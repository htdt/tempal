import sys
import time
from tqdm import trange
import torch
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from model import init_model


def render(cfg_name, steps):
    cfg = load_cfg(cfg_name)
    cfg['env']['num'] = 1
    env = make_vec_envs(**cfg['env'])
    model, n_start = init_model(cfg, env, 'cpu', resume=True)
    assert n_start > 0
    model.eval()
    print(f'running {n_start}')

    obs = env.reset()
    for n_iter in trange(steps):
        with torch.no_grad():
            a = model(obs)[0].sample().unsqueeze(1)
        obs = env.step(a)[0]
        env.render()
        time.sleep(1/30)


if __name__ == '__main__':
    assert len(sys.argv) == 3, 'required: config name, steps'
    render(sys.argv[1], int(sys.argv[2]))
