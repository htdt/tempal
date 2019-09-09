import sys
import torch
from tqdm import trange

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from common.logger import Logger
from agent import Agent
from model import ActorCritic
from runner import EnvRunner
from st_dim import STDIM


# TODO terminal mask on obs_emb


def train(cfg_name, resume):
    emb_size = 32
    emb_stack = 50

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(cfg_name)
    log = Logger(device=device)
    envs = make_vec_envs(**cfg['env'])
    model = ActorCritic(output_size=envs.action_space.n,
                        device=device, emb_size=emb_size, emb_stack=emb_stack)
    model.train().to(device=device)
    st_dim = STDIM(emb_size=emb_size, device=device)

    runner = EnvRunner(
        rollout_size=cfg['train']['rollout_size'],
        envs=envs,
        model=model,
        device=device,
        encoder=st_dim.encoder,
        emb_size=emb_size,
        emb_stack=emb_stack)

    optim = ParamOptim(**cfg['optimizer'], params=model.parameters())
    agent = Agent(model=model, optim=optim, emb_stack=emb_stack,
        **cfg['agent'])

    n_start = 0
    cp_iter = cfg['train']['checkpoint_every']
    log_iter = cfg['train']['log_every']
    n_end = cfg['train']['steps']
    cp_name = cfg['train']['checkpoint_name']

    for n_iter, rollout in zip(trange(n_start, n_end), runner):
        progress = n_iter / n_end
        optim.update(progress)
        st_dim.optim.update(progress)
        agent_log = agent.update(rollout, progress)
        emb_log = st_dim.update(rollout['obs'])

        if n_iter % log_iter == 0:
            log.output({**agent_log, **emb_log, **runner.get_logs()}, n_iter)

        if n_iter > n_start and n_iter % cp_iter == 0:
            f = cp_name.format(n_iter=n_iter//cp_iter)
            torch.save(model.state_dict(), f)


if __name__ == '__main__':
    assert len(sys.argv) in [2, 3], 'config name required'
    resume = len(sys.argv) == 3 and sys.argv[2] == 'resume'
    train(sys.argv[1], resume)
