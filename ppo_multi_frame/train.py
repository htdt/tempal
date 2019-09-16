import sys
import torch
from tqdm import trange

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from common.logger import Logger
from ppo.agent import Agent
from frame_stack.model import ActorCritic
from frame_stack.runner import EnvRunner


def train(cfg_name, resume):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(cfg_name)
    log = Logger(device=device)
    envs = make_vec_envs(**cfg['env'])

    emb = cfg['embedding']
    model = ActorCritic(output_size=envs.action_space.n, device=device,
                        emb_size=emb['size'])
    model.train().to(device=device)

    runner = EnvRunner(
        rollout_size=cfg['train']['rollout_size'],
        envs=envs,
        model=model,
        device=device,
        emb_stack=emb['stack'],
    )

    optim = ParamOptim(**cfg['optimizer'], params=model.parameters())
    agent = Agent(model=model, optim=optim, **cfg['agent'])

    n_start = 0
    cp_iter = cfg['train']['checkpoint_every']
    log_iter = cfg['train']['log_every']
    n_end = cfg['train']['steps']
    cp_name = cfg['train']['checkpoint_name']

    for n_iter, rollout in zip(trange(n_start, n_end), runner):
        progress = n_iter / n_end
        optim.update(progress)
        agent_log = agent.update(rollout, progress)

        if n_iter % log_iter == 0:
            log.output({**agent_log, **runner.get_logs()}, n_iter)

        if n_iter > n_start and n_iter % cp_iter == 0:
            f = cp_name.format(n_iter=n_iter//cp_iter)
            torch.save(model.state_dict(), f)


if __name__ == '__main__':
    assert len(sys.argv) in [2, 3], 'config name required'
    resume = len(sys.argv) == 3 and sys.argv[2] == 'resume'
    train(sys.argv[1], resume)