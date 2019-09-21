import argparse
import torch
from tqdm import trange

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.make_obstacle_tower import make_obstacle_tower
from common.cfg import load_cfg
from common.logger import Logger
from ppo.agent import Agent
from ppo_multi_frame.model import ActorCritic
from ppo_multi_frame.runner import EnvRunner
from ppo_multi_frame.eval import eval_model


def train(cfg_name, env_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(cfg_name)
    log = Logger(device=device)
    if env_name == 'OT':
        envs = make_obstacle_tower(cfg['train']['num_env'])
    else:
        envs = make_vec_envs(env_name + 'NoFrameskip-v4',
                             cfg['train']['num_env'])

    emb = cfg['embedding']
    model = ActorCritic(output_size=envs.action_space.n, device=device,
                        emb_size=emb['size'])
    model.train().to(device=device)

    runner = EnvRunner(
        rollout_size=cfg['train']['rollout_size'],
        envs=envs,
        model=model,
        device=device,
        emb_stack=emb['history_size'],
    )

    optim = ParamOptim(**cfg['optimizer'], params=model.parameters())
    agent = Agent(model=model, optim=optim, **cfg['agent'])

    n_start = 0
    log_iter = cfg['train']['log_every']
    n_end = cfg['train']['steps']

    log.log.add_text('env', env_name)

    for n_iter, rollout in zip(trange(n_start, n_end), runner):
        progress = n_iter / n_end
        optim.update(progress)
        agent_log = agent.update(rollout, progress)
        if n_iter % log_iter == 0:
            log.output({**agent_log, **runner.get_logs()}, n_iter)

    reward = eval_model(model, envs, emb['history_size'], emb['size'], device)
    reward_str = f'{reward.mean():.2f} ± {reward.std():.2f}'
    log.log.add_text('final', reward_str)
    log.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default='plain')
    parser.add_argument('--env', type=str, default='MsPacman')
    args = parser.parse_args()
    train(args.cfg, args.env)
