import sys
import torch
from tqdm import trange

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from common.logger import Logger
from ppo.agent import Agent
from ppo.model import ActorCritic
from ppo.runner import EnvRunner

from encoders.st_dim import STDIM
from encoders.iic import IIC
emb_trainers = {'st-dim': STDIM, 'iic': IIC}


def train(cfg_name, resume):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(cfg_name)
    log = Logger(device=device)
    envs = make_vec_envs(**cfg['env'])

    emb = cfg['embedding']
    concat_actions = emb.get('concat_actions', False)
    model_emb_size = emb['size'] + envs.action_space.n * concat_actions
    model = ActorCritic(output_size=envs.action_space.n, device=device,
                        emb_size=model_emb_size, emb_stack=emb['stack'],
                        use_rnn=emb['use_rnn'])
    model.train().to(device=device)

    emb_trainer = emb_trainers[emb['method']](
        emb_size=emb['size'], epochs=emb.get('epochs', 1),
        n_step=emb.get('n_step', 1), device=device)

    runner = EnvRunner(
        rollout_size=cfg['train']['rollout_size'],
        envs=envs,
        model=model,
        device=device,
        encoder=emb_trainer.encoder,
        emb_size=emb['size'],
        emb_stack=emb['stack'],
        concat_actions=concat_actions,
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
        emb_trainer.optim.update(progress)
        agent_log = agent.update(rollout, progress)
        emb_log = emb_trainer.update(rollout['obs'])

        if (n_iter + 1) % log_iter == 0:
            log.output({**agent_log, **emb_log, **runner.get_logs()}, n_iter)

        if (n_iter + 1) % cp_iter == 0:
            f = cp_name.format(n_iter=n_iter//cp_iter)
            dump = [model.state_dict(), emb_trainer.encoder.state_dict()]
            torch.save(dump, f)


if __name__ == '__main__':
    assert len(sys.argv) in [2, 3], 'config name required'
    resume = len(sys.argv) == 3 and sys.argv[2] == 'resume'
    train(sys.argv[1], resume)
