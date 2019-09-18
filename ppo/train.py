import torch
from tqdm import trange
import argparse

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from common.logger import Logger
from ppo.agent import Agent
from ppo.model import ActorCritic
from ppo.runner import EnvRunner
from ppo.eval import eval_model
from encoders.iic import IIC


def train(cfg_name, env_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(cfg_name)
    log = Logger(device=device)
    env_name += 'NoFrameskip-v4'
    envs = make_vec_envs(name=env_name, num=cfg['train']['num_env'])

    emb = cfg['embedding']
    model = ActorCritic(
        output_size=envs.action_space.n,
        device=device,
        emb_size=emb['size'],
        history_size=emb['history_size'],
        emb_hidden_size=emb.get('hidden_size'),
    )
    model.train().to(device=device)

    emb_trainer = IIC(
        emb_size=emb['size'],
        epochs=emb.get('epochs', 1),
        n_step=emb.get('n_step', 1),
        batch_size=emb.get('batch_size', 256),
        lr=emb.get('lr', 1e-4),
        device=device,
    )

    runner = EnvRunner(
        rollout_size=cfg['train']['rollout_size'],
        envs=envs,
        model=model,
        device=device,
        encoder=emb_trainer.encoder,
        emb_size=emb['size'],
        history_size=emb['history_size'],
    )

    optim = ParamOptim(**cfg['optimizer'], params=model.parameters())
    agent = Agent(model=model, optim=optim, **cfg['agent'])

    n_start = 0
    cp_iter = cfg['train']['checkpoint_every']
    log_iter = cfg['train']['log_every']
    n_end = cfg['train']['steps']
    cp_name = cfg['train']['checkpoint_name']

    log.log.add_text('env', env_name)
    log.log.add_text('hparams', str(emb))

    for n_iter, rollout in zip(trange(n_start, n_end), runner):
        progress = n_iter / n_end

        if progress >= emb['pretrain'] and\
                emb_trainer.encoder.head_main is None:
            head_id = emb_trainer.select_head()
            log.log.add_text('iic', f'head {head_id}', n_iter)

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

    reward = eval_model(model, envs, emb_trainer.encoder,
                        emb['history_size'], emb['size'], device)
    reward_str = f'{reward.mean():.2f} ± {reward.std():.2f}'
    log.log.add_text('final', reward_str)
    log.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default='plain')
    parser.add_argument('--env', type=str, default='MsPacman', choices=[
                        'MsPacman', 'SpaceInvaders', 'Breakout', 'Gravitar',
                        'QBert', 'Seaquest', 'Enduro', 'MontezumaRevenge'])
    args = parser.parse_args()
    train(args.cfg, args.env)
