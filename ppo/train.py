import argparse
import time
import torch
from tqdm import trange

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.make_obstacle_tower import make_obstacle_tower
from common.cfg import load_cfg
from common.logger import Logger
from ppo.agent import Agent
from ppo.model import ActorCritic
from ppo.runner import EnvRunner
from ppo.eval import eval_model
from encoders.iic import IIC


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'running on {device}')
    cfg = load_cfg(args.cfg)
    log = Logger(device=device)
    if args.env == 'OT':
        envs = make_obstacle_tower(cfg['train']['num_env'], args.seed)
    else:
        envs = make_vec_envs(args.env + 'NoFrameskip-v4',
                             cfg['train']['num_env'])

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

    if args.load is not None:
        dump = torch.load(args.load, map_location=device)
        model.load_state_dict(dump[0])
        emb_trainer.encoder.load_state_dict(dump[1])
        emb_trainer.encoder.head_main = args.head

    log.log.add_text('env', args.env)
    log.log.add_text('hparams', str(emb))

    if not args.skip_train:
        n_end = cfg['train']['steps']
        for n_iter, rollout in zip(trange(n_end), runner):
            progress = n_iter / n_end

            if progress >= emb['pretrain'] and\
                    emb_trainer.encoder.head_main is None:
                head_id = emb_trainer.select_head()
                log.log.add_text('iic', f'head {head_id}', n_iter)

            optim.update(progress)
            emb_trainer.optim.update(progress)
            agent_log = agent.update(rollout, progress)
            emb_log = emb_trainer.update(rollout['obs'])

            if n_iter % cfg['train']['log_every'] == 0:
                log.output({**agent_log, **emb_log, **runner.get_logs()},
                           n_iter)

        filename = f'models/{int(time.time())}.pt'
        dump = [model.state_dict(), emb_trainer.encoder.state_dict()]
        torch.save(dump, filename)
        log.log.add_text('filename', filename)

    reward = eval_model(model, envs, emb_trainer.encoder,
                        emb['history_size'], emb['size'], device,
                        args.eval_ep)
    reward_str = f'{reward.mean():.2f} {reward.std():.2f}'
    log.log.add_text('final', reward_str)
    log.log.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--cfg', type=str, default='plain')
    parser.add_argument('--env', type=str, default='MsPacman')

    parser.add_argument('--load', type=str)
    parser.add_argument('--head', type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--skip-train', action='store_true', default=False)
    parser.add_argument('--eval-ep', type=int, default=100)

    train(parser.parse_args())
