import argparse
import time
import torch
from tqdm import trange
import wandb

from common.optim import ParamOptim
from common.make_env import make_vec_envs
from common.make_obstacle_tower import make_obstacle_tower
from common.cfg import load_cfg
from ppo.agent import Agent
from ppo.model import ActorCritic
from ppo.runner import EnvRunner
from ppo.eval import eval_model
from ppo.runner_rand import random_rollout
from xent import Xent


def train(args):
    cfg = load_cfg(args.cfg)
    if args.env == "OT":
        envs = make_obstacle_tower(cfg["train"]["num_env"], args.seed)
    else:
        name = args.env + "NoFrameskip-v4"
        envs = make_vec_envs(name, cfg["train"]["num_env"], args.seed)

    emb = cfg["embedding"]
    model = ActorCritic(
        output_size=envs.action_space.n,
        emb_size=emb["size"],
        history_size=emb["history_size"],
        emb_fc_size=emb["fc_size"],
        device="cuda",
    )
    model.train().cuda()

    emb_trainer = Xent(
        emb_size=emb["size"],
        spatial_shift=emb["spatial_shift"],
        temporal_shift=emb["temporal_shift"],
        batch_size=emb["batch_size"],
        lr=emb["lr"],
        tau=emb["tau"],
    )

    optim = ParamOptim(**cfg["optimizer"], params=model.parameters())
    agent = Agent(model=model, optim=optim, **cfg["agent"])

    if args.load is not None:
        dump = torch.load(args.load, map_location="cuda")
        model.load_state_dict(dump[0])
        emb_trainer.encoder.load_state_dict(dump[1])

    wandb.init(project="edhr", config={**cfg, **vars(args)})

    rollout_size = cfg["train"]["rollout_size"]
    iic_buf, last_rand = random_rollout(
        emb["pretrain_steps"], rollout_size + 1, envs, "cuda"
    )
    iic_cursor = 0

    for epoch in trange(emb["pretrain_epochs"]):
        log = emb_trainer.update(iic_buf)
        if (epoch + 1) % (emb["pretrain_epochs"] // 100) == 0:
            wandb.log(log)

    runner = EnvRunner(
        rollout_size=rollout_size,
        envs=envs,
        model=model,
        encoder=emb_trainer.encoder,
        emb_size=emb["size"],
        history_size=emb["history_size"],
        device="cuda",
    )

    emb_trainer.encoder.eval()
    if not args.skip_train:
        n_end = int(
            (cfg["train"]["total_steps"] - emb["pretrain_steps"])
            / (envs.num_envs * rollout_size)
        )
        for n_iter, rollout in zip(trange(n_end), runner):
            progress = n_iter / n_end
            optim.update(progress)
            emb_trainer.optim.update(progress)
            agent_log = agent.update(rollout, progress)

            iic_buf[:, iic_cursor] = rollout["obs"][:, :, -1:]
            iic_cursor = (iic_cursor + 1) % iic_buf.shape[1]
            emb_trainer.encoder.train()
            for epoch in range(emb["epochs"]):
                emb_log = emb_trainer.update(iic_buf)
            emb_trainer.encoder.eval()

            if (n_iter + 1) % cfg["train"]["log_every"] == 0:
                wandb.log(
                    {**agent_log, **emb_log, **runner.get_logs(), "n_iter": n_iter}
                )

        filename = f"models/{int(time.time())}.pt"
        dump = [model.state_dict(), emb_trainer.encoder.state_dict()]
        torch.save(dump, filename)
        wandb.log({"filename": filename})

    reward = eval_model(
        model,
        envs,
        emb_trainer.encoder,
        emb["history_size"],
        emb["size"],
        "cuda",
        args.eval_ep,
    )
    wandb.log({"final/reward_mean": reward.mean(), "final/reward_std": reward.std()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MsPacman")
    parser.add_argument("--load", type=str)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-train", action="store_true", default=False)
    parser.add_argument("--eval-ep", type=int, default=100)

    train(parser.parse_args())
