import argparse
import time
import torch
from tqdm import trange
import wandb

from common.make_env import make_vec_envs
from common.cfg import load_cfg
from ppo.agent import Agent
from ppo.model import ActorCritic, ActorCriticHistory, ActorCriticInstant
from ppo.runner import EnvRunner
from ppo.eval import eval_model
from xent import Xent


def train(args):
    cfg = load_cfg(args.cfg, "cfg")
    if args.mode == "instant":
        cfg["model"]["num_obs"] = 4
    wandb.init(project="edhr", config={**cfg, **vars(args)})
    emb = cfg["embedding"]
    envs = make_vec_envs(
        name=args.env + "NoFrameskip-v4",
        num=cfg["train"]["num_env"],
        nstack=cfg["model"]["num_obs"],
        clip_rewards=cfg["train"]["clip_rewards"],
        max_ep_steps=cfg["train"]["max_ep_steps"],
        seed=args.seed,
    )

    models = {
        "both": ActorCritic,
        "history": ActorCriticHistory,
        "instant": ActorCriticInstant,
    }
    model = models[args.mode](
        **cfg["model"], obs_size=emb["size"], num_action=envs.action_space.n
    )
    model.train().cuda()

    if args.mode != "instant":
        emb_trainer = Xent(
            emb_size=emb["size"],
            spatial_shift=emb["spatial_shift"],
            temporal_shift=emb["temporal_shift"],
            batch_size=emb["batch_size"],
            rollouts_in_batch=emb["rollouts_in_batch"],
            optimizer=emb["optimizer"],
            tau=emb["tau"],
        )
        encoder = emb_trainer.encoder
        if emb["pretrain"]["epochs"] == 0:
            encoder.load_state_dict(torch.load("models/encoder.pt"))
    else:
        encoder = None

    agent = Agent(model=model, encoder=encoder, **cfg["agent"])

    rollout_size = cfg["train"]["rollout_size"]
    runner = EnvRunner(
        rollout_size=rollout_size,
        envs=envs,
        model=model,
        encoder=encoder,
        emb_size=emb["size"],
        input_size=cfg["model"]["num_obs"],
        device="cuda",
    )

    pretrain_steps = int(emb["pretrain"]["steps"] / (envs.num_envs * rollout_size))
    obs_shape = list(envs.observation_space.shape)
    obs_shape[0] = 1
    if args.mode != "instant":
        buffer = torch.empty(
            rollout_size + 1,
            pretrain_steps,
            envs.num_envs,
            *obs_shape,
            dtype=torch.uint8,
            device="cuda",
        )
        buf_cursor = 0
    else:
        emb_log = {}

    n_end = int(cfg["train"]["total_steps"] / (envs.num_envs * rollout_size))
    for n_iter, rollout in zip(trange(n_end), runner):
        if args.mode != "instant":
            buffer[:, buf_cursor] = rollout["obs"][:, :, -1:]
            buf_cursor = (buf_cursor + 1) % buffer.shape[1]
        if n_iter < pretrain_steps:
            continue
        elif n_iter == pretrain_steps:
            if args.mode != "instant":
                encoder.train()
                for epoch in trange(emb["pretrain"]["epochs"]):
                    log = emb_trainer.update(buffer)
                    if (epoch + 1) % (emb["pretrain"]["epochs"] // 100) == 0:
                        wandb.log(log)
                encoder.eval()
                buf_cursor = 0
                buffer = buffer[:, : emb["rollouts_in_batch"]]
                if emb["pretrain"]["epochs"] > 0:
                    torch.save(encoder.state_dict(), "models/encoder.pt")
            runner.rnd = False

        else:
            progress = n_iter / n_end
            agent.optim.update(progress)

            if args.mode != "instant":
                emb_trainer.optim.update(progress) 
                encoder.train()
                for epoch in range(emb["epochs"]):
                    emb_log = emb_trainer.update(buffer)
                encoder.eval()
            agent_log = agent.update(rollout, progress)
            if (n_iter + 1) % cfg["train"]["log_every"] == 0:
                wandb.log(
                    {**agent_log, **emb_log, **runner.get_logs(), "progress": progress}
                )

    filename = f"models/{int(time.time())}.pt"
    dump = [model.state_dict()]
    if encoder is not None:
        dump += [encoder.state_dict()]
    torch.save(dump, filename)
    wandb.log({"filename": filename})

    reward = eval_model(model, envs, encoder)
    wandb.log({"final/reward_mean": reward.mean(), "final/reward_std": reward.std()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MsPacman")
    parser.add_argument(
        "--mode", type=str, choices=["both", "instant", "history"], default="history"
    )
    parser.add_argument("--seed", type=int, default=0)
    train(parser.parse_args())
