from tqdm import trange
import argparse
import time
import torch
from common.make_env import make_vec_envs
from common.cfg import load_cfg
from ppo.model import ActorCritic, ActorCriticHistory, ActorCriticInstant
from xent import Encoder


def render(args):
    cfg = load_cfg(args.cfg, "cfg")
    env = make_vec_envs(
        args.env + "NoFrameskip-v4", 1, cfg["model"]["num_obs"], args.seed
    )

    emb = cfg["embedding"]
    encoder = Encoder(emb["size"])
    models = {
        "both": ActorCritic,
        "history": ActorCriticHistory,
        "instant": ActorCriticInstant,
    }
    model = models[args.mode](
        **cfg["model"], obs_size=emb["size"], num_action=env.action_space.n
    )
    model.eval()
    encoder.eval()

    dump = torch.load(args.load, map_location="cpu")
    model.load_state_dict(dump[0])
    encoder.load_state_dict(dump[1])

    def model_enc(obs):
        bs, steps, width, height = obs.shape
        x = obs.float() / 255
        x_emb = x.view(bs * steps, 1, width, height)
        x_emb = encoder(x_emb)
        x_emb = x_emb.view(bs, steps, x_emb.shape[-1])
        return model(x[:, -4:], x_emb), x_emb[:, -1]

    emb_stack = []
    obs = env.reset()
    for _ in trange(args.steps):
        with torch.no_grad():
            x = model_enc(obs)
            emb_stack.append(x[1])
            a = x[0][0].sample().unsqueeze(1)
        obs, r, terms, infos = env.step(a)
        env.render()
        time.sleep(1 / 10)

    if args.save:
        emb_stack = torch.cat(emb_stack)
        torch.save(emb_stack, "models/emb_stack.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--cfg", type=str, default="default")
    parser.add_argument("--env", type=str, default="MsPacman")
    parser.add_argument("--load", type=str, required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--mode", type=str, choices=["both", "instant", "history"], default="history"
    )
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    render(parser.parse_args())
