import argparse
import torch
from tqdm import tqdm
from common.make_env import make_vec_envs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--env", type=str, default="MsPacman")
    parser.add_argument("--num_ep", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    envs = make_vec_envs(
        name=args.env + "NoFrameskip-v4",
        num=8,
        nstack=1,
        max_ep_steps=10000,
        seed=args.seed,
    )
    envs.reset()
    ep_reward = []
    with tqdm(total=args.num_ep) as pbar:
        while len(ep_reward) < args.num_ep:
            a = [envs.action_space.sample() for _ in range(envs.num_envs)]
            a = torch.tensor(a).unsqueeze(-1)
            _, _, _, infos = envs.step(a)

            for info in infos:
                if 'episode' in info.keys():
                    ep_reward.append(info['episode']['r'])
                    pbar.update(1)
    r = torch.tensor(ep_reward)
    print(f"{r.mean().item():.1f} +- {r.std().item():.1f}")
