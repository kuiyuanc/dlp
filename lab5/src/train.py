import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import wandb
from util import config


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default="./saved_models")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)

    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--anneal-steps", type=int, default=1000000)
    parser.add_argument("--return-steps", type=int, default=3)

    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument('--skip-frames', type=int, default=4)

    parser.add_argument("--task", "-t", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=str(int(time.time())))

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    args = get_args()
    set_seed(args.seed)

    wandb_project, enhance, _, _, Agent = config(task=args.task)
    id = args.wandb_id
    wandb.init(project=wandb_project, name=id, config=dict(args._get_kwargs()), id=id, resume="allow")
    args.save_dir = Path(args.save_dir, wandb_project, enhance, id)
    args.model_path = Path(args.save_dir, f"{args.load_model}.pt") if args.load_model else None
    args.args_path = Path(args.save_dir, f"{args.load_model}.pkl") if args.load_model else None
    agent = Agent(args)
    agent.run(args.num_episodes)
    wandb.finish()


if __name__ == "__main__":
    main()
