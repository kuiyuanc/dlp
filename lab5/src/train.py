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

    parser.add_argument("--double", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--beta-anneal", action="store_true")
    parser.add_argument("--anneal-steps", type=int, default=1000000)
    parser.add_argument("--return-steps", type=int, default=3)

    parser.add_argument("--reward-scaling", type=float, default=1.0)
    parser.add_argument("--skip-frames", type=int, default=4)

    parser.add_argument("--task", "-t", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=str(int(time.time())))

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--num-sweep", type=int, default=9)
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument('--wandb-api-key', type=str, required=True)

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train():
    args = get_args()
    set_seed(args.seed)

    wandb_project, enhance, _, _, Agent = config(task=args.task)
    id = args.wandb_id
    wandb.init(project=wandb_project, name=id, config=dict(args._get_kwargs()), id=id, resume="allow")

    args.batch_size = wandb.config.get("batch_size", args.batch_size)
    args.lr = wandb.config.get("lr", args.lr)
    args.epsilon_decay = wandb.config.get("epsilon_decay", args.epsilon_decay)
    args.alpha = wandb.config.get("alpha", args.alpha)
    args.beta = wandb.config.get("beta", args.beta)
    args.return_steps = wandb.config.get("return_steps", args.return_steps)

    args.save_dir = Path(args.save_dir, wandb_project, enhance, id)
    args.model_path = Path(args.save_dir, f"{args.load_model}.pt") if args.load_model else None
    args.args_path = Path(args.save_dir, f"{args.load_model}.pkl") if args.load_model else None
    agent = Agent(args)
    agent.run(args.num_episodes)
    wandb.finish()


def sweep(args, wandb_project):
    sweep_config = {
        "method": "bayes",
        "name": "sweep",
        "metric": {"goal": "maximize", "name": "Eval Reward"},
        "parameters": {
            "batch_size": {"values": [32, 64]},
            "lr": {"max": 1e-4, "min": 1e-5},
            "epsilon_decay": {"max": 1 - 5e-7, "min": 1 - 5e-6},
            "alpha": {"min": 0.4, "max": 0.7},
            "beta": {"min": 0.3, "max": 0.5},
            "return_steps": {"values": [3, 5]},
        },
        "early_terminate": {"type": "hyperband", "min_iter": 16, "eta": 3},
    }
    args.sweep_id = args.sweep_id if args.sweep_id else wandb.sweep(sweep=sweep_config, project=wandb_project)
    wandb.agent(args.sweep_id, function=train, count=args.num_sweep)


def main():
    args = get_args()
    wandb.login(key=args.wandb_api_key)
    wandb_project, _, _, _, _ = config(task=args.task)
    if args.sweep:
        sweep(args, wandb_project)
    else:
        train()


if __name__ == "__main__":
    main()
