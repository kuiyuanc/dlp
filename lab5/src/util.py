import argparse
import operator
import random
import time
from itertools import repeat
from pathlib import Path

import conv_dqn
import dqn
import enhanced_conv_dqn
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--save-dir", type=str, default="./weights")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--memory-size", type=int, default=32768)
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.00025)
    parser.add_argument("--discount-factor", "--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999374)
    parser.add_argument("--epsilon-min", type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=128)
    parser.add_argument("--replay-start-size", type=int, default=32768)
    parser.add_argument("--max-episode-steps", type=int, default=100000)
    parser.add_argument("--train-per-step", type=int, default=4)

    parser.add_argument("--double", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--return-steps", type=int, default=3)

    parser.add_argument("--skip-frames", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.002)

    parser.add_argument("--task", "-t", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-episodes", type=int, default=1000)
    parser.add_argument("--eval-frequency", type=int, default=1)
    parser.add_argument("--backup-frequency", type=int, default=100)
    parser.add_argument("--early-stop", type=float, default=19)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=str(int(time.time())))

    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-method", type=str, default="bayes")
    parser.add_argument("--min-iter", type=int, default=16)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--num-sweep", type=int, default=27)
    parser.add_argument("--sweep-id", type=str, default=None)
    parser.add_argument("--prior-sweep-id", type=str, default=None)
    parser.add_argument("--wandb-api-key", type=str, required=True)

    return parser.parse_args()


def args_to_config(args: argparse.Namespace) -> dict:
    config = vars(args)

    config.pop("save_dir")
    config.pop("num_episodes")
    config.pop("eval_frequency")
    config.pop("backup_frequency")
    config.pop("early_stop")
    config.pop("load_model")
    config.pop("wandb_id")

    config.pop("sweep")
    config.pop("sweep_method")
    config.pop("min_iter")
    config.pop("eta")
    config.pop("num_sweep")
    config.pop("sweep_id")
    config.pop("prior_sweep_id")
    config.pop("wandb_api_key")

    if args.task != 3:
        config.pop("alpha")
        config.pop("beta")
        config.pop("epsilon")
        config.pop("return_steps")
        config.pop("skip_frames")

    return config


def args_to_sweep_config(args: argparse.Namespace) -> dict:
    parameters = {
        # searching space
        "batch_size": {"values": get_geometry_series(base=1, ratio=2, begin=4, end=8)},
        "learning_rate": {"values": get_geometry_series(base=1, ratio=10**-0.5, begin=4, end=12)},
        "epsilon_decay": {"values": tuple(1 - x for x in get_geometry_series(base=1, ratio=0.1, begin=2, end=6))},
        "target_update_frequency": {"values": get_geometry_series(base=256, ratio=2, begin=0, end=5)},
        "tau": {"values": get_geometry_series(base=0.1, ratio=0.1, begin=0, end=3)},
        # preset
        "memory_size": {"value": args.memory_size},
        "discount_factor": {"value": args.discount_factor},
        "epsilon_start": {"value": args.epsilon_start},
        "epsilon_min": {"value": args.epsilon_min},
        "replay_start_size": {"value": args.replay_start_size},
        "max_episode_steps": {"value": args.max_episode_steps},
        "train_per_step": {"value": args.train_per_step},
        "double": {"value": args.double},
        "skip_frames": {"value": args.skip_frames},
        "task": {"value": args.task},
        "seed": {"value": args.seed},
        "device": {"value": args.device},
    }

    if args.task == 3:
        parameters |= {
            "alpha": {"values": get_linear_series(base=0.0, diff=0.2, begin=0, end=5)},
            "beta": {"values": get_linear_series(base=0.0, diff=0.2, begin=0, end=5)},
            "epsilon": {"values": get_geometry_series(base=0.1, ratio=0.1, begin=0, end=2)},
            "return_steps": {"values": get_geometry_series(base=1, ratio=2, begin=0, end=2)},
        }

    config = {
        "method": args.sweep_method,
        "metric": {"goal": "maximize", "name": "EWMA Return"},
        "parameters": parameters,
        "early_terminate": {"type": "hyperband", "min_iter": args.min_iter, "eta": args.eta},
    }

    return config


def load_prior_runs(args: argparse.Namespace) -> list:
    if args.prior_sweep_id:
        return [path.stem.removeprefix("config-") for path in Path(f"wandb/sweep-{args.wandb_id}").iterdir()]
    elif Path("wandb/sweep.txt").exists():
        with open("wandb/sweep.txt", "r") as f:
            return f.readlines()
    else:
        return []


def get_config(*, task: int) -> tuple[str, str, bool, type, type]:
    if task == 1:
        env_name = "CartPole-v1"
        enhance = "vanilla"
        atari = False
        DQN = dqn.DQN
        Agent = dqn.DQNAgent
    elif task == 2:
        env_name = "Pong-v5"
        enhance = "vanilla"
        atari = True
        DQN = conv_dqn.ConvDQN
        Agent = conv_dqn.ConvDQNAgent
    elif task == 3:
        env_name = "Pong-v5"
        enhance = "enhanced"
        atari = True
        DQN = conv_dqn.ConvDQN
        Agent = enhanced_conv_dqn.EnhancedConvDQNAgent
    else:
        raise ValueError("Invalid task")

    return env_name, enhance, atari, DQN, Agent


def get_geometry_series(base, ratio, begin: int, end: int) -> tuple:
    exponents = [ratio**begin]
    for _ in range(begin + 1, end + 1):
        exponents.append(exponents[-1] * ratio)
    return tuple(map(operator.mul, repeat(base), exponents))


def get_linear_series(base, diff, begin: int, end: int) -> tuple:
    return tuple(base + diff * i for i in range(begin, end + 1))
