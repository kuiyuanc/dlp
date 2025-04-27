from pathlib import Path

import wandb
import pretty_errors  # noqa: F401
from util import args_to_config, args_to_sweep_config, get_args, get_config, load_prior_runs, set_seed


def train():
    args = get_args()
    set_seed(args.seed)
    project, enhance, _, _, Agent = get_config(task=args.task)

    group = f"{args.task=}"
    dir = Path("wandb", project, "enhanced" if args.task == 3 else "vanilla")

    if args.sweep:
        wandb.init(group=group, dir=dir)

        args.batch_size = wandb.config["batch_size"]
        args.learning_rate = wandb.config["learning_rate"]
        args.epsilon_decay = wandb.config["epsilon_decay"]
        args.target_update_frequency = wandb.config["target_update_frequency"]
        args.tau = wandb.config["tau"]
        if args.task == 3:
            args.alpha = wandb.config["alpha"]
            args.beta = wandb.config["beta"]
            args.epsilon = wandb.config["epsilon"]
            args.return_steps = wandb.config["return_steps"]
    else:
        config = args_to_config(args)
        if args.wandb_id:
            wandb.init(group=group, dir=dir, project=project, config=config, id=args.wandb_id, resume="allow")
        else:
            wandb.init(group=group, dir=dir, project=project, config=config)

    assert wandb.run
    wandb.run.name = wandb.run.id

    args.wandb_id = wandb.run.id
    args.save_dir = Path(args.save_dir, project, enhance, args.wandb_id)
    args.model_path = Path(args.save_dir, f"{args.load_model}.pt") if args.load_model else None
    args.args_path = Path(args.save_dir, f"{args.load_model}.pkl") if args.load_model else None

    agent = Agent(args)
    agent.run(args.num_episodes)

    wandb.finish()


def sweep(args):
    wandb_project, _, _, _, _ = get_config(task=args.task)
    if not args.sweep_id:
        config = args_to_sweep_config(args)
        prior_runs = load_prior_runs(args)
        args.sweep_id = wandb.sweep(sweep=config, project=wandb_project, prior_runs=prior_runs)
    wandb.agent(args.sweep_id, project=wandb_project, function=train, count=args.num_sweep)


def main():
    args = get_args()
    wandb.login(key=args.wandb_api_key)
    if args.sweep:
        sweep(args)
    else:
        train()


if __name__ == "__main__":
    main()
