from pathlib import Path

import wandb
from util import args_to_config, args_to_sweep_config, get_args, get_config, load_prior_runs, set_seed


def train():
    args = get_args()
    set_seed(args.seed)
    wandb_project, enhance, _, _, Agent = get_config(task=args.task)

    if args.sweep:
        run = wandb.init()
        run.name = run.id

        args.batch_size = wandb.config["batch_size"]
        args.learning_rate = wandb.config["learning_rate"]
        args.epsilon_decay = wandb.config["epsilon_decay"]
        args.target_update_frequency = wandb.config["target_update_frequency"]
        args.replay_start_size = args.memory_size
        args.tau = wandb.config["tau"]
        if args.task == 3:
            args.alpha = wandb.config["alpha"]
            args.beta = wandb.config["beta"]
            args.epsilon = wandb.config["epsilon"]
            args.return_steps = wandb.config["return_steps"]
    else:
        config = args_to_config(args)
        wandb.init(project=wandb_project, name=args.wandb_id, config=config, id=args.wandb_id, resume="allow")

    args.save_dir = Path(args.save_dir, wandb_project, enhance, args.wandb_id)
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
