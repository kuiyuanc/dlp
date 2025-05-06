from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pretty_errors  # noqa: F401
import torch
import torch.nn as nn
import wandb
from dataset import iclevrDataset
from ddpm import DDPM
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from rich.progress import Progress, TimeElapsedColumn
from torch.utils.data import DataLoader
from util import load_checkpoint, save_checkpoint


def get_random_timesteps(batch_size, timesteps, device):
    return torch.randint(0, timesteps, (batch_size,)).long().to(device)


def train_one_epoch(epoch, model, optimizer, loader, criterion, noise_scheduler, timesteps, device):
    model.train()
    losses = []

    with Progress(*Progress.get_default_columns(), TimeElapsedColumn(), auto_refresh=False) as progress:
        task = progress.add_task(f"Training Epoch: {epoch:3d}", total=len(loader))
        for x, label in loader:
            x, label = x.to(device), label.to(device)
            noise = torch.randn_like(x)

            timesteps = get_random_timesteps(loader.batch_size, timesteps, device)
            noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
            output = model(noisy_x, timesteps, label)

            loss = criterion(output, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            progress.update(task, advance=1)
            progress.refresh()

    return np.mean(losses)


def train(args, model, loader, noise_scheduler, criterion, optimizer, device, epoch_last=0):
    if device.type.startswith("xpu"):
        import intel_extension_for_pytorch as ipex  # type: ignore

        model, optimizer = ipex.optimize(model, torch.float32, optimizer, inplace=True)

    for epoch in range(epoch_last + 1, args.num_epochs + 1):
        loss = train_one_epoch(epoch, model, optimizer, loader, criterion, noise_scheduler, args.timesteps, device)
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, args.save_dir, epoch)
        try:
            wandb.log({"Loss/train": loss, "epoch": epoch})
        except Exception:
            print("wandb error")
        save_checkpoint(model, optimizer, args.save_dir, epoch, latest=True)


def mkdir(args):
    args.save_dir = Path(args.save_dir)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = Path(args.log_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)


def wandb_init(args, epoch_last):
    resume_from = f"{args.wandb_id}?_step={epoch_last}" if args.resume and args.wandb_id else None
    config = {
        "learning rate": args.lr,
        "timesteps": args.timesteps,
        "beta schedule": args.beta_schedule,
        "batch size": args.batch_size,
        "num workers": args.num_workers,
        "device": args.device,
    }

    wandb.login(key=args.api_key)
    wandb.init(
        project="dll-lab6",
        dir=args.log_dir,
        id=args.wandb_id,
        config=config,
        group=args.group,
        resume="allow",
        resume_from=resume_from,
        sync_tensorboard=True,
    )

    assert wandb.run
    wandb.run.name = wandb.run.id


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dataset", type=str, default="iclevr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", type=str, default="weights/latest.pth")
    parser.add_argument("--save-dir", type=str, default="weights")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--wandb-id", type=str, default=None)
    parser.add_argument("--group", type=str, default=None)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    mkdir(args)

    dataset = iclevrDataset(args.dataset, "train")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    device = torch.device(args.device)
    model = DDPM().to(device)
    criterion = nn.MSELoss()
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    epoch_last = load_checkpoint(model, optimizer, args.checkpoint, device) if args.resume else 0

    wandb_init(args, epoch_last)
    train(args, model, train_loader, noise_scheduler, criterion, optimizer, device, epoch_last)
    wandb.finish()


if __name__ == "__main__":
    main()
