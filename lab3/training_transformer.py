import argparse
import os
from enum import Enum, IntEnum, auto

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from models import MaskGit as VQGANTransformer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import utils as vutils
from tqdm import tqdm
from utils import LoadTrainData


# TODO2 step1-4: design the transformer training strategy
class TrainTransformer:
    class Mode(Enum):
        Train = 0
        Val = 1

    class Color(IntEnum):
        Red = 91
        Green = auto()
        Yellow = auto()
        Blue = auto()
        Pink = auto()
        Cyan = auto()
        White = auto()

    def __init__(self, args: argparse.Namespace, MaskGit_CONFIGS: dict):
        self.device = args.device if torch.cuda.is_available() else "cpu"
        self.model = VQGANTransformer(MaskGit_CONFIGS["model_param"]).to(device=self.device)
        self.optim, self.scheduler = self.configure_optimizers(args.learning_rate)
        self.prepare_training(args.save_dir)
        self.accum_grad: int = args.accum_grad
        self.save_dir: str = args.save_dir
        self.logger = SummaryWriter(log_dir=args.logdir)

    @staticmethod
    def prepare_training(save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        return self.run_one_epoch(TrainTransformer.Mode.Train, epoch, self.train_loader)

    @torch.no_grad()
    def eval_one_epoch(self, epoch: int) -> float:
        self.model.eval()
        return self.run_one_epoch(TrainTransformer.Mode.Val, epoch, self.val_loader)

    def configure_optimizers(self, lr: float) -> tuple[torch.optim.Optimizer, None]:
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = None
        return optimizer, scheduler

    @staticmethod
    def color_str(message: str, color: Color) -> str:
        return f"\033[{color.value}m{message}\033[00m" if isinstance(color, TrainTransformer.Color) else message

    def run_one_epoch(self, mode: Mode, epoch: int, loader: DataLoader, gamma: float = (5**0.5 - 1) / 2) -> float:
        epoch_loss, discounted_loss = 0, 1.5 - 0.5 * min(epoch / 200, 1)
        progress = tqdm(loader, desc=f"{mode.name:5} epoch {epoch:3}", total=len(loader), colour="cyan")
        self.optim.zero_grad()

        for i, X in enumerate(progress, 1):
            X = X.to(self.device)

            logits, Y = self.model(X)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))

            epoch_loss += loss.item()
            discounted_loss = gamma * discounted_loss + (1 - gamma) * loss.item()

            if mode == TrainTransformer.Mode.Train:
                loss.backward()
                if i % self.accum_grad == 0 or i == len(loader):
                    self.optim.step()
                    self.optim.zero_grad()

            if mode == TrainTransformer.Mode.Train:
                progress.set_postfix_str(self.color_str(f"loss={discounted_loss:.3f}", TrainTransformer.Color.Cyan))
            else:
                progress.set_postfix_str(
                    self.color_str(f"loss={epoch_loss / (progress.n + 1):.3f}", TrainTransformer.Color.Green)
                )

        progress.close()

        epoch_loss /= len(loader)
        self.logger.add_scalar(f"{mode.name}/Loss", epoch_loss, epoch)

        return epoch_loss

    def set_loader(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader

    def save(self, epoch: int, loss: float) -> None:
        self.model.save_transformer_checkpoint(f"{self.save_dir}/{epoch:02x}_{loss:.4f}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MaskGIT")
    # TODO2:check your dataset path is correct
    parser.add_argument("--train_d_path", type=str, default="./dataset/train/", help="Training Dataset Path")
    parser.add_argument("--val_d_path", type=str, default="./dataset/val/", help="Validation Dataset Path")
    parser.add_argument("--checkpoint-path", type=str, default="./saved_models/last.pt", help="Path to checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Which device the training is on.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for training.")
    parser.add_argument(
        "--partial", type=float, default=1.0, help="Number of epochs to train (default: 50)"
    )  # decrepated
    parser.add_argument("--accum-grad", type=int, default=10, help="Number for gradient accumulation.")

    # you can modify the hyperparameters
    parser.add_argument("--epochs", type=int, default=216, help="Number of epochs to train (default: 216)")
    parser.add_argument(
        "--save-per-epoch", type=int, default=1, help="Save CKPT per ** epochs(defcault: 1)"
    )  # decrepated
    parser.add_argument("--start-from-epoch", type=int, default=0, help="Start from epoch (default: 0)")
    parser.add_argument("--ckpt-interval", type=int, default=8, help="Minimum epochs between checkpoints (default: 8)")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate (default: 1e-4)")

    parser.add_argument(
        "--MaskGitConfig", type=str, default="config/MaskGit.yml", help="Configurations for TransformerVQGAN"
    )

    parser.add_argument("--save-dir", type=str, default="saved_models", help="Path to save checkpoints.")
    parser.add_argument("--logdir", type=str, default="logs", help="Path to save logs.")

    args = parser.parse_args()

    MaskGit_CONFIGS = yaml.safe_load(open(args.MaskGitConfig, "r"))
    train_transformer = TrainTransformer(args, MaskGit_CONFIGS)

    train_dataset = LoadTrainData(root=args.train_d_path, partial=args.partial)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=True,
    )

    val_dataset = LoadTrainData(root=args.val_d_path, partial=args.partial)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
        shuffle=False,
    )

    # TODO2 step1-5:
    best, best_epoch = np.inf, 0
    train_transformer.set_loader(train_loader, val_loader)
    for epoch in range(args.start_from_epoch + 1, args.epochs + 1):
        train_transformer.train_one_epoch(epoch)
        loss = train_transformer.eval_one_epoch(epoch)
        if loss < best and epoch - best_epoch >= args.ckpt_interval:
            best, best_epoch = loss, epoch
            train_transformer.save(epoch, loss)
