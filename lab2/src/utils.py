import argparse
import random
from enum import Enum, IntEnum, auto
from os import listdir, makedirs, path

import numpy as np
import torch
from models import BinaryClassify2d, ResNet34_UNet, UNet
from oxford_pet import data_loader
from torch import cuda, no_grad
from torch.nn import BCELoss, Module
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


class Mode(Enum):
    Train = 0
    Val = 1
    Test = 2


class Color(IntEnum):
    Red = 91
    Green = auto()
    Yellow = auto()
    Blue = auto()
    Pink = auto()
    Cyan = auto()
    White = auto()


def color_str(message, color):
    return f"\033[{color.value}m{message}\033[00m" if isinstance(color, Color) else message


def color_print(message, color):
    print(color_str(message, color))


def print_divider(char="=", length=72):
    print(char * length)


def print_dict(d):
    for k, v in d.items():
        print(f"{k}: {v}")


def print_segment(title=""):
    def outer(func):
        def wrapper(*args, **kwargs):
            color_print(kwargs.get("title", title), Color.Blue)
            print_divider()
            func(*args, **kwargs)
            print_divider()

        return wrapper

    return outer


def print_common_args(batch_size, batch_norm, seed):
    kwargs = {"batch size": batch_size, "batch normalization": batch_norm, "random seed": seed}
    print_dict(kwargs)


@print_segment("Training configuration:")
def print_eval_args(batch_size, batch_norm, seed):
    print_common_args(batch_size, batch_norm, seed)


@print_segment("Training configuration:")
def print_train_args(batch_size, epochs, lr, batch_norm, seed):
    print_common_args(batch_size, batch_norm, seed)
    print_dict({"number of epochs": epochs, "inital learning rate": lr})


def get_common_args(args):
    make_deterministic(args.seed)

    hparams = (f"batch-norm={args.batch_norm}", f"batch={args.batch_size}")
    save_dir = path.join(args.save_dir, *hparams)
    makedirs(save_dir, exist_ok=True)

    device = "cuda:0" if args.gpu and cuda.is_available() else "cpu"

    net_kwargs = {"in_channels": 3, "out_channels": 1, "batch_norm": args.batch_norm, "keep_dim": args.keep_dim}
    match args.nets:
        case "u":
            nets = [UNet(**net_kwargs)]
        case "mix":
            nets = [ResNet34_UNet(**net_kwargs)]
        case "all":
            nets = [ResNet34_UNet(**net_kwargs), UNet(**net_kwargs)]
        case _:
            raise ValueError("legal values for nets: 'u', 'mix', 'all'")
    nets = tuple(map(BinaryClassify2d, nets))
    nets = tuple(map(Module.to, nets, [device] * len(nets)))

    return nets, save_dir, device, hparams


def get_infer_args(args):
    nets, loader, device, hparams = get_eval_args(args, "test")
    output_dir = path.join(args.output_dir, *hparams)
    makedirs(output_dir, exist_ok=True)
    return nets, loader, output_dir, device


def get_eval_args(args, mode, verbose=True):
    nets, save_dir, device, hparams = get_common_args(args)
    nets = load_nets(nets, save_dir, device, args.model)
    loader = data_loader(args.dataset, mode, shuffle=False)
    if verbose:
        print_eval_args(args.batch_size, args.batch_norm, args.seed)
    return nets, loader, device, hparams


def get_train_args(args):
    nets, save_dir, device, hparams = get_common_args(args)

    loader_train = data_loader(args.dataset, "train", batch_size=args.batch_size)
    loader_val = data_loader(args.dataset, "valid", shuffle=False)

    lr = args.learning_rate

    log_dir = path.join(args.log_dir, *hparams)
    logger = SummaryWriter(log_dir)

    print_train_args(args.batch_size, args.epochs, args.learning_rate, args.batch_norm, args.seed)

    return nets, loader_train, loader_val, args.epochs, lr, logger, save_dir, device


# TODO: configuration of resize in preprocessing
def get_args():
    parser = argparse.ArgumentParser(description="Train the neural networks on images and target masks")
    parser.add_argument("--dataset", type=str, default="dataset/oxford-iiit-pet", help="path of the input data")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="number of epochs")
    parser.add_argument("--batch-size", "-b", type=int, default=1, help="batch size")
    parser.add_argument("--learning-rate", "-lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--nets", "-n", type=str, default="all", help="network architecture")
    parser.add_argument(
        "--batch-norm", "-bn", action="store_true", help="add batch normalization for each convolutional layer"
    )
    parser.add_argument("--keep-dim", "-kd", action="store_true", help="keep input dimension in convolutional layers")
    parser.add_argument("--seed", "-s", type=int, default=42, help="random seed")
    parser.add_argument("--save-dir", "-sd", type=str, default="saved_models", help="path to save checkpoints")
    parser.add_argument("--log-dir", "-ld", type=str, default="logs", help="path to save logs")
    parser.add_argument("--output-dir", '-o', type=str, default="data", help="path to save inference results")
    parser.add_argument("--gpu", "-g", action="store_true", help="use gpu")
    parser.add_argument("--model", "-m", type=str, default=None, help="path to the stored model weight")

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda.manual_seed_all(seed)


def make_deterministic(seed=42):
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def log(logger, name, mode, loss, dice, step):
    logger.add_scalar(f"{name}/{mode.name}/Loss", loss, step)
    logger.add_scalar(f"{name}/{mode.name}/Dice Score", dice, step)


def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    pred_mask = pred_mask >= 0.5
    gt_mask = gt_mask >= 0.5
    common = (pred_mask == gt_mask).sum(dim=(-2, -1))
    total = pred_mask.size(-2) * pred_mask.size(-1) + gt_mask.size(-2) * gt_mask.size(-1)
    return (2 * common) / total


# TODO: various loss
def forward(net, loader, device, mode, logger=None, epoch=0, epochs=1, optim=None, gamma=0.618):
    epoch_loss, epoch_dice = 0, 0
    discounted_loss, discounted_dice = 0.7 - 0.4 * min(epoch / 6, 1), 0.5 + 0.4 * min(epoch / 6, 1)
    progress = tqdm(loader, desc=f"{mode.name:5} epoch {epoch + 1:2}/{epochs:2}", total=len(loader), colour="cyan")
    last = epoch * len(loader.dataset)
    steps = range(last + loader.batch_size, last + len(loader.dataset) + 1, loader.batch_size)
    criterion = BCELoss()

    for step, batch in zip(steps, progress):
        X, Y = batch["image"].to(device).float(), batch["mask"].to(device).float()
        pred = net(X)
        loss = criterion(pred, Y)
        dice = dice_score(pred, Y).mean().item()

        epoch_loss += loss.item()
        epoch_dice += dice
        discounted_loss = gamma * discounted_loss + (1 - gamma) * loss.item()
        discounted_dice = gamma * discounted_dice + (1 - gamma) * dice

        if mode == Mode.Train:
            assert optim
            optim.zero_grad()
            loss.backward()
            optim.step()
            log(logger, net.name, mode, loss.item(), dice, step)

        if mode == Mode.Train:
            colored_dice = color_str(f"dice={discounted_dice:.3f}", Color.Cyan)
            progress.set_postfix_str(f"loss={discounted_loss:.3f}, {colored_dice}")
        else:
            colored_dice = color_str(f"dice={epoch_dice / (progress.n + 1):.3f}", Color.Green)
            progress.set_postfix_str(f"loss={epoch_loss / (progress.n + 1):.3f}, {colored_dice}")

    if mode == Mode.Val:
        log(logger, net.name, mode, epoch_loss / len(loader), epoch_dice / len(loader), epoch + 1)

    progress.close()

    return epoch_dice / len(loader)


def run_epoch(net, loader, device, logger=None, epoch=0, epochs=1, optim=None, *, mode):
    if mode == Mode.Train and optim:
        net.train()
        dice = forward(net, loader, device, mode, logger, epoch, epochs, optim)
    else:
        net.eval()
        with no_grad():
            dice = forward(net, loader, device, mode, logger, epoch, epochs)
    return dice


def load_nets(nets, save_dir, device="cpu", model=None):
    weights = listdir(save_dir)
    for net in nets:
        if model and path.exists(path.join(save_dir, model)):
            model = path.join(save_dir, model)
        else:
            available = filter(lambda x: x.startswith(net.name), weights)
            model = path.join(save_dir, max(available))
        state = torch.load(model, map_location=device)
        net.load_state_dict(state)
    return nets


# TODO: visualization
