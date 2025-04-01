from os import path

from torch import save
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import Mode, get_args, get_train_args, print_segment, run_epoch


@print_segment()
def train_net(net, loader_train, loader_val, epochs, lr, logger, save_dir, device, **_):
    optim = Adam(net.parameters(), lr=lr)
    sched = ReduceLROnPlateau(optim, mode="max", factor=10**-0.5, patience=0, threshold=0.01)
    logger.add_scalar(f"{net.name}/Learning Rate", lr, 0)
    best_dice = 0

    for epoch in range(epochs):
        run_epoch(net, loader_train, device, logger, epoch, epochs, optim, mode=Mode.Train)
        dice = run_epoch(net, loader_val, device, logger, epoch, epochs, mode=Mode.Val)

        sched.step(dice)
        if lr != (temp := sched.get_last_lr()[0]):
            lr = temp
            print(f"learning rate update: {lr}")

        logger.add_scalar(f"{net.name}/Learning Rate", lr, epoch + 1)

        if dice >= best_dice:
            best_dice = dice
            save(net.state_dict(), path.join(save_dir, f"{net.name}-{epoch + 1:02x}.pth"))


# TODO: various optimizers & schedulers
def train(nets, loader_train, loader_val, epochs, lr, logger, save_dir, device="cpu"):
    # implement the training function here
    for net in nets:
        train_net(net, loader_train, loader_val, epochs, lr, logger, save_dir, device, title=f"Training {net.name}:")
    logger.close()


def main():
    args = get_args()
    args = get_train_args(args)
    train(*args)


if __name__ == "__main__":
    main()
