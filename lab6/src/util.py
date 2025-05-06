from pathlib import Path

import torch
from ddpm import DDPM


def save_checkpoint(model, optimizer, path, epoch, latest=False):
    name = "latest.pth" if latest else f"checkpoint_{epoch}.pth"
    save_dir = Path(path, name)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, save_dir)


def load_checkpoint(model, optimizer, path, device):
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["epoch"]


def load_model(weight_path, device):
    model = DDPM().to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint["epoch"]
