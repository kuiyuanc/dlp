from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pretty_errors  # noqa: F401
import torch
from dataset import iclevrDataset
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from evaluator import evaluation_model
from rich.progress import Progress, TimeElapsedColumn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from util import load_model


@torch.no_grad()
def inference(dataloader, noise_scheduler, timesteps, model, eval_model, save_dir=Path("result"), device="cpu"):
    all_results = []
    accuracy = []

    with Progress(*Progress.get_default_columns(), TimeElapsedColumn()) as progress_bar:
        task = progress_bar.add_task("Testing", total=len(dataloader))

        for idx, y in enumerate(dataloader):
            x, y = torch.randn(1, 3, 64, 64).to(device), y.to(device)

            denoising_result = []
            for i, t in enumerate(noise_scheduler.timesteps):
                residual = model(x, t, y)
                x = noise_scheduler.step(residual, t, x).prev_sample
                if i % (timesteps // 10) == 0:
                    denoising_result.append(x)

            accuracy.append(eval_model.eval(x, y))
            progress_bar.update(task, advance=1, description=f"image: {idx + 1}, accuracy: {accuracy[-1]:.4f}")

            denoising_result.append(x)
            denoising_result = (torch.cat(denoising_result) + 1) / 2
            row_image = make_grid(denoising_result, nrow=denoising_result.shape[0])

            save_image(row_image, f"{save_dir}/{idx}.png")
            all_results.append(x)

    all_results = (torch.cat(all_results) + 1) / 2
    all_results = make_grid(all_results, nrow=8)
    save_image(all_results, f"{save_dir}/result.png")

    return accuracy


def test(args, noise_scheduler, model, eval_model, mode):
    loader = DataLoader(iclevrDataset(args.dataset, mode))
    save_dir = Path(args.save_dir, mode)
    accuracy = inference(loader, noise_scheduler, args.timesteps, model, eval_model, save_dir, args.device)
    print(f"{mode} accuracy: {np.mean(accuracy)}")


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="iclevr")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2")
    parser.add_argument("--weight", type=str, default="latest.pth")
    parser.add_argument("--weight-dir", type=str, default="weights")
    parser.add_argument("--save-dir", type=str, default="images")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    weight_path = Path(args.weight_dir, args.weight)
    model, epoch = load_model(weight_path, args.device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.timesteps, beta_schedule=args.beta_schedule)
    eval_model = evaluation_model()

    print("model loaded: epoch", epoch)
    args.save_dir = Path(args.save_dir, f"{epoch=}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    test(args, noise_scheduler, model, eval_model, "test")
    test(args, noise_scheduler, model, eval_model, "new_test")


if __name__ == "__main__":
    main()
