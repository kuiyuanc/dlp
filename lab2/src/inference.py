from os import makedirs, path

import matplotlib.pyplot as plt
from torch import no_grad
from tqdm import tqdm
from utils import get_args, get_infer_args, print_segment


def compare(images, masks, predictions, output_dir, batch_idx):
    batch_size = images.size(0)
    for i in range(batch_size):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2-by-3 grid
        axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

        axes[0].imshow(images[i].int().cpu().permute(1, 2, 0))
        axes[0].set_title("Original Image")

        axes[1].imshow(masks[i].cpu().squeeze(), cmap="gray")
        axes[1].set_title("Ground Truth Mask")

        axes[2].imshow(predictions[i].cpu().squeeze(), cmap="gray")
        axes[2].set_title("Predicted Mask (Raw)")

        binary_pred = (predictions[i] >= 0.5).cpu().squeeze()
        axes[3].imshow(binary_pred, cmap="gray")
        axes[3].set_title("Predicted Mask (Thresholded)")

        axes[4].imshow(images[i].int().cpu().permute(1, 2, 0))
        axes[4].imshow(binary_pred, cmap="jet", alpha=0.5)
        axes[4].set_title("Overlay: Original + Predicted Mask")

        for ax in axes:
            ax.axis("off")

        plt.savefig(f"{output_dir}/{batch_idx * batch_size + i}.png")
        plt.close()


@print_segment()
def inference_net(net, loader, output_dir, device="cpu", **_):
    output_dir = path.join(output_dir, net.name)
    makedirs(output_dir, exist_ok=True)
    with no_grad():
        for i, batch in tqdm(enumerate(loader), desc="Inferencing", total=len(loader), colour="cyan"):
            X, Y = batch["image"].to(device).float(), batch["mask"].to(device).float()
            compare(X, Y, net(X), output_dir, i)


def inference(nets, loader, output_dir, device="cpu"):
    with no_grad():
        for net in nets:
            inference_net(net, loader, output_dir, device, title=f"Inferencing with {net.name}:")


def main():
    args = get_args()
    nets, loader, output_dir, device = get_infer_args(args)
    inference(nets, loader, output_dir, device)


if __name__ == "__main__":
    main()
