import torch.nn as nn
from diffusers.models.unets.unet_2d import UNet2DModel


class DDPM(nn.Module):
    def __init__(self, num_classes=24, dim=512):
        super().__init__()

        channel = dim // 4
        block_out_channels = (channel, channel, channel * 2, channel * 2, channel * 4, channel * 4)
        down_block_types = ("DownBlock2D","DownBlock2D","DownBlock2D","DownBlock2D","AttnDownBlock2D","DownBlock2D")
        up_block_types = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D")

        self.ddpm = UNet2DModel(
            sample_size=64,
            in_channels=3,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=block_out_channels,
            down_block_types=down_block_types,
            up_block_types=up_block_types,
            class_embed_type="identity",
        )
        self.embed = nn.Linear(num_classes, dim)

    def forward(self, x, t, label):
        embed = self.embed(label)
        return self.ddpm(x, t, embed).sample
