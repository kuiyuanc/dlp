# Implement your UNet model here
# ref: https://github.com/milesial/Pytorch-UNet.git
from torch.nn import BatchNorm2d, Conv2d, Module, Sequential

from models.blocks import DoubleConv2d, Down, Up


# TODO: add Dropout for robustness
class UNet(Module):
    def __init__(self, in_channels, out_channels, *, batch_norm, keep_dim=True, name="UNet"):
        super(UNet, self).__init__()

        self.in_conv = DoubleConv2d(in_channels, 64, batch_norm, keep_dim)
        self.down1 = Down(64, 128, batch_norm, keep_dim)
        self.down2 = Down(128, 256, batch_norm, keep_dim)
        self.down3 = Down(256, 512, batch_norm, keep_dim)
        self.down4 = Down(512, 1024, batch_norm, keep_dim)

        self.up1 = Up(1024, 512, 512, batch_norm, keep_dim)
        self.up2 = Up(512, 256, 256, batch_norm, keep_dim)
        self.up3 = Up(256, 128, 128, batch_norm, keep_dim)
        self.up4 = Up(128, 64, 64, batch_norm, keep_dim)

        self.out_conv = Sequential(Conv2d(64, out_channels, kernel_size=1))
        if batch_norm:
            self.out_conv.append(BatchNorm2d(out_channels))

        self.name = name

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
