# Implement your ResNet34_UNet model here
# ref: https://github.com/weiaicunzai/pytorch-cifar100.git
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, MaxPool2d, Module, ReLU, Sequential

from models.blocks import MultiDoubleConv2d, Up


class ResNet34_UNet(Module):
    def __init__(self, in_channels, out_channels, *, batch_norm, keep_dim=True, name="ResNet34_UNet"):
        super(ResNet34_UNet, self).__init__()

        padding = 3 if keep_dim else 0
        layers = [Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=padding), ReLU(inplace=True)]
        if batch_norm:
            layers.insert(1, BatchNorm2d(64))
        self.conv1 = Sequential(*layers)

        padding = 1 if keep_dim else 0
        self.conv2 = Sequential(
            MaxPool2d(3, stride=2, padding=padding),
            MultiDoubleConv2d(3, 64, 64, batch_norm, keep_dim),
        )

        self.conv3 = MultiDoubleConv2d(4, 64, 128, batch_norm, keep_dim, stride=2)
        self.conv4 = MultiDoubleConv2d(6, 128, 256, batch_norm, keep_dim, stride=2)
        self.conv5 = MultiDoubleConv2d(3, 256, 512, batch_norm, keep_dim, stride=2)

        padding = 1 if keep_dim else 0
        layers = [Conv2d(512, 256, kernel_size=3, padding=padding), ReLU(inplace=True)]
        if batch_norm:
            layers.insert(1, BatchNorm2d(256))
        self.conv6 = Sequential(*layers)

        self.up1 = Up(256, 512, 128, batch_norm, keep_dim)
        self.up2 = Up(128, 256, 64, batch_norm, keep_dim)
        self.up3 = Up(64, 128, 32, batch_norm, keep_dim)
        self.up4 = Up(32, 64, 32, batch_norm, keep_dim)

        layers = [
            ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ReLU(inplace=True),
            ConvTranspose2d(32, 32, kernel_size=2, stride=2),
            ReLU(inplace=True),
            Conv2d(32, out_channels, kernel_size=1),
        ]
        if batch_norm:
            layers.insert(1, BatchNorm2d(32))
            layers.insert(4, BatchNorm2d(32))
            layers.append(BatchNorm2d(out_channels))
        self.out_conv = Sequential(*layers)

        self.name = name

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x = self.conv6(x5)
        x = self.up1(x, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        return self.out_conv(x)
