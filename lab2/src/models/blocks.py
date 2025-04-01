# ref: https://github.com/milesial/Pytorch-UNet.git
# ref: https://github.com/weiaicunzai/pytorch-cifar100.git
import torch
from torch.nn import BatchNorm2d, Conv2d, ConvTranspose2d, Identity, MaxPool2d, Module, ReLU, Sequential
from torch.nn.functional import pad


class DoubleConv2d(Module):
    def __init__(self, in_channels, out_channels, batch_norm, keep_dim=True, stride=1, residual=False):
        super(DoubleConv2d, self).__init__()

        padding = 1 if keep_dim else 0

        layers = [
            Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, stride=stride),
            ReLU(inplace=True),
            Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
        ]

        if batch_norm:
            layers.insert(1, BatchNorm2d(out_channels))
            layers.insert(4, BatchNorm2d(out_channels))

        self.residual = residual
        if residual and in_channels != out_channels:
            self.shortcut = Sequential(Conv2d(in_channels, out_channels, kernel_size=1, stride=2))
            if batch_norm:
                self.shortcut.append(BatchNorm2d(out_channels))
        else:
            self.shortcut = Identity()

        self.nn = Sequential(*layers)

    def forward(self, x):
        x_down = self.nn(x)
        x_skip = self.shortcut(x)
        x = x_down + x_skip if self.residual else x_down
        return ReLU(inplace=True)(x)


class Down(Module):
    def __init__(self, in_channels, out_channels, batch_norm, keep_dim=True):
        super(Down, self).__init__()

        self.nn = Sequential(MaxPool2d(2), DoubleConv2d(in_channels, out_channels, batch_norm, keep_dim))

    def forward(self, x):
        return self.nn(x)


class Up(Module):
    def __init__(self, in_channels_up, in_channels_skip, out_channels, batch_norm, keep_dim=True):
        super(Up, self).__init__()

        in_channels = in_channels_up // 2 + in_channels_skip
        self.up = ConvTranspose2d(in_channels_up, in_channels_up // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv2d(in_channels, out_channels, batch_norm, keep_dim)

    def forward(self, x_up, x_skip):
        x_up = self.up(x_up)

        delta_2, delta_1 = x_skip.size(-2) - x_up.size(-2), x_skip.size(-1) - x_up.size(-1)
        x_up = pad(x_up, (delta_1 // 2, delta_1 - delta_1 // 2, delta_2 // 2, delta_2 - delta_2 // 2))

        x = torch.cat((x_skip, x_up), dim=-3)

        return self.conv(x)


class MultiDoubleConv2d(Module):
    def __init__(self, num, in_channels, out_channels, batch_norm, keep_dim=True, stride=1, residual=True):
        super(MultiDoubleConv2d, self).__init__()

        assert num > 0
        layers = [DoubleConv2d(in_channels, out_channels, batch_norm, keep_dim, stride, residual=residual)]
        layers.extend(
            DoubleConv2d(out_channels, out_channels, batch_norm, keep_dim, residual=residual) for _ in range(num - 1)
        )
        self.nn = Sequential(*layers)

    def forward(self, x):
        return self.nn(x)
