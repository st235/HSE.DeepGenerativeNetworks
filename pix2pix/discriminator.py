import torch
from torch import nn


class _BasicBlock(nn.Module):
    """PatchGAN classifier basic block.

    Each block in the discriminator is: Convolution -> Instance normalization -> Leaky ReLU.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 apply_instance_norm: bool = True):
        super().__init__()
        self.__conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.__instance_norm = None
        if apply_instance_norm:
            self.__instance_norm = nn.InstanceNorm2d(out_channels)
        self.__leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.__conv(x)
        if self.__instance_norm is not None:
            fx = self.__instance_norm(fx)
        fx = self.__leaky_relu(fx)
        return fx


class Discriminator(nn.Module):
    """Pix2Pix Conditional Discriminator."""

    def __init__(self, ):
        super().__init__()
        self.__block1 = _BasicBlock(6, 64, apply_instance_norm=False)
        self.__block2 = _BasicBlock(64, 128)
        self.__block3 = _BasicBlock(128, 256)
        self.__block4 = _BasicBlock(256, 512)
        self.__block5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, cond):
        x = torch.cat([x, cond], dim=1)
        fx = self.__block1(x)
        fx = self.__block2(fx)
        fx = self.__block3(fx)
        fx = self.__block4(fx)
        fx = self.__block5(fx)
        return fx


class DiscriminatorLoss(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.__bce = nn.BCEWithLogitsLoss()

    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        fake_loss = self.__bce(fake_pred, fake_target)
        real_loss = self.__bce(real_pred, real_target)
        loss = (fake_loss + real_loss) / 2
        return loss
