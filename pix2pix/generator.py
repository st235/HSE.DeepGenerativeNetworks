import torch
from torch import nn
from torch.nn import functional as F

class _EncoderBlock(nn.Module):
    """U-Net Encoder block.

    Each block in the encoder is: Convolution -> Batch normalization -> Leaky ReLU.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 apply_batch_norm: bool = True):
        super().__init__()
        self.__conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.__batch_norm = None
        if apply_batch_norm:
            self.__batch_norm = nn.BatchNorm2d(out_channels)

        self.__leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.__conv(x)
        if self.__batch_norm is not None:
            fx = self.__batch_norm(fx)
        fx = self.__leaky_relu(fx)

        return fx


class _DecoderBlock(nn.Module):
    """U-Net Decoder block.

    Each block in the decoder is:
    Transposed convolution -> Batch normalization -> Dropout (applied to the first 3 blocks) -> ReLU.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 4,
                 stride: int = 2,
                 padding: int = 1,
                 apply_dropout: bool = False):
        super().__init__()
        self.__conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.__batch_norm = nn.BatchNorm2d(out_channels)

        self.__dropout = None
        if apply_dropout:
            self.__dropout = nn.Dropout2d(p=0.5, inplace=True)

        self.__relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.__conv_transpose(x)
        fx = self.__batch_norm(fx)
        if self.__dropout is not None:
            fx = self.__dropout(fx)
        fx = self.__relu(fx)

        return fx


class UnetGenerator(nn.Module):
    """U-Net-like model.

    The generator of pix2pix cGAN is a modified U-Net.
    A U-Net consists of an encoder (down sampler) and decoder (up sampler).
    """

    def __init__(self, ):
        super().__init__()

        self.__encoder1 = nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1)
        self.__encoder2 = _EncoderBlock(64, 128)
        self.__encoder3 = _EncoderBlock(128, 256)
        self.__encoder4 = _EncoderBlock(256, 512)
        self.__encoder5 = _EncoderBlock(512, 512)
        self.__encoder6 = _EncoderBlock(512, 512)
        self.__encoder7 = _EncoderBlock(512, 512)
        self.__encoder8 = _EncoderBlock(512, 512, apply_batch_norm=False)

        self.__decoder8 = _DecoderBlock(512, 512, apply_dropout=True)
        self.__decoder7 = _DecoderBlock(2 * 512, 512, apply_dropout=True)
        self.__decoder6 = _DecoderBlock(2 * 512, 512, apply_dropout=True)
        self.__decoder5 = _DecoderBlock(2 * 512, 512)
        self.__decoder4 = _DecoderBlock(2 * 512, 256)
        self.__decoder3 = _DecoderBlock(2 * 256, 128)
        self.__decoder2 = _DecoderBlock(2 * 128, 64)
        self.__decoder1 = nn.ConvTranspose2d(2 * 64, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        e1 = self.__encoder1(x)
        e2 = self.__encoder2(e1)
        e3 = self.__encoder3(e2)
        e4 = self.__encoder4(e3)
        e5 = self.__encoder5(e4)
        e6 = self.__encoder6(e5)
        e7 = self.__encoder7(e6)
        e8 = self.__encoder8(e7)

        # decoder forward + skip connections
        d8 = self.__decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)
        d7 = self.__decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)
        d6 = self.__decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)
        d5 = self.__decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)
        d4 = self.__decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)
        d3 = self.__decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)
        d2 = F.relu(self.__decoder2(d3))
        d2 = torch.cat([d2, e1], dim=1)
        d1 = self.__decoder1(d2)

        return torch.tanh(d1)


class GeneratorLoss(nn.Module):
    def __init__(self,
                 alpha: int = 100):
        super().__init__()
        self.__alpha = alpha
        self.__bce = nn.BCEWithLogitsLoss()
        self.__l1 = nn.L1Loss()

    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.__bce(fake_pred, fake_target) + self.__alpha * self.__l1(fake, real)
        return loss
