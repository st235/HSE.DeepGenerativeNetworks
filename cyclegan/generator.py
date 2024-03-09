import torch
import torch.nn as nn


class _DownSamplingConvolutionalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 apply_activation: bool = True,
                 **kwargs):
        super().__init__()
        self.__conv = nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
        self.__instance_norm = nn.InstanceNorm2d(out_channels)
        self.__relu = None
        if apply_activation:
            self.__relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.__conv(x)
        fx = self.__instance_norm(fx)
        if self.__relu is not None:
            fx = self.__relu(fx)
        return fx


class _UpSamplingConvolutionalBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 apply_activation: bool = True,
                 **kwargs):
        super().__init__()
        self.__conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.__instance_norm = nn.InstanceNorm2d(out_channels)
        self.__relu = None
        if apply_activation:
            self.__relu = nn.ReLU(inplace=True)

    def forward(self, x):
        fx = self.__conv_transpose(x)
        fx = self.__instance_norm(fx)
        if self.__relu is not None:
            fx = self.__relu(fx)
        return fx


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.__block = nn.Sequential(
            _DownSamplingConvolutionalBlock(channels, channels, add_activation=True, kernel_size=3, padding=1),
            _DownSamplingConvolutionalBlock(channels, channels, add_activation=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.__block(x)


class Generator(nn.Module):
    def __init__(self,
                 img_channels: int,
                 num_features: int = 64,
                 num_residuals: int = 6):
        """
        Generator consists of 2 layers of down sampling/encoding layer,
        followed by 6 residual blocks for 128 × 128 training images
        and then 3 up sampling/decoding layer.

        There are multiple architecture variants:
        - smaller one with 6 residual blocks: c7s1–64, d128, d256, R256, R256, R256, R256, R256, R256, u128, u64,
        and c7s1–3.
        - bigger one with 9 residual blocks: c7s1–64, d128, d256, R256, R256, R256, R256, R256, R256, R256, R256,
        R256, u128, u64, and c7s1–3.
        """
        super().__init__()
        self.__initial_layer = nn.Sequential(
            nn.Conv2d(
                img_channels,
                num_features,
                kernel_size=7,
                stride=1,
                padding=3,
                padding_mode="reflect",
            ),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )

        self.__down_sampling_layers = nn.ModuleList(
            [
                _DownSamplingConvolutionalBlock(
                    num_features,
                    num_features * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                _DownSamplingConvolutionalBlock(
                    num_features * 2,
                    num_features * 4,
                    is_downsampling=True,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
            ]
        )

        self.__residual_layers = nn.Sequential(
            *[_ResidualBlock(num_features * 4) for _ in range(num_residuals)]
        )

        self.__up_sampling_layers = nn.ModuleList(
            [
                _UpSamplingConvolutionalBlock(
                    num_features * 4,
                    num_features * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                _UpSamplingConvolutionalBlock(
                    num_features * 2,
                    num_features * 1,
                    is_downsampling=False,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
            ]
        )

        self.__flattening_layer = nn.Conv2d(
            num_features * 1,
            img_channels,
            kernel_size=7,
            stride=1,
            padding=3,
            padding_mode="reflect",
        )

    def forward(self, x):
        fx = self.__initial_layer(x)
        for layer in self.__down_sampling_layers:
            fx = layer(fx)
        fx = self.__residual_layers(fx)
        for layer in self.__up_sampling_layers:
            fx = layer(fx)
        return torch.tanh(self.__flattening_layer(fx))


