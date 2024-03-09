import torch
import torch.nn as nn


class _BasicBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 stride: int):
        """Basic discriminator block.

        Block consists of: Convolution, InstanceNorm, and LeakyReLU with a slope of 0.2.
        """
        super().__init__()

        self.__conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=stride,
                padding=1,
                bias=True,
                padding_mode="reflect",)

        self.__instance_norm = nn.InstanceNorm2d(out_channels)
        self.__leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.__conv(x)
        fx = self.__instance_norm(fx)
        return self.__leaky_relu(fx)


class Discriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 features: list[int] = [64, 128, 256, 512]):
        super().__init__()
        self.__initial_layer = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(_BasicBlock(in_channels, feature, stride=1 if feature == features[-1] else 2))
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels,
                1,
                kernel_size=4,
                stride=1,
                padding=1,
                padding_mode="reflect",
            )
        )
        self.__last_layer = nn.Sequential(*layers)

    def forward(self, x):
        fx = self.__initial_layer(x)
        logits = self.__last_layer(fx)
        return torch.sigmoid(logits)
