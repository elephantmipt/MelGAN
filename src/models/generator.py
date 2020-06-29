import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dilation: int = 1):
        """
        Residual Block
        simple 1d residual block. Shortcut is one conv layer.
        Args:
            dim: number of channels
            dilation: dilation for the first convolution in normal connection
        """
        super().__init__()
        # nothing about it in paper,
        # but in official repo they do residual connection like this
        self.shortcut = nn.utils.weight_norm(nn.Conv1d(dim, dim, kernel_size=1))
        # normal (not residual) connection
        self.layers = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            nn.utils.weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv1d(dim, dim, kernel_size=1)),
        )

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            inp: tensor shape of (dim, seq_length)

        Returns:
            tensor output from residual block
        """
        return self.layers(inp) + self.shortcut(inp)


class Generator(nn.Module):
    def __init__(self, inp_channels: int = 80, normalization: bool = False):
        """
        @TODO
        Args:
            inp_channels:
            normalization:
        """
        super().__init__()
        self.normalization = normalization
        input_channels = [512, 256, 128, 64]
        strides = [8, 8, 2, 2]
        layers = nn.ModuleDict()
        layers["padding"] = nn.ReflectionPad1d(3)
        layers["conv_layer_0"] = nn.utils.weight_norm(nn.Conv1d(inp_channels, 512, kernel_size=7))
        idx = 1
        for inp_ch, st in zip(input_channels, strides):
            layers[f"conv_transpose_{idx}"] = nn.Sequential(
                nn.LeakyReLU(),
                nn.utils.weight_norm(nn.ConvTranspose1d(
                    inp_ch,
                    inp_ch // 2,
                    kernel_size=st * 2,
                    stride=st,
                    padding=st // 2,
                )),
            )
            for i in range(3):
                layers[f"res_block_{idx+1}"] = ResidualBlock(inp_ch // 2, dilation=3 ** i)
            idx += 2

        layers[f"conv_layer_{idx}"] = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            nn.utils.weight_norm(nn.Conv1d(32, 1, kernel_size=7, stride=1)),
            nn.Tanh(),
        )
        self.layers = layers

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward method
        Args:
            inp: MEL spectrogram

        Returns:
            generator output
        """
        inp = (inp + 5.0) / 5.0  # roughly normalize spectrogram
        for _name, layer in self.layers.items():
            inp = layer(inp)
        return inp
