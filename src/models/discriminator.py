from typing import Dict

import torch
from torch import nn


class DiscriminatorBlock(nn.Module):
    """DiscriminatorBlock for Discriminator"""

    def __init__(
        self,
        downsampling_layers_num: int = 4,
        features: int = 16,
        downsampling_factor: int = 1,
    ):
        """
        DiscriminatorBlock for Discriminator
        Args:
            downsampling_layers_num: number of downsampling layers
            features: features number after first conv layer
            downsampling_factor: downsampling factor
        """
        super().__init__()
        layers = nn.ModuleDict()  # to prevent alphabetic order
        layers["input_padding"] = nn.ReflectionPad1d(padding=7)
        layers["conv_layer_0"] = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=features, kernel_size=15,),
            nn.LeakyReLU(),
        )
        kernel_size = downsampling_factor * 10 + 1
        current_feature_dim = features
        for layer_idx in range(1, downsampling_layers_num + 1):

            layers[f"downsampling_layer_{layer_idx}"] = nn.Sequential(
                nn.Conv1d(
                    in_channels=min(current_feature_dim, 1024),
                    out_channels=min(
                        current_feature_dim * downsampling_factor, 1024
                    ),
                    kernel_size=kernel_size,
                    stride=downsampling_factor,
                    groups=min(
                        current_feature_dim * downsampling_factor // 4, 256
                    ),
                ),
                nn.LeakyReLU(),
            )
            current_feature_dim *= downsampling_factor

        layers[f"conv_layer_{downsampling_layers_num+1}"] = nn.Sequential(
            nn.Conv1d(
                in_channels=min(current_feature_dim, 1024),
                out_channels=min(current_feature_dim * 2, 1024),
                kernel_size=5,
                padding=2,
            ),
            nn.ReLU(),
        )
        layers["output_layer"] = nn.Conv1d(
            in_channels=min(current_feature_dim * 2, 1024),
            out_channels=1,
            kernel_size=3,
            padding=1,
        )
        self.layers = layers

    def forward(self, inp: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Froward method
        Args:
            inp: downsampled input
        Returns:
            dict with feature mapping and output
        """
        padded = self.layers["input_padding"](inp)
        features = padded
        output_dict = {}
        for key, layer in self.layers.items():
            features = layer(features)
            if "output" not in key:
                output_dict[key + "_ouput"] = features

        score = features
        return {"features": output_dict, "score": score}


class Discriminator(nn.Module):
    """Discriminator model for MelGAN"""

    def __init__(
        self, discriminator_number: int = 3, downsampling_factor: int = 4,
    ):
        """
        Discriminator model for MelGAN
        Consists of several discriminators with
        various downsampling factors.
        Base  discriminator consists only of convolutional layers.
        Args:
            discriminator_number: number of discriminator blocks
            downsampling_factor: downsampling factor for every
                discriminator block.
        """

        super().__init__()
        self.downsampler = nn.AvgPool1d(
            4, stride=2, padding=1, count_include_pad=False,
        )
        self.discriminators = nn.ModuleDict()
        for idx in range(discriminator_number):
            self.discriminators[f"disc_{idx}"] = DiscriminatorBlock(
                downsampling_factor=downsampling_factor,
            )

    def forward(self, inp: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward method

        Args:
            inp: Input audio tensor

        Returns:
            Dict with all discriminators output
        """
        output_dict = {}
        for name, desc in self.discriminators.items():
            output_dict[name + "_output"] = desc(inp)
            inp = self.downsampler(inp)
        return output_dict
