import torch.nn as nn
import torch
import numpy as np

__all__ = ['Conv']


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=in_channels,
                      kernel_size=kernel_size, 
                      padding=padding,
                      groups=in_channels), 
            nn.Conv1d(in_channels=in_channels, 
                      out_channels=out_channels,
                      kernel_size=1), 
            nn.LeakyReLU(negative_slope=0.1, 
                         inplace=True)
            )

    def forward(self, x):
        return self.block(x)


class Conv(nn.Module):
    """ Convolutional architecture with variable receptive field.
    """

    def __init__(self, 
                 receptive_field, 
                 points_per_unit, 
                 num_layers, 
                 num_channels):
        super(Conv, self).__init__()

        kernel_size = self._compute_kernel_size(receptive_field, 
                                                points_per_unit, 
                                                num_layers)

        print(f'kernel size = {kernel_size}')
        padding = kernel_size // 2

        layers = []

        for _ in range(num_layers):
            layers += [DepthwiseSeparableConv1d(num_channels, 
                                                num_channels, 
                                                kernel_size, 
                                                padding)]

        self.net = nn.Sequential(*layers)

        self.num_halving_layers = 0
        self.in_channels = num_channels
        self.out_channels = num_channels
        self.num_channels = num_channels

    def forward(self, x):
        output = self.net(x)
        return output

    @staticmethod
    def _compute_kernel_size(receptive_field, points_per_unit, num_layers):
        receptive_points = receptive_field * points_per_unit
        kernel_size = 1 + (receptive_points - 1) / num_layers
        return int(np.ceil(kernel_size) // 2 * 2 + 1)

    @property
    def num_params(self):
        """Number of parameters in model."""
        return np.sum([torch.tensor(param.shape).prod()
                       for param in self.parameters()])
