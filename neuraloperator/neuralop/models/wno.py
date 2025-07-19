from functools import partialmethod
from typing import Tuple, List, Union, Literal

Number = Union[float, int]

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.embeddings import GridEmbeddingND, GridEmbedding2D
from ..layers.spectral_convolution import SpectralConv
from ..layers.padding import DomainPadding
from ..layers.fno_block import FNOBlocks
from ..layers.channel_mlp import ChannelMLP
from ..layers.complex import ComplexValued
from .base_model import BaseModel
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d, WaveConv3d


class WNO(BaseModel, name="WNO"):
    def __init__(
        self,
        n_modes: Tuple[int],
        in_channels: int,
        out_channels: int,
        lifting: int = 40,
        omega: int = 4,
        level: int = 5,
        padding: int = 0,
        hidden_dim: int = 128,
        layers: int = 4,
        wavelet_type: str = "db2",
    ):
        super(WNO, self).__init__()
        self.n_modes = n_modes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lifting = lifting
        self.omega = omega
        self.level = level
        self.layers = layers
        self.padding = padding
        self.wavelet_type = wavelet_type
        self.hidden_dim = hidden_dim

        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()
        wave_convs = {1: WaveConv1d, 2: WaveConv2d, 3: WaveConv3d}
        standard_convs = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
        self.fc0 = nn.Linear(self.in_channels, self.lifting)
        for i in range(self.layers):
            self.conv.append(
                wave_convs[len(self.n_modes)](
                    in_channels=self.lifting,
                    out_channels=self.lifting,
                    level=self.level,
                    size=self.n_modes,
                    wavelet=self.wavelet_type,
                    omega=self.omega,
                )
            )
            self.w.append(
                standard_convs[len(self.n_modes)](self.lifting, self.lifting, 1)
            )

        self.fc1 = nn.Linear(self.lifting, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.out_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        spatial_dims = x.shape[2:]
        x = x.reshape(batch_size, *spatial_dims, -1)
        x = self.fc0(x)
        x = x.reshape(batch_size, self.lifting, *spatial_dims)
        if self.padding != 0:
            x = F.pad(x, (0, self.padding))
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.layers - 1:
                x = F.mish(x)

        if self.padding != 0:
            x = x[:, :, self.padding : -self.padding]
        x = x.reshape(batch_size, *spatial_dims, self.lifting)
        x = self.fc1(x)
        x = F.mish(x)
        x = self.fc2(x)
        x = x.reshape(batch_size, self.out_channels, *spatial_dims)
        return x


class WNO1d(WNO):
    def __init__(
        self,
        width: int,
        in_channels: int,
        out_channels: int,
        lifting: int = 40,
        omega: int = 4,
        level: int = 5,
        padding: int = 0,
        hidden_dim: int = 128,
        layers: int = 4,
        wavelet_type: str = "db2",
    ):
        super().__init__(
            n_modes=(width,),
            in_channels=in_channels,
            out_channels=out_channels,
            lifting=lifting,
            omega=omega,
            level=level,
            padding=padding,
            hidden_dim=hidden_dim,
            layers=layers,
            wavelet_type=wavelet_type,
        )
        self.n_modes_width = width


class WNO2d(WNO):
    def __init__(
        self,
        width: int,
        height: int,
        in_channels: int,
        out_channels: int,
        lifting: int = 40,
        omega: int = 4,
        level: int = 5,
        padding: int = 0,
        hidden_dim: int = 128,
        layers: int = 4,
        wavelet_type: str = "db2",
    ):
        super().__init__(
            n_modes=(width, height),
            in_channels=in_channels,
            out_channels=out_channels,
            lifting=lifting,
            omega=omega,
            level=level,
            padding=padding,
            hidden_dim=hidden_dim,
            layers=layers,
            wavelet_type=wavelet_type,
        )
        self.n_modes_width = width
        self.n_modes_height = height


class WNO3d(WNO):
    def __init__(
        self,
        width: int,
        height: int,
        depth: int,
        in_channels: int,
        out_channels: int,
        lifting: int = 40,
        omega: int = 4,
        level: int = 5,
        padding: int = 0,
        hidden_dim: int = 128,
        layers: int = 4,
        wavelet_type: str = "db2",
    ):
        super().__init__(
            n_modes=(width, height, depth),
            in_channels=in_channels,
            out_channels=out_channels,
            lifting=lifting,
            omega=omega,
            level=level,
            padding=padding,
            hidden_dim=hidden_dim,
            layers=layers,
            wavelet_type=wavelet_type,
        )
        self.n_modes_width = width
        self.n_modes_height = height
        self.n_modes_depth = depth
