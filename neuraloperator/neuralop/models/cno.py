from matplotlib.artist import get
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.channel_mlp import ChannelMLP


def get_conv(dim):
    if dim == 1:
        conv = nn.Conv1d
    elif dim == 2:
        conv = nn.Conv2d
    elif dim == 3:
        conv = nn.Conv3d
    else:
        raise NotImplementedError
    return conv


def get_norm(dim):
    if dim == 1:
        norm = nn.BatchNorm1d
    elif dim == 2:
        norm = nn.BatchNorm2d
    elif dim == 3:
        norm = nn.BatchNorm3d
    else:
        raise NotImplementedError
    return norm


class CNO_LReLu(nn.Module):
    def __init__(self, in_size, out_size, dim=1):
        super(CNO_LReLu, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dim = dim
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(x)


class CNOBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, in_size, out_size, dim=1, use_bn=True
    ):
        super(CNOBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.dim = dim

        conv = get_conv(dim)
        self.convolution = conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            bn = get_norm(dim)
            self.batch_norm = bn(self.out_channels)
        else:
            self.batch_norm = nn.Identity()

        self.act = CNO_LReLu(in_size=self.in_size, out_size=self.out_size, dim=dim)

    def forward(self, x):
        x = self.convolution(x)
        x = self.batch_norm(x)
        return self.act(x)


class LiftProjectBlock(nn.Module):
    def __init__(self, in_channels, out_channels, size, dim=1, latent_dim=64):
        super(LiftProjectBlock, self).__init__()
        self.inter_CNOBlock = CNOBlock(
            in_channels=in_channels,
            out_channels=latent_dim,
            in_size=size,
            out_size=size,
            dim=dim,
            use_bn=False,
        )

        conv = get_conv(dim)
        self.convolution = conv(
            in_channels=latent_dim, out_channels=out_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        x = self.inter_CNOBlock(x)
        x = self.convolution(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, size, dim=1, use_bn=True):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.size = size
        self.dim = dim

        conv = get_conv(dim)
        self.convolution1 = conv(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )
        self.convolution2 = conv(
            in_channels=self.channels,
            out_channels=self.channels,
            kernel_size=3,
            padding=1,
        )

        if use_bn:
            bn = get_norm(dim)
            self.batch_norm1 = bn(self.channels)
            self.batch_norm2 = bn(self.channels)
        else:
            self.batch_norm1 = nn.Identity()
            self.batch_norm2 = nn.Identity()

        self.act = CNO_LReLu(in_size=self.size, out_size=self.size, dim=dim)

    def forward(self, x):
        out = self.convolution1(x)
        out = self.batch_norm1(out)
        out = self.act.act(out)
        out = self.convolution2(out)
        out = self.batch_norm2(out)
        return x + out


class ResNet(nn.Module):
    def __init__(self, channels, size, num_blocks, dim=1, use_bn=True):
        super(ResNet, self).__init__()
        self.channels = channels
        self.size = size
        self.num_blocks = num_blocks
        self.dim = dim

        self.res_nets = nn.Sequential(
            *[
                ResidualBlock(channels=channels, size=size, dim=dim, use_bn=use_bn)
                for _ in range(self.num_blocks)
            ]
        )

    def forward(self, x):
        return self.res_nets(x)


class CNO(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        size,
        n_layers,
        n_res=4,
        n_res_neck=4,
        channel_multiplier=16,
        use_bn=True,
        dim=1,
    ):
        super(CNO, self).__init__()

        self.n_layers = int(n_layers)
        self.lift_dim = channel_multiplier // 2
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channel_multiplier = channel_multiplier
        self.dim = dim

        # Features evolution
        self.encoder_features = [self.lift_dim]
        for i in range(self.n_layers):
            self.encoder_features.append(2**i * self.channel_multiplier)

        self.decoder_features_in = self.encoder_features[1:]
        self.decoder_features_in.reverse()
        self.decoder_features_out = self.encoder_features[:-1]
        self.decoder_features_out.reverse()

        for i in range(1, self.n_layers):
            self.decoder_features_in[i] = 2 * self.decoder_features_in[i]

        # Spatial sizes evolution
        self.encoder_sizes = []
        self.decoder_sizes = []
        for i in range(self.n_layers + 1):
            self.encoder_sizes.append(size // 2**i)
            self.decoder_sizes.append(size // 2 ** (self.n_layers - i))

        # Lift and Project blocks
        self.lift = ChannelMLP(
            in_channels=in_channels,
            out_channels=self.encoder_features[0],
            n_layers=1,
            n_dim=self.dim,
        )

        self.project = ChannelMLP(
            in_channels=self.encoder_features[0] + self.decoder_features_out[-1],
            out_channels=out_channels,
            n_layers=1,
            n_dim=self.dim,
        )

        # Encoder, ED Linker and Decoder networks
        self.encoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i + 1],
                    in_size=self.encoder_sizes[i],
                    out_size=self.encoder_sizes[i + 1],
                    dim=dim,
                    use_bn=use_bn,
                )
                for i in range(self.n_layers)
            ]
        )

        self.ED_expansion = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.encoder_features[i],
                    out_channels=self.encoder_features[i],
                    in_size=self.encoder_sizes[i],
                    out_size=self.decoder_sizes[self.n_layers - i],
                    dim=dim,
                    use_bn=use_bn,
                )
                for i in range(self.n_layers + 1)
            ]
        )

        self.decoder = nn.ModuleList(
            [
                CNOBlock(
                    in_channels=self.decoder_features_in[i],
                    out_channels=self.decoder_features_out[i],
                    in_size=self.decoder_sizes[i],
                    out_size=self.decoder_sizes[i + 1],
                    dim=dim,
                    use_bn=use_bn,
                )
                for i in range(self.n_layers)
            ]
        )

        # ResNet Blocks
        self.res_nets = nn.ModuleList(
            [
                ResNet(
                    channels=self.encoder_features[l],
                    size=self.encoder_sizes[l],
                    num_blocks=n_res,
                    dim=dim,
                    use_bn=use_bn,
                )
                for l in range(self.n_layers)
            ]
        )

        self.res_net_neck = ResNet(
            channels=self.encoder_features[self.n_layers],
            size=self.encoder_sizes[self.n_layers],
            num_blocks=n_res_neck,
            dim=dim,
            use_bn=use_bn,
        )

    def forward(self, x):
        x = self.lift(x)
        skip = []

        # Encoder
        for i in range(self.n_layers):
            y = self.res_nets[i](x)
            skip.append(y)
            x = self.encoder[i](x)

        # Bottleneck
        x = self.res_net_neck(x)

        # Decoder
        for i in range(self.n_layers):
            if i == 0:
                x = self.ED_expansion[self.n_layers - i](x)
            else:
                x = torch.cat((x, self.ED_expansion[self.n_layers - i](skip[-i])), 1)
            x = self.decoder[i](x)

        # Projection
        x = torch.cat((x, self.ED_expansion[0](skip[0])), 1)
        x = self.project(x)
        return x


class CNO1d(CNO):
    def __init__(self, *args, **kwargs):
        super(CNO1d, self).__init__(*args, dim=1, **kwargs)


class CNO2d(CNO):
    def __init__(self, *args, **kwargs):
        super(CNO2d, self).__init__(*args, dim=2, **kwargs)
