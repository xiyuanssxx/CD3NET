"""
---
title: U-Net model for Denoising Diffusion Probabilistic Models (DDPM)
summary: >
  UNet model for Denoising Diffusion Probabilistic Models (DDPM)
---

# U-Net model for [Denoising Diffusion Probabilistic Models (DDPM)](index.html)

This is a [U-Net](../../unet/index.html) based model to predict noise
$\textcolor{lightgreen}{\epsilon_\theta}(x_t, t)$.

U-Net is a gets it's name from the U shape in the model diagram.
It processes a given image by progressively lowering (halving) the feature map resolution and then
increasing the resolution.
There are pass-through connection at each resolution.

![U-Net diagram from paper](../../unet/unet.png)

This implementation contains a bunch of modifications to original U-Net (residual blocks, multi-head attention)
 and also adds time-step embeddings $t$.
"""

import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

from labml_helpers.module import Module


class Swish(Module):
    """
    ### Swish actiavation function

    $$x \cdot \sigma(x)$$
    """

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels
        # First linear layer
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        # [same as those from the transformer](../../transformers/positional_encoding.html)
        #
        # \begin{align}
        # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
        # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
        # \end{align}
        #
        # where $d$ is `half_dim`
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        #
        return emb


class ResidualBlock(Module):
    """
    ### Residual block

    A residual block has two convolution layers with group normalization.
    Each resolution is processed with two residual blocks.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, n_groups: int = 32):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of input channels
        * `time_channels` is the number channels in the time step ($t$) embeddings
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()
        # Group normalization and the first convolution layer
        self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # Group normalization and the second convolution layer
        self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()

        # Linear layer for time embeddings
        self.time_emb = nn.Linear(time_channels, out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer
        h = self.conv1(self.act1(self.norm1(x)))
        # Add time embeddings
        h += self.time_emb(t)[:, :, None, None]
        # Second convolution layer
        h = self.conv2(self.act2(self.norm2(h)))

        # Add the shortcut connection and return
        return h + self.shortcut(x)


class AttentionBlock(Module):
    """
    ### Attention block

    This is similar to [transformer multi-head attention](../../transformers/mha.html).
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

        #
        return res


class DownBlock(Module):
    """
    ### Down block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the first half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):

        x = self.attn(x)
        x = self.res(x, t)
        #####先atten在归一
        # x = self.leaky_relu(x)
        # x = self.pixel_norm(x)
        #######
        return x


class UpBlock(Module):
    """
    ### Up block

    This combines `ResidualBlock` and `AttentionBlock`. These are used in the second half of U-Net at each resolution.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(Module):
    """
    ### Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another `ResidualBlock`.
    This block is applied at the lowest resolution of the U-Net.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Scale up the feature map by $2 \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Scale down the feature map by $\frac{1}{2} \times$
    """

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.pixel_norm = PixelNormal()
        # self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 1), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        return self.conv(self.leaky_relu(self.pixel_norm(x)))
        # return self.conv(x)


class UNet(Module):
    """
    ## U-Net
    """

    def __init__(self, eeg_channels: int = 3, n_channels: int = 64,
                 # ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 4, 2, 2),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `eeg_channels` is the number of channels in the image. $3$ for RGB.
        * `n_channels` is number of channels in the initial feature map that we transform the image into
        * `ch_mults` is the list of channel numbers at each resolution. The number of channels is `ch_mults[i] * n_channels`
        * `is_attn` is a list of booleans that indicate whether to use attention at each resolution
        * `n_blocks` is the number of `UpDownBlocks` at each resolution
        """
        super().__init__()

        # Number of resolutions
        n_resolutions = len(ch_mults)

        # Project image into feature map
        self.image_proj = nn.Conv2d(eeg_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = TimeEmbedding(n_channels * 4)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(Downsample(in_channels))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4, )

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(Upsample(in_channels))

        # Combine the set of modules
        self.up = nn.ModuleList(up)

        # Final normalization and convolution layer
        self.norm = nn.GroupNorm(8, n_channels)
        self.act = Swish()
        self.final = nn.Conv2d(in_channels, eeg_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor, debug=False):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """

        # Get time-step embeddings
        t = self.time_emb(t)

        # Get image projection
        x = self.image_proj(x)

        # `h` will store outputs at each resolution for skip connection
        h = [x]
        # # First half of U-Net
        for m in self.down:
            if debug:
                print('down sampling x size before m is {}'.format(x.size()))
            x = m(x, t)
            if debug:
                print('down sampling x size after m is {}'.format(x.size()))
            h.append(x)


        # Middle (bottom)
        x = self.middle(x, t)

        # # Second half of U-Net
        for m in self.up:
            if isinstance(m, Upsample):
                if debug:
                    print('up sampling x size before m is {}'.format(x.size()))
                x = m(x, t)
                if debug:
                    print('up sampling x size after m is {}'.format(x.size()))
            else:
                # print('skip connection')
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                if debug:
                    print('x size is {}'.format(x.size()))
                    print('s size is {}'.format(s.size()))
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, t)


        # Final normalization and convolution
        return self.final(self.act(self.norm(x)))

import torch
import torch.nn as nn
from torch.nn import Module, Conv2d, Sequential
# from src.unet_eeg import UNet
class FromData(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(FromData, self).__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                       stride=stride, padding=padding)


class ToData(FromData):
    pass
#/Ã¯Â¼Å¸

class UNetBlock(Module):
    def __init__(self, in_channels, out_channels, channel_exp=5):
        super(UNetBlock, self).__init__()
        channels = int(2 ** channel_exp)
        self.dc1 = Sequential(
            SeparableConv2d(in_channels, channels, 3, 2, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.dc2 = Sequential(
            SeparableConv2d(channels, channels * 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.dc3 = Sequential(
            SeparableConv2d(channels * 2, channels * 2, 3, 2, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.dc4 = Sequential(
            SeparableConv2d(channels * 2, channels * 4, 3, 2, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.uc1 = Sequential(
            nn.Upsample(scale_factor=2),
            SeparableConv2d(channels * 4, channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.uc2 = Sequential(
            nn.Upsample(scale_factor=2),
            SeparableConv2d(channels * 4, channels * 2, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.uc3 = Sequential(
            nn.Upsample(scale_factor=2),
            SeparableConv2d(channels * 4, channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())
        self.uc4 = Sequential(
            nn.Upsample(scale_factor=2),
            SeparableConv2d(channels * 2, out_channels, 3, 1, 1),
            nn.LeakyReLU(0.2),
            PixelNormal())

    def forward(self, x):
        # down sampling
        # print('input-x',x.shape)

        d1 = self.dc1(x)
        # print('d1',d1.shape)
        d2 = self.dc2(d1)
        # print('d2',d2.shape)
        d3 = self.dc3(d2)
        # print('d3',d3.shape)
        d4 = self.dc4(d3)
        # print('d4',d4.shape)
        # up sampling
        u1 = self.uc1(d4)
        # print('u1',u1.shape)
        #=======
        avg_pool_layer = nn.AdaptiveMaxPool2d((28,1))
        u1 = avg_pool_layer(u1)
        #=======
        u2 = self.uc2(torch.cat([u1, d3], dim=1))
        # print('u2',u2.shape)
        u3 = self.uc3(torch.cat([u2, d2], dim=1))
        # print('u3',u3.shape)
        u4 = self.uc4(torch.cat([u3, d1], dim=1))
        # print('output',u4.shape)
        return u4

# class UNet(nn.Module):
#     def __init__(self, eeg_channels: int, n_channels: int, ch_mults: tuple, is_attn: tuple):
#         super(UNet, self).__init__()
#         self.image_proj = nn.Conv2d(eeg_channels, n_channels, kernel_size=1)
#
#         self.down = nn.ModuleList()
#         for mult, has_attn in zip(ch_mults, is_attn):
#             self.down.append(DownBlock(n_channels, n_channels * mult, n_channels, has_attn))
#             n_channels *= mult
#
#         self.middle = MiddleBlock(n_channels, n_channels)
#
#         self.up = nn.ModuleList()
#         for mult in reversed(ch_mults[1:]):
#             self.up.append(UpBlock(n_channels * 2, n_channels // mult, n_channels, True))
#             n_channels //= mult
#
#         self.norm = nn.InstanceNorm2d(n_channels)
#         self.act = Swish()
#         self.final = nn.Conv2d(n_channels, eeg_channels, kernel_size=1)
#
#         self.time_embedding = TimeEmbedding(n_channels)
#
#     def forward(self, x, t):
#         t_emb = self.time_embedding(t)
#         x = self.image_proj(x)
#
#         skips = []
#         for block in self.down:
#             x = block(x, t_emb)
#             skips.append(x)
#
#         x = self.middle(x, t_emb)
#
#         for block, skip in zip(self.up, reversed(skips)):
#             x = torch.cat([x, skip], dim=1)
#             x = block(x, t_emb)
#
#         x = self.norm(x)
#         x = self.act(x)
#         x = self.final(x)
#
#         return x

class PixelNormal(Module):
    def __init__(self, epsilon=1e-8, normal=2):
        super(PixelNormal, self).__init__()
        self.normal = normal
        self.epsilon = epsilon

    def forward(self, x):
        if self.normal == 2:
            _l = x.pow(2.).mean(dim=1, keepdim=True).add(self.epsilon).sqrt()
        else:
            _l = x.abs().mean(dim=1, keepdim=True).add(self.epsilon)
        return x / _l


class SeparableConv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True):
        # print('')
        super(SeparableConv2d, self).__init__()
        self.depthwise = Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation,
                                groups=in_channels, bias=bias)
        self.pointwise = Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

# class FFPCell(Module):
#     def __init__(self, in_channels, out_channels, channel_exp=5):
#         super(FFPCell, self).__init__()
#         self.block = UNetBlock(in_channels, out_channels, channel_exp)
#         # self.block = UNet(in_channels, out_channels, channel_exp)
#     # UNet
#     # def _forward(self, x, mask):
#     def _forward(self,x):
#         # y = torch.mul(self.block(x), mask)
#         y = self.block(x)
#         y = torch.cat([y, x], dim=1)
#         return y
#
#     def forward(self, *args):
#         return self._forward(args[0])

class FFPCell(Module):
    def __init__(self, in_channels, out_channels, channel_exp=5, n_channels=64,
                 ch_mults=(1, 4, 2, 2), is_attn=(True, True, True, True), n_blocks=2):
        super(FFPCell, self).__init__()
        self.block = UNet(eeg_channels=in_channels, n_channels=n_channels,
                          ch_mults=ch_mults, is_attn=is_attn, n_blocks=n_blocks)


    def _forward(self, x, t):
        y = self.block(x, t)
        y = torch.cat([y, x], dim=1)
        return y

    def forward(self, *args):
        x, t = args
        return self._forward(x, t)


class FFPOutput(Module):
    def __init__(self, in_channels, out_channels):
        super(FFPOutput, self).__init__()
        self.to_data = ToData(in_channels, out_channels, 1, 1, 0)

    def _forward(self, x):
        # return torch.mul(self.to_data(x), mask)
        return self.to_data(x)

    def forward(self, *args):
        return self._forward(args[0])


# class FFPModel(nn.Module):
#     def __init__(self, in_channels, out_channels, cells_out_channels, cells_exp):
#         super(FFPModel, self).__init__()
#         self.list = nn.ModuleList()
#         # print('==============================')
#         for co, ce in zip(cells_out_channels, cells_exp):
#             _cell = FFPCell(in_channels, co, ce)
#             in_channels = in_channels + co
#             self.list.append(_cell)
#         self.list.append(FFPOutput(in_channels, out_channels))
#
#     # def _forward(self, x, mask):
#     def _forward(self,x):
#         for _cell in self.list:
#             # x = _cell(x, mask)
#             x = _cell(x)
#         return x
#
#     def forward(self, *args):
#         # return self._forward(args[0], args[1])
#         return self._forward(args[0])

class FFPModel(nn.Module):
    def __init__(self, in_channels, out_channels, cells_out_channels, cells_exp):
        super(FFPModel, self).__init__()
        self.list = nn.ModuleList()
        for co, ce in zip(cells_out_channels, cells_exp):
            _cell = FFPCell(in_channels, co, ce)
            in_channels = in_channels + co
            self.list.append(_cell)
        self.list.append(FFPOutput(in_channels, out_channels))

    def _forward(self, x, t):
        for _cell in self.list:
            x = _cell(x, t)
        return x

    def forward(self, *args):
        x, t = args
        return self._forward(x, t)

if __name__ == '__main__':
    from torchsummaryX import summary
    _x = torch.ones(size=[10, 3, 128, 128])
    _mask = torch.ones(size=[10, 1, 128, 128])
    _m = FFPModel(3, 3, [30] * 4, [6] * 4)
    summary(_m, _x, _mask)
