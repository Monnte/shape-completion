"""
This file contains model basd on DiffCompelete from the paper "https://arxiv.org/pdf/2306.16329.pdf".

Code taken from https://github.com/openai/improved-diffusion, and modifed by Peter Zdravecký.
"""

from abc import abstractmethod
from . import logger
import math
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .nn import (
    SiLU,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
    checkpoint,
)


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(self, channels, use_conv, dims=3, factor=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        self.factor = factor
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2] * 2 * self.factor, x.shape[3] * 2 * self.factor, x.shape[4] * 2 * self.factor), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2* self.factor, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(self, channels, use_conv, dims=3, factor=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2*factor if dims != 3 else (2*factor, 2*factor, 2*factor)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=3,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint

        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention()
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), self.use_checkpoint)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b * self.num_heads, -1, qkv.shape[2])
        h = self.attention(qkv)
        h = h.reshape(b, -1, h.shape[-1])
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention.
    """

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (C * 3) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x C x T] tensor after attention.
        """
        ch = qkv.shape[1] // 3
        q, k, v = th.split(qkv, ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        return th.einsum("bts,bcs->bct", weight, v)

    @staticmethod
    def count_flops(model, _x, y):
        """
        A counter for the `thop` package to count the operations in an
        attention operation.

        Meant to be used like:

            macs, params = thop.profile(
                model,
                inputs=(inputs, timestamps),
                custom_ops={QKVAttention: QKVAttention.count_flops},
            )

        """
        b, c, *spatial = y[0].shape
        num_spatial = int(np.prod(spatial))
        # We perform two matmuls with the same number of ops.
        # The first computes the weight matrix, the second computes
        # the combination of the value vectors.
        matmul_ops = 2 * b * (num_spatial ** 2) * c
        model.total_ops += th.DoubleTensor([matmul_ops])


class ProjectionLayer(nn.Module):
    """
    A 1x1 convolutional projection layer.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.project = conv_nd(3, in_channels, in_channels, 1, padding=0)
    
    def forward(self, x):
        return self.project(x)


class DiffCompelete(nn.Module):
    """
    The enhanced DiffCompelete model from the paper "https://arxiv.org/pdf/2306.16329.pdf".

    :param in_channels: channels in the input Tensor.
    :param cond_in_channels: channels in the condition input Tensor.
    :param out_channels: channels in the output Tensor.
    :param dropout: the dropout probability.
    :param attention_resolutions: a list of attention resolutions.
    :param in_scale_factor: the factor to downsample input to match condition.
    :param conv_resample: if True, use convolutional resampling.
    :param use_checkpoint: if True, use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param use_scale_shift_norm: if True, use scale and shift normalization in the ResBlocks.
    """

    def __init__(
        self,
        in_channels,
        cond_in_channels,
        out_channels,
        dropout=0,
        attention_resolutions = [],
        in_scale_factor=0,
        conv_resample=True,
        use_checkpoint=False,
        num_heads=1,
        use_scale_shift_norm=False,
    ):
        super().__init__()

        model_channels=32
        num_res_blocks=3
        channel_mult=(2, 4, 4, 4)
        dims=3

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.use_scale_shift_norm = use_scale_shift_norm
        self.attention_resolutions = attention_resolutions
        self.in_scale_factor = in_scale_factor
        
        self.channel_mult = channel_mult
        self.num_res_blocks = num_res_blocks
        self.model_channels = model_channels
        
        logger.log(f"""Model Parameters:
            in_channels: {self.in_channels}
            out_channels: {self.out_channels}
            cond_in_channels: {cond_in_channels}
            dropout: {self.dropout}
            conv_resample: {self.conv_resample}
            use_checkpoint: {self.use_checkpoint}
            num_heads: {self.num_heads}
            use_scale_shift_norm: {self.use_scale_shift_norm}
            attention_resolutions: {self.attention_resolutions}
            in_scale_factor: {self.in_scale_factor}
            channel_mult: {self.channel_mult}
            num_res_blocks: {self.num_res_blocks}
            model_channels: {self.model_channels}
        """)


        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )


        self.cond_project = nn.ModuleList([])
        self.preprocess = TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1),
                    conv_nd(dims, model_channels, model_channels * channel_mult[0], 3, padding=1),
                )
        self.preprocess_cond = TimestepEmbedSequential(
                    conv_nd(dims, cond_in_channels, model_channels, 3, padding=1),
                    conv_nd(dims, model_channels, model_channels * channel_mult[0], 3, padding=1),
                )
        self.cond_project.append(TimestepEmbedSequential(
                    ProjectionLayer(model_channels * channel_mult[0]))
                )

        self.input_blocks = nn.ModuleList([])
        self.input_blocks_cond = nn.ModuleList([])
        
        
        if self.in_scale_factor != 0:
            self.in_down = nn.Sequential(*[Downsample(model_channels * channel_mult[0], conv_resample, dims=dims) for _ in range(self.in_scale_factor)])

    
        input_block_chans = [model_channels * channel_mult[0]]
        ch = model_channels * channel_mult[0]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                layers_cond = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                    layers_cond.append(
                        AttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads
                        )
                    )
                ch = mult * model_channels
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self.input_blocks_cond.append(TimestepEmbedSequential(*layers_cond))
                self.cond_project.append(TimestepEmbedSequential(ProjectionLayer(ch)))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                self.input_blocks_cond.append(
                    TimestepEmbedSequential(Downsample(ch, conv_resample, dims=dims))
                )
                self.cond_project.append(TimestepEmbedSequential(ProjectionLayer(ch)))
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.middle_block_cond = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.f_mid_cond_proj = TimestepEmbedSequential(ProjectionLayer(ch))

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                        )
                    )
                ch = model_channels * mult
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))


        if self.in_scale_factor != 0:
            self.out_up = nn.Sequential(*[Upsample(ch, conv_resample, dims=dims) for _ in range(self.in_scale_factor)])

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            conv_nd(dims, model_channels * channel_mult[0], model_channels, 3, padding=1),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )


    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, cond=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param cond: an [N x C x ...] Tensor of condition inputs.
        :return: an [N x C x ...] Tensor of outputs.
        """ 
        
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        hs = []
        hc = []

        h = x.type(self.inner_dtype)
        h_cond = cond.type(self.inner_dtype)

        #preprocess
        h = self.preprocess(h, emb)   
        if self.in_scale_factor != 0:
            h = self.in_down(h)

        hs.append(h)
        h_cond = self.preprocess_cond(h_cond, emb)
        hc.append(h_cond)
    
        # fuse
        f = h + h_cond

        # control branch
        for module in self.input_blocks_cond:
            f = module(f, emb)
            hc.append(f)

        f = self.middle_block_cond(f, emb)
        f_mid = self.f_mid_cond_proj(f,emb)

        # control branch projection
        for i, module in enumerate(self.cond_project):
            hc[i] = module(hc[i], emb)

        
        # main branch
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)


        h = self.middle_block(h, emb)
        h = h + f_mid

        for module in self.output_blocks:
            hc_skip = hc.pop()
            hs_skip = hs.pop()
            skip_in = hs_skip + hc_skip
            cat_in = th.cat([h, skip_in], dim=1)          
            h = module(cat_in, emb)
            
        h = h.type(x.dtype)
        if self.in_scale_factor != 0:
            h = self.out_up(h)

        return self.out(h)