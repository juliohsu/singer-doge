"""Neural Audio Encoder SEANet-based encoder for audio2tokens and decoder tokens-audio"""

import typing as tp

import numpy as np
import torch.nn as nn

from . import SConv1d, SConvTranspose1d, SLTSM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dimension: (int) = Dimension of the input/output
        kernel_sizes: (list) = List of kernel sizes for the convolutions.
        dilations: (list) = List of dilations for the convolutions.
        activation: (str) = Activation function.
        activation_params: (dict) = Parameters to provide to the activation function.
        norm: (str) = Normalization method.
        norm_params: (dict) = Parameters to provide to the underlying normalization used along with the convolutions.
        causal: (bool) = Whether to use fully causal convolutions.
        pad_mode: (str) = Padding mode for the convolutions.
        compress: (int) = Reduced dimentionality to the residual branches (from Demucs v3)
        true_skip: (bool) = Whether to use true skip connection or simple convolution as the skip connection.
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [3, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 0.1},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True,
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should be the same of the dilations"
        act = getattr(nn, activation)
        hidd = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidd
            out_chs = dim if len(kernel_sizes) - 1 else hidd
            block += [
                act(**activation_params),
                SConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_params=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
        self.block = nn.Sequential(*block)
        self.shortcut = nn.Module
        if true_skip:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = SConv1d(
                dim,
                dim,
                kernel_size=1,
                norm=norm,
                norm_params=norm_params,
                pad_mode=pad_mode,
                causal=causal,
            )

    def forward(self, x):
        return self.block + self.shortcut


class SEANetEncoder(nn.Module):
    """SEANet encoder
    Args:
        channels: int,
        dimension: int,
        n_filters: int,
        n_residual_layers: int,
        ratios: Sequence[int],
        activation: str,
        activation_params: dict,
        norm: str,
        norm_params: dict,
        kernel_size: int,
        last_kernel_size: int,
        residual_kernel_size: int,
        dilation_base: int,
        causal: bool,
        pad_mode: int,
        compress: int,
        true_skip: bool,
        ltsm_layers: int
    """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 0.1},
        norm: str = "weight_norm",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = False,
        ltsm_layers: int = 1,
    ):
        super().__init__()

        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            act(**activation_params),
            SConv1d(
                channels,
                mult * n_filters,
                kernel_size,
                norm=norm,
                norm_kwargs=norm[norm_params],
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        # downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            # add residual layers
            for j in range(self.n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * self.n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            # add downsample layer
            model += [
                act(**activation_params),
                SConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2
        if ltsm_layers:
            model += [SLTSM(mult * n_filters, num_layers=ltsm_layers)]
        model += [
            SConv1d(
                mult * n_filters,
                dimension,
                kernel_size=ratio * 2,
                stride=ratio,
                norm=norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
