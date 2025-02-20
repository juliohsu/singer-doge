"""Convolutional layers wrappers and utilities."""

import math
import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_norm import ConvLayerNorm
from torch.nn.utils import weight_norm, spectral_norm

CONV_NORMALIZATIONS = frozenset(
    [
        "none",
        "weight_norm",
        "spectral_norm",
        "layer_norm",
        "time_layer_norm",
        "time_group_norm",
    ]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none") -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We have already check if "norm" is in CONV_NORMALIZATIONS,
        # so any other choice doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool, norm: str = "none", **norm_kwargs
) -> nn.Module:
    """Return the proper normalization module. If the causal is true, ensure that the returned module is causal,
    or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "layer_norm":
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames - 1) * stride) + (kernel_size - padding_total)
    return ideal_length - length


def pad_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
):
    """Add extra padding for the 1d convolution."""
    extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
    return F.pad(x, (0, extra_padding))


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "zero",
    value: float = 1.0,
):
    """Tiny wrapper around F.pad"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if max_pad >= length:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)
