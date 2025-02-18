"""Neural Audio Encoder SEANet-based encoder for audio2tokens and decoder tokens-audio"""

import typing as tp

import numpy as np
import torch.nn as nn

from . import SConv1d, SConvTranspose1d, SLTSM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model.
    Args:
        dimension: (int) = Dimension of the input/output
        kernel_size: (list) = List of kernel sizes for the convolutions.
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
    