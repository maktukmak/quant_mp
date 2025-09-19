from math import prod
from typing import Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from torch.autograd import Function

from quant_mp.algs.minmax import MinMax
from quant_mp.config import QuantConfig, QuantModuleConfig
from quant_mp.datatypes.template import DataFormat
from quant_mp.quantizer import dequant, quant


class QuantFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        quant_config: QuantConfig,
    ):
        output, mask = quant(quant_config.qval_data_format, input, scale, shift)
        output = dequant(output, scale, shift)
        # Avoid saving None in autograd context
        if shift is None:
            ctx.has_shift = False  # type: ignore[attr-defined]
            ctx.save_for_backward(input, scale, mask)
        else:
            ctx.has_shift = True  # type: ignore[attr-defined]
            ctx.save_for_backward(input, scale, shift, mask)
        ctx.quant_config = quant_config
        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        quant_config: QuantConfig = ctx.quant_config
        assert quant_config.algorithm is not None
        saved = ctx.saved_tensors
        if getattr(ctx, "has_shift", False):
            input, scale, shift, mask = saved
        else:
            input, scale, mask = saved
            shift = None
        grad_output = grad_outputs[0]
        grad_input, grad_scale, grad_shift = quant_config.algorithm.compute_gradients(
            ctx,
            quant_config.qval_data_format,
            input,
            scale,
            shift,
            mask,
            grad_output,
        )

        # Return gradients for inputs: (input, scale, shift, quant_config)
        # No gradient for quant_config (non-Tensor)
        return grad_input, grad_scale, grad_shift, None


def init_activation_minmax(
    data_format: DataFormat,
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor | None,
):
    """One-time activation of activation quantization using minmax as proxy"""
    return MinMax().fit_params(data_format, input, scale, shift)


class QLinear(nn.Linear):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        bias=True,
        device=None,
        dtype=None,
        qlinear_config: Optional[QuantModuleConfig] = None,
    ):
        super().__init__(
            input_features, output_features, bias=bias, device=device, dtype=dtype
        )
        logger.trace(f"Initializing QLinear with quant config: {qlinear_config}")
        self.config = qlinear_config

        if qlinear_config is not None and qlinear_config.weight is not None:
            logger.trace(f"Configuring weight quantizer {qlinear_config.weight}")
            if qlinear_config.weight.algorithm is None:
                msg = "Invalid qlinear config: Must have weight quant algorithm set."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            self.weight_qconfig = qlinear_config.weight
            self.weight_alg = qlinear_config.weight.algorithm

            # Initialize params
            if qlinear_config.weight.qblock_size is None:
                block_size = output_features * input_features
            elif isinstance(qlinear_config.weight.qblock_size, int):
                block_size = qlinear_config.weight.qblock_size
                assert output_features * input_features % block_size == 0, (
                    f"Linear dimensions ({output_features} x {input_features}) is not divisible by block size of {block_size}"
                )
            elif qlinear_config.weight.qblock_size == "channel":
                block_size = output_features
            else:
                raise ValueError(
                    f"Unsupported block size {qlinear_config.weight.qblock_size}."
                )
            num_blocks = output_features * input_features // block_size
            logger.trace(
                f"Config weight block size: {qlinear_config.weight.qblock_size} => {input_features=} {output_features=} {num_blocks=} {block_size=}"
            )
            self.block_size = block_size
            self.num_blocks = num_blocks

            weight_scale = (
                torch.ones(num_blocks)
                .reshape(num_blocks, 1)
                .to(device=device, dtype=dtype)
            )
            # To trigger initialization on first iter if untrained
            weight_scale[0] = float("nan")
            if self.weight_qconfig.symmetric:
                weight_shift = None
            else:
                weight_shift = (
                    torch.zeros(num_blocks)
                    .reshape(num_blocks, 1)
                    .to(device=device, dtype=dtype)
                )

            requires_grad = not self.weight_alg.has_fit_params
            self.weight_scale = torch.nn.Parameter(
                weight_scale, requires_grad=requires_grad
            )
            if weight_shift is not None:
                self.weight_shift = torch.nn.Parameter(
                    weight_shift, requires_grad=requires_grad
                )

        if qlinear_config is not None and qlinear_config.activation is not None:
            if qlinear_config.activation.algorithm is None:
                msg = "Invalid qlinear config: Must have activation quant algorithm set if activation quantconfig exists."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            if qlinear_config.activation.qblock_size is not None:
                msg = "Invalid qlinear config: Activation qconfig must be tensor-wise."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qlinear_config}")
                raise ValueError(msg)
            self.activation_qconfig = qlinear_config.activation
            self.activation_alg = qlinear_config.activation.algorithm
            activation_scale = torch.tensor([float("nan")]).to(
                device=device, dtype=dtype
            )
            # NOTE: Activation shift values are zeroed for forced symmetric quant
            requires_grad = not self.activation_alg.has_fit_params
            self.activation_scale = torch.nn.Parameter(
                activation_scale, requires_grad=requires_grad
            )
            if not self.activation_qconfig.symmetric:
                self.activation_shift = torch.nn.Parameter(
                    torch.zeros_like(activation_scale).to(device=device, dtype=dtype),
                    requires_grad=requires_grad,
                )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        weight = self.weight.to(device)
        if self.config is not None and self.config.weight is not None:
            orig_shape = self.weight.shape
            weight = weight.view(self.num_blocks, self.block_size)
            scale = self.weight_scale.to(device)
            shift = (
                None if self.weight_qconfig.symmetric else self.weight_shift.to(device)
            )

            # NOTE: MinMax initialization on first run
            if torch.any(torch.isnan(self.weight_scale)).item():
                with torch.no_grad():
                    scale, shift = init_activation_minmax(
                        self.weight_qconfig.qval_data_format, weight, scale, shift
                    )
                # Manually update scale and shift
                _ = self.weight_scale.data.copy_(scale)
                if not self.weight_qconfig.symmetric:
                    assert shift is not None
                    _ = self.weight_shift.data.copy_(shift)

            if self.training and self.weight_alg.has_fit_params:
                with torch.no_grad():
                    scale, shift = self.weight_alg.fit_params(
                        self.weight_qconfig.qval_data_format, weight, scale, shift
                    )
                    # Manually update scale and shift
                    _ = self.weight_scale.data.copy_(scale)
                    if not self.weight_qconfig.symmetric:
                        assert shift is not None
                        _ = self.weight_shift.data.copy_(shift)

            weight: torch.Tensor = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                weight,
                scale,
                shift,
                self.weight_qconfig,
            )

            weight = weight.view(orig_shape)

        if self.config is not None and self.config.activation is not None:
            input_orig_shape = input.shape
            input = input.view(1, -1)
            scale = self.activation_scale.to(device)
            shift = (
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device)
            )

            # NOTE: MinMax initialization on first run
            if torch.any(torch.isnan(self.activation_scale)).item():
                with torch.no_grad():
                    scale, shift = init_activation_minmax(
                        self.activation_qconfig.qval_data_format, input, scale, shift
                    )
                # Manually update scale and shift
                _ = self.activation_scale.data.copy_(scale)
                if not self.activation_qconfig.symmetric:
                    assert shift is not None
                    _ = self.activation_shift.data.copy_(shift)

            if self.training and self.activation_alg.has_fit_params:
                with torch.no_grad():
                    scale, shift = self.activation_alg.fit_params(
                        self.activation_qconfig.qval_data_format, input, scale, shift
                    )
                    # Manually update scale and shift
                    _ = self.activation_scale.data.copy_(scale)
                    if not self.activation_qconfig.symmetric:
                        assert shift is not None
                        _ = self.activation_shift.data.copy_(shift)

            input = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                input,
                self.activation_scale.to(device),
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device),
                self.activation_qconfig,
            )

            input = input.view(input_orig_shape)

        out = nn.functional.linear(
            input, weight, None if self.bias is None else self.bias.to(input.device)
        )
        return out


class QConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Tuple[int, int],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        qconv_config: Optional[QuantModuleConfig] = None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        self.config = qconv_config

        if qconv_config is not None and qconv_config.weight is not None:
            if qconv_config.weight.algorithm is None:
                msg = "Invalid qconv2d config: Must have weight quant algorithm set."
                logger.error(msg)
                logger.debug(f"Quant linear config: {qconv_config}")
                raise ValueError(msg)
            self.weight_qconfig = qconv_config.weight
            self.weight_alg = qconv_config.weight.algorithm

            if qconv_config.weight.qblock_size is None:
                block_size = in_channels * out_channels * prod(self.kernel_size)
            elif isinstance(qconv_config.weight.qblock_size, int):
                block_size = qconv_config.weight.qblock_size
                assert prod(self.weight.shape) % block_size == 0, (
                    f"Conv dimensions ({self.weight.shape}) is not divisible by block size of {block_size}"
                )
            elif qconv_config.weight.qblock_size == "channel":
                block_size = out_channels
            else:
                raise ValueError(
                    f"Unsupported block size {qconv_config.weight.qblock_size}."
                )
            num_blocks = prod(self.weight.shape) // block_size
            self.block_size = block_size
            self.num_blocks = num_blocks

            # NOTE: Minmax usage here may need to change.
            weight_scale = torch.ones(num_blocks).reshape(num_blocks, 1)
            if self.weight_qconfig.symmetric:
                weight_shift = None
            else:
                weight_shift = torch.zeros(num_blocks).reshape(num_blocks, 1)

            with torch.no_grad():
                weight_scale, weight_shift = MinMax().fit_params(
                    self.weight_qconfig.qval_data_format,
                    self.weight.view(num_blocks, block_size),
                    weight_scale,
                    weight_shift,
                )
            requires_grad = not self.weight_alg.has_fit_params
            self.weight_scale = torch.nn.Parameter(
                weight_scale, requires_grad=requires_grad
            )
            if weight_shift is not None:
                self.weight_shift = torch.nn.Parameter(
                    weight_shift, requires_grad=requires_grad
                )

        if qconv_config is not None and qconv_config.activation is not None:
            if qconv_config.activation.algorithm is None:
                msg = "Invalid qconv config: Must have activation quant algorithm set if activation quantconfig exists."
                logger.error(msg)
                logger.debug(f"Quant conv config: {qconv_config}")
                raise ValueError(msg)
            if qconv_config.activation.qblock_size is not None:
                msg = "Invalid qconv config: Activation qconfig must be tensor-wise."
                logger.error(msg)
                logger.debug(f"Quant conv config: {qconv_config}")
                raise ValueError(msg)
            self.activation_qconfig = qconv_config.activation
            self.activation_alg = qconv_config.activation.algorithm
            activation_scale = torch.tensor([float("nan")])
            # NOTE: Activation shift values are zeroed for forced symmetric quant
            requires_grad = not self.activation_alg.has_fit_params
            # TODO: May be possible to enable block-wise activation quantization as well.
            self.activation_scale = torch.nn.Parameter(
                activation_scale, requires_grad=requires_grad
            )
            if not self.activation_qconfig.symmetric:
                self.activation_shift = torch.nn.Parameter(
                    torch.zeros_like(activation_scale), requires_grad=requires_grad
                )

    def forward(self, input: torch.Tensor):
        device = input.device
        weight = self.weight.to(device)
        if self.config is not None and self.config.weight is not None:
            orig_shape = self.weight.shape
            weight = weight.view(self.num_blocks, self.block_size)
            scale = self.weight_scale.to(device)
            shift = (
                None if self.weight_qconfig.symmetric else self.weight_shift.to(device)
            )
            if self.training and self.weight_alg.has_fit_params:
                with torch.no_grad():
                    scale, shift = self.weight_alg.fit_params(
                        self.weight_qconfig.qval_data_format, weight, scale, shift
                    )
                    # Manually update scale and shift
                    _ = self.weight_scale.data.copy_(scale)
                    if not self.weight_qconfig.symmetric:
                        assert shift is not None
                        _ = self.weight_shift.data.copy_(shift)

            weight: torch.Tensor = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                weight,
                scale,
                shift,
                self.weight_qconfig,
            )

            weight = weight.view(orig_shape)

        if self.config is not None and self.config.activation is not None:
            input_orig_shape = input.shape
            input = input.view(1, -1)
            scale = self.activation_scale.to(device)
            shift = (
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device)
            )

            # NOTE: MinMax activation on first run
            if torch.any(torch.isnan(self.activation_scale)).item():
                with torch.no_grad():
                    scale, shift = init_activation_minmax(
                        self.activation_qconfig.qval_data_format, input, scale, shift
                    )
                # Manually update scale and shift
                _ = self.activation_scale.data.copy_(scale)
                if not self.activation_qconfig.symmetric:
                    assert shift is not None
                    _ = self.activation_shift.data.copy_(shift)

            if self.training and self.activation_alg.has_fit_params:
                with torch.no_grad():
                    scale, shift = self.activation_alg.fit_params(
                        self.activation_qconfig.qval_data_format, input, scale, shift
                    )
                    # Manually update scale and shift
                    _ = self.activation_scale.data.copy_(scale)
                    if not self.activation_qconfig.symmetric:
                        assert shift is not None
                        _ = self.activation_shift.data.copy_(shift)

            input = QuantFunction.apply(  # pyright: ignore[reportAssignmentType]
                input,
                self.activation_scale.to(device),
                None
                if self.activation_qconfig.symmetric
                else self.activation_shift.to(device),
                self.activation_qconfig,
            )

            input = input.view(input_orig_shape)
        return self._conv_forward(input, weight, self.bias)
