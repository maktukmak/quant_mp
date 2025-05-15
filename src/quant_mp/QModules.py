from typing import Optional, Tuple, Type
import torch
from torch.autograd import Function
import torch.nn as nn
from quant_mp.config import QuantConfig, QuantLinearConfig
from quant_mp.quantizer import (
    FloatQuantizer,
    UniformQuantizer,
    get_quantizer,
    QuantizerBase,
)
from torch.nn.functional import conv2d
from math import prod


# FIXME: Not updated to new API yet
def step_quantizer_delayed(tensor, quantizer: Optional[QuantizerBase]):
    if quantizer:
        # Delayed scaling does not work when block-wise
        delayed_scaling = False
        if delayed_scaling:
            params_prev = quantizer.params
            tensor, params, mask = quantizer.fit_and_quant(
                tensor, params_prev
            )  # Currently fake
            params_new = params_prev if params_prev else params
            # quantizer.s = quantizer.qconfig['beta'] * scale_prev + (1-quantizer.qconfig['beta']) * s if scale_prev else s

        else:
            tensor, params_new, mask = quantizer.fit_and_quant(tensor, None)

        return tensor, params_new[0], mask

    return tensor, torch.tensor(1.0), torch.ones_like(tensor)


class QuantFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor,
        quantizer: QuantizerBase,
        is_training: bool = False,
    ):
        if is_training:
            scale, shift = quantizer.fit(input, scale, shift)
        output, mask = quantizer.quant(input, scale, shift)
        output = quantizer.dequant(output, scale, shift)
        ctx.save_for_backward(mask)
        return output, scale, shift

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, grad_scale, grad_shift = grad_outputs
        mask = ctx.saved_tensors[0]
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * mask

        return grad_input, grad_scale, grad_shift, None, None


def get_quantize_function_cls(qconfig: QuantConfig) -> Type[Function]:
    if qconfig.alg in ["minmax", "iterative", "normal"]:
        return QuantFunction
    raise ValueError(f"No quantization function found for {qconfig}")


def init_activation_minmax(
    input: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    quantizer: QuantizerBase,
):
    """One-time activation of activation quantization using minmax as proxy"""
    assert isinstance(quantizer, (FloatQuantizer, UniformQuantizer))
    return quantizer.fit_minmax(input, scale, shift)


class QLinear(nn.Linear):
    def __init__(
        self,
        input_features: int,
        output_features: int,
        rconfig: QuantLinearConfig,
        bias=True,
        device=torch.device("cuda"),
    ):
        super().__init__(input_features, output_features, bias=bias)
        self.input_features = input_features
        self.output_features = output_features

        self.rconfig = rconfig

        # Initialize quantizer and param for quant weights
        self.quantizer_weight = None
        if rconfig.weight.is_quantized:
            if rconfig.weight.qblock_size is None:
                block_size = output_features * input_features
            elif isinstance(rconfig.weight.qblock_size, int):
                block_size = rconfig.weight.qblock_size
            elif rconfig.weight.qblock_size == "channel":
                block_size = output_features
            else:
                raise ValueError(
                    f"Unsupported block size {rconfig.weight.qblock_size}."
                )
            num_blocks = output_features * input_features // block_size
            self.block_size = block_size
            self.num_blocks = num_blocks
            self.quantizer_weight = get_quantizer(
                qconfig=self.rconfig.weight, device=device
            )
            self.quantize_weight_cls = get_quantize_function_cls(self.rconfig.weight)
            # Initialize weight and shift val using minmax
            assert isinstance(self.quantizer_weight, (FloatQuantizer, UniformQuantizer))
            with torch.no_grad():
                weight_clip_val, weight_shift_val = self.quantizer_weight.fit_minmax(
                    self.weight.view(num_blocks, block_size),
                    torch.ones(num_blocks),
                    torch.zeros(num_blocks),
                )
            if rconfig.weight.alg_requires_grad_params:
                self.weight_clip_val = torch.nn.Parameter(weight_clip_val)
                self.weight_shift_val = torch.nn.Parameter(weight_shift_val)
            else:
                self.register_buffer("weight_clip_val", weight_clip_val)
                self.register_buffer("weight_shift_val", weight_shift_val)

        self.quantizer_act = get_quantizer(
            qconfig=self.rconfig.activation, device=device
        )
        if rconfig.activation.is_quantized:
            self.quantize_act_cls = get_quantize_function_cls(self.rconfig.activation)
            activation_clip_val = torch.tensor([float("nan")])
            # NOTE: Activation shift values are zeroed for forced symmetric quant
            activation_shift_val = torch.zeros_like(activation_clip_val, device=device)
            if rconfig.activation.alg_requires_grad_params:
                self.activation_clip_val = torch.nn.Parameter(activation_clip_val)
                self.activation_shift_val = torch.nn.Parameter(activation_shift_val)
            else:
                self.register_buffer("activation_clip_val", activation_clip_val)
                self.register_buffer("activation_shift_val", activation_shift_val)

        if (
            self.quantizer_act is not None
            and self.rconfig.activation.qblock_size is not None
        ):
            print(
                f"Block size ({self.rconfig.activation.qblock_size}) is not supported for activations. Tensor-wise quantization will be applied"
            )

        self.qgrad = get_quantizer(qconfig=self.rconfig.grad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        weight = self.weight.to(device)
        if self.quantizer_weight is not None:
            orig_shape = self.weight.shape
            weight = weight.view(self.num_blocks, self.block_size)
            weight, scale, shift = self.quantize_weight_cls.apply(
                weight,
                self.weight_clip_val.to(device),
                self.weight_shift_val.to(device),
                self.quantizer_weight,
                self.training,
            )  # type: ignore

            # Only manually update if not requiring grad
            if not self.rconfig.weight.alg_requires_grad_params:
                self.weight_clip_val.data.copy_(scale)
                self.weight_shift_val.data.copy_(shift)
            weight = weight.view(orig_shape)

        if self.quantizer_act is not None:
            input_orig_shape = input.shape
            input = input.view(1, -1)
            if torch.any(torch.isnan(self.activation_clip_val)).item():
                with torch.no_grad():
                    scale, shift = init_activation_minmax(
                        input,
                        self.activation_clip_val.to(device),
                        self.activation_shift_val.to(device),
                        self.quantizer_act,
                    )
                    self.activation_clip_val.data.copy_(scale)
                    self.activation_shift_val.data.copy_(shift)

            input, scale, shift = self.quantize_act_cls.apply(
                input,
                self.activation_clip_val.to(device),
                self.activation_shift_val.to(device),
                self.quantizer_act,
                self.training,
            )  # type: ignore

            # Only manually update if not requiring grad
            if not self.rconfig.activation.alg_requires_grad_params:
                self.activation_clip_val.data = scale.data
                self.activation_shift_val.data = shift.data
            input = input.view(input_orig_shape)

        out = nn.functional.linear(input, weight)
        if self.bias is not None:
            out += self.bias.unsqueeze(0).expand_as(out).to(input.device)
        return out


# FIXME: Conv2D function not converted to new API
class QConv2dFunction(Function):
    @staticmethod
    def forward(
        ctx,
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        qweight=None,
        qact=None,
        qgrad=None,
    ):
        scale_bw = torch.tensor([1.0, 1.0])
        weight, scale_bw[0], wmask = step_quantizer_delayed(weight, qweight)
        input, scale_bw[1], amask = step_quantizer_delayed(input, qact)
        if qweight and not qact:
            weight = qweight.dequant(weight, (scale_bw[0], 0.0))
            scale_bw[0] = 1.0

        if not qweight and qact:
            input = qact.dequant(input, (scale_bw[1], 0.0))
            scale_bw[1] = 1.0

        output = (
            scale_bw[1]
            * conv2d(input, weight, None, stride, padding, dilation, groups)
            * scale_bw[0].T
        )
        if bias is not None:
            output += bias[None, :, None, None]

        ctx.save_for_backward(input, weight, bias, wmask, amask)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        ctx.qgrad = qgrad
        ctx.scale_bw = scale_bw

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # import pydevd
        # pydevd.settrace(suspend=False, trace_only_current_thread=True)

        input, weight, bias, wmask, amask = ctx.saved_tensors

        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups

        qgrad = ctx.qgrad
        scales = ctx.scale_bw
        grad_input = grad_weight = grad_bias = None

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3)).squeeze(0)

        grad_output, scale, _ = step_quantizer_delayed(grad_output, qgrad)

        if ctx.needs_input_grad[0]:
            grad_input = (
                torch.nn.grad.conv2d_input(
                    input.shape, weight, grad_output, stride, padding, dilation, groups
                )
                * scales[0]
                * scale
                * amask
            )

        if ctx.needs_input_grad[1]:
            grad_weight = (
                torch.nn.grad.conv2d_weight(
                    input, weight.shape, grad_output, stride, padding, dilation, groups
                )
                * scales[1]
                * scale
                * wmask
            )

        return (
            grad_input,
            grad_weight,
            grad_bias,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


# FIXME: Conv2d not converted to new API yet
class QConv2d(nn.Module):
    def __init__(
        self,
        rconfig: QuantLinearConfig,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int],
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_size = kernel_size

        self.rconfig = rconfig

        self.qweight = get_quantizer(qconfig=rconfig.weight)
        self.qact = get_quantizer(qconfig=rconfig.activation)
        self.qgrad = get_quantizer(qconfig=rconfig.grad)

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size[0], kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        k = torch.sqrt(groups / (in_channels * torch.tensor(prod(kernel_size))))
        nn.init.uniform_(self.weight, -k, k)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -k, k)
            # nn.init.zeros_(self.bias)

    def forward(self, input):
        return QConv2dFunction.apply(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.qweight,
            self.qact,
            self.qgrad,
        )
