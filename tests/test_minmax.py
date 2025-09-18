import pytest
import torch

from quant_mp.algs.template import get_algorithm
from quant_mp.config import QuantConfig
from quant_mp.datatypes.template import get_data_format as _qmp_get_df
from quant_mp.QModules import QuantFunction


class ReferenceMinMax(torch.autograd.Function):
    """Reference STE for MinMax algorithm matching QuantFunction + DataFormat semantics."""

    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        num_bits: int,
    ):
        # Identity for high precision not supported here (we test 2/4/8 only)
        df = _qmp_get_df(f"int{num_bits}")

        out = input.clone()
        if shift is not None:
            out = out - shift
        out = out / scale
        cast = df.cast(out)
        mask = df.get_output_mask(out)
        y = cast * scale
        if shift is not None:
            y = y + shift

        ctx.save_for_backward(mask)
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (mask,) = ctx.saved_tensors
        grad_input = mask * grad_output
        return grad_input, None, None, None


def quantmp_minmax(
    input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor | None, num_bits: int
):
    df = _qmp_get_df(f"int{num_bits}")
    qcfg = QuantConfig(
        qval_data_format=df,
        qparam_data_format=df,
        algorithm=get_algorithm("minmax"),
        symmetric=(shift is None),
    )
    return QuantFunction.apply(input, scale, shift, qcfg)


def _prepare_inputs(B: int, N: int, device: torch.device, layerwise: bool):
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    if layerwise:
        scale = torch.tensor([0.2], device=device, dtype=torch.float32)
    else:
        scale = torch.rand(B, 1, device=device, dtype=torch.float32) * 0.5 + 0.1
    shift = None  # symmetric for this test
    return x, scale, shift


@pytest.mark.parametrize(
    "num_bits,layerwise",
    [
        (2, False),
        (4, False),
        (8, False),
        (2, True),
        (4, True),
        (8, True),
    ],
)
def test_minmax_reference_vs_quantmp(num_bits: int, layerwise: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 16
    rtol, atol = 1e-4, 1e-5

    x_data, scale_data, shift_data = _prepare_inputs(B, N, device, layerwise)

    # Reference forward/backward
    x = x_data.clone().detach().requires_grad_(True)
    scale = scale_data.clone().detach().requires_grad_(True)
    shift = (
        None if shift_data is None else shift_data.clone().detach().requires_grad_(True)
    )
    y_ref = ReferenceMinMax.apply(x, scale, shift, num_bits)
    y_ref.sum().backward()
    grad_x_ref = x.grad.detach()
    grad_scale_ref = scale.grad
    grad_shift_ref = None if shift is None else shift.grad

    # QuantMP forward/backward
    x = x_data.clone().detach().requires_grad_(True)
    scale = scale_data.clone().detach().requires_grad_(True)
    shift = (
        None if shift_data is None else shift_data.clone().detach().requires_grad_(True)
    )
    y_qmp = quantmp_minmax(x, scale, shift, num_bits)
    y_qmp.sum().backward()
    grad_x_qmp = x.grad.detach()
    grad_scale_qmp = scale.grad
    grad_shift_qmp = None if shift is None else shift.grad

    y_ref, y_qmp = y_ref.detach(), y_qmp.detach()

    # Forward equality
    assert y_ref.shape == y_qmp.shape
    assert torch.allclose(y_ref, y_qmp, rtol=rtol, atol=atol)

    # Grad input equality and finiteness
    assert (
        grad_x_ref is not None
        and grad_x_qmp is not None
        and grad_x_ref.shape == grad_x_qmp.shape
    )
    assert torch.isfinite(grad_x_ref).all() and torch.isfinite(grad_x_qmp).all()
    assert torch.allclose(grad_x_ref, grad_x_qmp, rtol=rtol, atol=atol)

    # MinMax uses STE only: no parameter gradients expected
    assert grad_scale_ref is None and grad_scale_qmp is None
    assert grad_shift_ref is None and grad_shift_qmp is None
