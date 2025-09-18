import math
from typing import Callable, Tuple

import pytest
import torch

from quant_mp.algs.template import get_algorithm
from quant_mp.config import QuantConfig
from quant_mp.datatypes.template import get_data_format as _qmp_get_df
from quant_mp.QModules import QuantFunction


class ReferenceLSQ(torch.autograd.Function):
    """Minimal LSQ-style autograd for sanity checking.

    - Uses numeric Qn/Qp and round+clamp for integer quantization.
    - Applies LSQ grad scaling: grad_scale = 1/sqrt(numel * Qp).
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, scale: torch.Tensor, num_bits: int, layerwise: bool
    ):
        ctx.num_bits = int(num_bits)
        ctx.layerwise = bool(layerwise)
        if num_bits >= 16:
            return input

        if num_bits in (1, 0):
            Qn, Qp = -1.0, 1.0
        else:
            Qn = -(2 ** (num_bits - 1))
            Qp = 2 ** (num_bits - 1) - 1

        # Ensure positive step
        eps = torch.tensor(1e-5, device=scale.device, dtype=scale.dtype)
        scale_eff = torch.where(scale > eps, scale, eps)

        # Save for backward
        grad_scale = 1.0 / math.sqrt(
            float(input.numel()) * (float(Qp) if num_bits not in (0, 1) else 1.0)
        )
        ctx.save_for_backward(input, scale_eff)
        ctx.other = (Qn, Qp, grad_scale)

        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / scale_eff).round().clamp(Qn, Qp)
        return q_w * scale_eff

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, scale = ctx.saved_tensors
        Qn, Qp, grad_scale = ctx.other
        q_w = input_ / scale
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        if ctx.num_bits == 1:
            if ctx.layerwise:
                grad_scale_param = (
                    (input_.sign() * grad_output * grad_scale).sum().unsqueeze(0)
                )
            else:
                grad_scale_param = (input_.sign() * grad_output * grad_scale).sum(
                    dim=-1, keepdim=True
                )
        else:
            base = (
                indicate_small * Qn
                + indicate_big * Qp
                + indicate_middle * (-q_w + q_w.round())
            )
            if ctx.layerwise:
                grad_scale_param = (base * grad_output * grad_scale).sum().unsqueeze(0)
            else:
                grad_scale_param = (base * grad_output * grad_scale).sum(
                    dim=-1, keepdim=True
                )

        grad_input = indicate_middle * grad_output
        return grad_input, grad_scale_param, None, None


def quantmp_lsq(
    input: torch.Tensor, scale: torch.Tensor, num_bits: int, layerwise: bool
):
    """Use the actual QuantMP LSQ via QuantFunction with an LSQ QuantConfig."""
    if num_bits >= 16:
        return input
    assert num_bits in (2, 3, 4, 8), "Only integer DataFormats supported: {2,3,4,8}"
    df = _qmp_get_df(f"int{num_bits}")
    qcfg = QuantConfig(
        qval_data_format=df,
        qparam_data_format=df,
        algorithm=get_algorithm("lsq"),
        symmetric=True,
        qblock_size="channel" if layerwise else None,
    )
    return QuantFunction.apply(input, scale, None, qcfg)


def _prepare_inputs(
    B: int, N: int, device: torch.device, layerwise: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    if layerwise:
        scale = torch.tensor([0.2], device=device, dtype=torch.float32)
    else:
        scale = torch.rand(B, 1, device=device, dtype=torch.float32) * 0.5 + 0.1
    return x, scale


def _finite_diff_scale(
    F: Callable, x: torch.Tensor, scale: torch.Tensor, num_bits: int, layerwise: bool
) -> torch.Tensor:
    # Centered finite difference of sum(F(x, alpha)) w.r.t alpha, scaled by LSQ grad_scale.
    # Use a relative step size w.r.t. alpha magnitude to reduce FD noise,
    # especially for higher bit-widths and layerwise settings.
    scale_mag = float(scale.detach().abs().mean().item())
    eps = max(1e-4, 2e-2 * scale_mag)

    def make_alpha(delta: float):
        a = scale.clone().detach()
        a = a + delta
        return a

    y_plus = F(x, make_alpha(eps), num_bits, layerwise).sum()
    y_minus = F(x, make_alpha(-eps), num_bits, layerwise).sum()
    g_est_scalar = (y_plus - y_minus) / (2 * eps)

    if num_bits >= 16:
        grad_scale = 1.0
    elif num_bits in (0, 1):
        grad_scale = 1.0 / math.sqrt(float(x.numel()) * 1.0)
    else:
        Qp = float(2 ** (num_bits - 1) - 1)
        grad_scale = 1.0 / math.sqrt(float(x.numel()) * Qp)

    g_est_scaled = g_est_scalar * grad_scale
    return torch.full_like(scale, g_est_scaled / scale.numel())


@pytest.mark.parametrize(
    "num_bits,layerwise",
    [
        (2, False),
        (4, False),
        (8, False),
        (16, False),
        (2, True),
        (4, True),
        (8, True),
        (16, True),
    ],
)
def test_lsq_reference_vs_quantmp(num_bits: int, layerwise: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 16
    rtol, atol = 1e-4, 1e-5

    # Shared inputs
    x_data, scale_data = _prepare_inputs(B, N, device, layerwise)

    # Reference forward/backward
    x = x_data.clone().detach().requires_grad_(True)
    scale = scale_data.clone().detach().requires_grad_(True)
    y_ref = ReferenceLSQ.apply(x, scale, num_bits, layerwise)
    y_ref.sum().backward()
    grad_x_ref = x.grad.detach() if x.grad is not None else None
    grad_scale_ref = None if scale.grad is None else scale.grad.detach()

    # QuantMP forward/backward via real LSQ
    x = x_data.clone().detach().requires_grad_(True)
    scale = scale_data.clone().detach().requires_grad_(True)
    y_qmp = quantmp_lsq(x, scale, num_bits, layerwise)
    y_qmp.sum().backward()
    grad_x_qmp = x.grad.detach() if x.grad is not None else None
    grad_scale_qmp = None if scale.grad is None else scale.grad.detach()

    # Forward/grad equality between implementations
    assert y_ref.shape == y_qmp.shape
    assert torch.allclose(y_ref, y_qmp, rtol=rtol, atol=atol)

    assert (
        grad_x_ref is not None
        and grad_x_qmp is not None
        and grad_x_ref.shape == grad_x_qmp.shape
    )
    assert torch.isfinite(grad_x_ref).all() and torch.isfinite(grad_x_qmp).all()
    assert torch.allclose(grad_x_ref, grad_x_qmp, rtol=rtol, atol=atol)

    if num_bits >= 16:
        # Identity path expectations
        assert torch.allclose(
            grad_x_ref, torch.ones_like(grad_x_ref), rtol=rtol, atol=atol
        )
        assert grad_scale_ref is None and grad_scale_qmp is None
        return

    # Alpha grad presence/shape/finite
    assert grad_scale_ref is not None and grad_scale_qmp is not None
    assert (
        grad_scale_ref.shape == scale_data.shape
        and grad_scale_qmp.shape == scale_data.shape
    )
    assert torch.isfinite(grad_scale_ref).all() and torch.isfinite(grad_scale_qmp).all()
    assert torch.allclose(grad_scale_ref, grad_scale_qmp, rtol=rtol, atol=atol)

    # Coarse finite-difference sanity (scaled like LSQ)
    def F_ref(inp, sc, nb, lw):
        return ReferenceLSQ.apply(inp, sc, nb, lw)

    g_est = _finite_diff_scale(F_ref, x_data, scale_data, num_bits, layerwise)
    lhs = float(grad_scale_ref.abs().mean().item())
    rhs = float(g_est.abs().mean().item())
    # Magnitude should be within a reasonable factor (very lenient due to non-smoothness).
    # Allow a slightly higher tolerance for higher bit-widths in layerwise mode, where
    # finite-difference estimates can be noisier.
    if lhs > 0 and rhs > 0:
        ratio = max(lhs, rhs) / min(lhs, rhs)
        assert ratio < 50.0
    else:
        pytest.fail(f"Zero gradient magnitude: lhs={lhs:.3e}, rhs={rhs:.3e}")
