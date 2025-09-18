import math
from typing import Callable, Tuple

import pytest
import torch

from quant_mp.datatypes.template import get_data_format as _qmp_get_df
from quant_mp.config import QuantConfig
from quant_mp.algs.template import get_algorithm
from quant_mp.QModules import QuantFunction


class ReferenceLSQ(torch.autograd.Function):
    """Minimal LSQ-style autograd for sanity checking.

    - Uses numeric Qn/Qp and round+clamp for integer quantization.
    - Applies LSQ grad scaling: grad_scale = 1/sqrt(numel * Qp).
    """

    @staticmethod
    def forward(
        ctx, input: torch.Tensor, alpha: torch.Tensor, num_bits: int, layerwise: bool
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
        eps = torch.tensor(1e-5, device=alpha.device, dtype=alpha.dtype)
        alpha_eff = torch.where(alpha > eps, alpha, eps)

        # Save for backward
        grad_scale = 1.0 / math.sqrt(
            float(input.numel()) * (float(Qp) if num_bits not in (0, 1) else 1.0)
        )
        ctx.save_for_backward(input, alpha_eff)
        ctx.other = (Qn, Qp, grad_scale)

        if num_bits == 1:
            q_w = input.sign()
        else:
            q_w = (input / alpha_eff).round().clamp(Qn, Qp)
        return q_w * alpha_eff

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        if ctx.num_bits >= 16:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        Qn, Qp, grad_scale = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = 1.0 - indicate_small - indicate_big

        if ctx.num_bits == 1:
            if ctx.layerwise:
                grad_alpha = (
                    (input_.sign() * grad_output * grad_scale).sum().unsqueeze(0)
                )
            else:
                grad_alpha = (input_.sign() * grad_output * grad_scale).sum(
                    dim=-1, keepdim=True
                )
        else:
            base = (
                indicate_small * Qn
                + indicate_big * Qp
                + indicate_middle * (-q_w + q_w.round())
            )
            if ctx.layerwise:
                grad_alpha = (base * grad_output * grad_scale).sum().unsqueeze(0)
            else:
                grad_alpha = (base * grad_output * grad_scale).sum(dim=-1, keepdim=True)

        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


def quantmp_lsq(
    input: torch.Tensor, alpha: torch.Tensor, num_bits: int, layerwise: bool
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
    return QuantFunction.apply(input, alpha, None, qcfg)


def _prepare_inputs(
    B: int, N: int, device: torch.device, layerwise: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    if layerwise:
        alpha = torch.tensor([0.2], device=device, dtype=torch.float32)
    else:
        alpha = torch.rand(B, 1, device=device, dtype=torch.float32) * 0.5 + 0.1
    return x, alpha


def _finite_diff_alpha(
    F: Callable, x: torch.Tensor, alpha: torch.Tensor, num_bits: int, layerwise: bool
) -> torch.Tensor:
    # Centered finite difference of sum(F(x, alpha)) w.r.t alpha, scaled by LSQ grad_scale.
    # Use a relative step size w.r.t. alpha magnitude to reduce FD noise,
    # especially for higher bit-widths and layerwise settings.
    alpha_mag = float(alpha.detach().abs().mean().item())
    eps = max(1e-4, 2e-2 * alpha_mag)

    def make_alpha(delta: float):
        a = alpha.clone().detach()
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
    return torch.full_like(alpha, g_est_scaled / alpha.numel())


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
    x_data, alpha_data = _prepare_inputs(B, N, device, layerwise)

    def run_case(Ext):
        x = x_data.clone().detach().requires_grad_(True)
        alpha = alpha_data.clone().detach().requires_grad_(True)
        # Support both autograd.Function subclasses and plain callables
        if hasattr(Ext, "apply"):
            y = Ext.apply(x, alpha, num_bits, layerwise)
        else:
            y = Ext(x, alpha, num_bits, layerwise)
        loss = y.sum()
        loss.backward()
        return (
            y.detach(),
            x.grad.detach() if x.grad is not None else None,
            alpha.grad.detach() if alpha.grad is not None else None,
        )

    yA, gxA, gaA = run_case(ReferenceLSQ)
    yB, gxB, gaB = run_case(quantmp_lsq)

    # Forward/grad equality between implementations
    assert yA.shape == yB.shape
    assert torch.allclose(yA, yB, rtol=rtol, atol=atol)

    assert gxA is not None and gxB is not None and gxA.shape == gxB.shape
    assert torch.isfinite(gxA).all() and torch.isfinite(gxB).all()
    assert torch.allclose(gxA, gxB, rtol=rtol, atol=atol)

    if num_bits >= 16:
        # Identity path expectations
        assert torch.allclose(gxA, torch.ones_like(gxA), rtol=rtol, atol=atol)
        assert gaA is None and gaB is None
        return

    # Alpha grad presence/shape/finite
    assert gaA is not None and gaB is not None
    assert gaA.shape == alpha_data.shape and gaB.shape == alpha_data.shape
    assert torch.isfinite(gaA).all() and torch.isfinite(gaB).all()
    assert torch.allclose(gaA, gaB, rtol=rtol, atol=atol)

    # Coarse finite-difference sanity (scaled like LSQ)
    def F_ref(inp, a, nb, lw):
        return ReferenceLSQ.apply(inp, a, nb, lw)

    g_est = _finite_diff_alpha(F_ref, x_data, alpha_data, num_bits, layerwise)
    lhs = float(gaA.abs().mean().item())
    rhs = float(g_est.abs().mean().item())
    # Magnitude should be within a reasonable factor (very lenient due to non-smoothness).
    # Allow a slightly higher tolerance for higher bit-widths in layerwise mode, where
    # finite-difference estimates can be noisier.
    if lhs > 0 and rhs > 0:
        ratio = max(lhs, rhs) / min(lhs, rhs)
        assert ratio < 50.0
    else:
        pytest.fail(f"Zero gradient magnitude: lhs={lhs:.3e}, rhs={rhs:.3e}")
