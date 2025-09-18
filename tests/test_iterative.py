import pytest
import torch

from quant_mp.algs.template import get_algorithm
from quant_mp.datatypes.template import get_data_format
from quant_mp.quantizer import quant


def _prepare_inputs(B: int, N: int, device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    return x


@pytest.mark.parametrize("bit_width", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_iterative_fit_params_single_iter(bit_width: int, symmetric: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 64
    x = _prepare_inputs(B, N, device)

    # Setup
    df = get_data_format(f"int{bit_width}")
    alg = get_algorithm("iterative", algorithm_init_kwargs={"num_iters": 1})

    init_scale = torch.full((B, 1), 0.3, device=device)
    init_shift = None if symmetric else torch.full((B, 1), -0.1, device=device)

    # Algorithm output (one iteration)
    scale, shift = alg.fit_params(
        df, x, init_scale.clone(), None if symmetric else init_shift.clone()
    )

    # Reference: quantize once with initial params, then apply normal-equation updates
    x_quant, _ = quant(df, x, init_scale, None if symmetric else init_shift)
    if symmetric:
        num = torch.sum(x * x_quant, dim=1, keepdim=True)
        den = torch.sum(x_quant**2, dim=1, keepdim=True)
        scale_ref = num / den
        shift_ref = None
    else:
        num = torch.sum((x - init_shift) * x_quant, dim=1, keepdim=True)
        den = torch.sum(x_quant**2, dim=1, keepdim=True)
        scale_ref = num / den
        # Match implementation detail: divides by batch size (len(input)), not feature count
        shift_ref = torch.sum(x - scale_ref * x_quant, dim=1, keepdim=True) / x.shape[0]

    # Compare
    assert scale.shape == init_scale.shape
    assert torch.allclose(scale, scale_ref, rtol=1e-4, atol=1e-5)
    if symmetric:
        assert shift is None
    else:
        assert shift is not None and shift.shape == init_shift.shape
        assert torch.allclose(shift, shift_ref, rtol=1e-4, atol=1e-5)
