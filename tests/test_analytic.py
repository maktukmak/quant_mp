import pytest
import torch

from quant_mp.algs.analytic import get_copt_uniform
from quant_mp.algs.template import get_algorithm
from quant_mp.datatypes.template import get_data_format


def _prepare_inputs(B: int, N: int, device: torch.device):
    torch.manual_seed(0)
    x = torch.randn(B, N, device=device, dtype=torch.float32)
    return x


@pytest.mark.parametrize("bit_width", [2, 4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
def test_analytic_fit_params_uniform(bit_width: int, symmetric: bool):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, N = 4, 64
    x = _prepare_inputs(B, N, device)

    # Setup
    df = get_data_format(f"int{bit_width}")
    alg = get_algorithm("analytic")
    init_scale = torch.ones(B, 1, device=device)
    init_shift = None if symmetric else torch.zeros(B, 1, device=device)

    # Algorithm output
    scale, shift = alg.fit_params(df, x, init_scale, init_shift)

    # Reference calculation
    x_std = torch.std(x, dim=1, keepdim=True)
    copt = get_copt_uniform(df)
    scale_ref = (2.0 * copt * x_std) / (df.n_values - 1)
    shift_ref = None if symmetric else x.mean(dim=1, keepdim=True)

    # Compare
    assert scale.shape == init_scale.shape
    assert torch.allclose(scale, scale_ref.reshape_as(scale), rtol=1e-4, atol=1e-5)
    if symmetric:
        assert shift is None
    else:
        assert shift is not None and shift.shape == init_shift.shape
        assert torch.allclose(shift, shift_ref.reshape_as(shift), rtol=1e-4, atol=1e-5)
