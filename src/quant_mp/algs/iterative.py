from typing import TYPE_CHECKING, Optional

import torch

from quant_mp.quantizer import quant

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Iterative(Algorithm):
    name = "iterative"
    has_fit_params = True
    has_custom_gradients = True
    num_iters: int

    def __init__(self, num_iters: int = 1) -> None:
        self.num_iters = num_iters
        super().__init__()

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        for _ in range(self.num_iters):
            x_quant, _ = quant(data_format, input, scale, shift)
            if shift is None:
                scale = torch.sum((input) * x_quant, dim=1, keepdim=True) / torch.sum(
                    x_quant**2, dim=1, keepdim=True
                )
            else:
                scale = torch.sum(
                    (input - shift) * x_quant, dim=1, keepdim=True
                ) / torch.sum(x_quant**2, dim=1, keepdim=True)
                num_z = torch.sum(input - scale * x_quant, dim=1, keepdim=True)
                denum_z = len(input)
                shift = num_z / denum_z
        return scale, shift

    def compute_gradients(
        self,
        ctx,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: torch.Tensor | None,
        quant_mask: torch.Tensor,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.ste(ctx, quant_mask, grad_output)
