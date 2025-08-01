from typing import TYPE_CHECKING, Optional

import torch

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class Octav(Algorithm):
    name = "octav"
    has_fit_params = True
    has_custom_gradients = True
    num_iters: int

    def __init__(self, num_iters: int = 10) -> None:
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
            outside_mask = torch.abs(input) > scale
            inside_mask = ~outside_mask
            if shift is None:
                scale = torch.sum(
                    torch.abs(input) * outside_mask, dim=1, keepdim=True
                ) / (
                    4**-data_format.bit_width
                    / 3
                    * torch.sum(inside_mask.float(), dim=1, keepdim=True)
                    + torch.sum(outside_mask.float(), dim=1, keepdim=True)
                )
            else:
                raise NotImplementedError("Non-symmetric is not implemented for octav")
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
        grad_input, _, _ = self.ste(ctx, quant_mask, grad_output)
        if grad_input is not None:
            outside_mask = ~quant_mask
            grad_input += (
                scale / torch.abs(input + 1e-8) * outside_mask.float() * grad_output
            )
        return grad_input, None, None
