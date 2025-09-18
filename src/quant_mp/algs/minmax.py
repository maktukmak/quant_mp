from typing import TYPE_CHECKING, Optional

import torch

from .template import Algorithm, register_algorithm

if TYPE_CHECKING:
    from quant_mp.datatypes.template import DataFormat


@register_algorithm
class MinMax(Algorithm):
    name = "minmax"
    has_fit_params = True
    has_custom_gradients = True

    def fit_params(
        self,
        data_format: "DataFormat",
        input: torch.Tensor,
        scale: torch.Tensor,
        shift: Optional[torch.Tensor] = None,  # pyright: ignore[reportDeprecated]
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # TODO: Validate if axis=1 is a valid assumption. If not, how do we generalize this properly?
        max_x = torch.max(torch.abs(input), dim=1)[0]
        if shift is None:
            new_scale = 2 * max_x / (data_format.max_value - data_format.min_value)
            return new_scale.reshape(scale.shape), None
        min_x = torch.min(input, dim=1)[0]
        # TODO: Check if this shift calculation is correct
        new_scale = (max_x - min_x) / (data_format.max_value - data_format.min_value)
        new_shift = min_x - data_format.min_value * new_scale
        return new_scale.reshape(scale.shape), new_shift.reshape(shift.shape)

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
