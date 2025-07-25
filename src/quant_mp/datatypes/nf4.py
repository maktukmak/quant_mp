
from functools import cache

import torch

from .template import DataFormat, register_data_format

class NF4DataFormat(DataFormat):
    signed: bool = True
    bit_width = 4

    def __str__(self) -> str:
        return f"NF4"

    @property
    def max_value(self) -> float:
        return 1.

    @property
    def min_value(self) -> float:
            return -1.0

    @property
    def n_values(self) -> int:
        return int(2**self.bit_width)

    def cast(self, data: torch.Tensor) -> torch.Tensor:

        orig_shape = data.shape
        data = data.flatten().unsqueeze(1) 
        data  = self.get_representable_values()[torch.argmin(torch.abs(data
         - self.get_representable_values()), dim=-1)]

        return data.reshape(orig_shape)

    @cache
    def get_representable_values(self) -> torch.Tensor:
        return torch.tensor([
            -1.0000000,
            -0.6961928,
            -0.52507305,
            -0.39491749,
            -0.28444138,
            -0.18477343,
            -0.09105004,
            0.0000000,
            0.07958030,
            0.16093020,
            0.24611230,
            0.33791524,
            0.44070983,
            0.56261700,
            0.72295684,
            1.0000000
        ])


@register_data_format
class NF4(NF4DataFormat):
    name = "nf4"
