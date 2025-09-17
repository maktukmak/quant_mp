from functools import cache

import torch

from .template import DataFormat, register_data_format


class UniformDataFormat(DataFormat):
    signed: bool

    def __str__(self) -> str:
        if self.signed:
            return f"int{self.bit_width}"
        else:
            return f"uint{self.bit_width}"

    @property
    def max_value(self) -> float:
        if self.signed:
            return 2 ** (self.bit_width - 1) - 1
        else:
            return 2**self.bit_width - 1

    @property
    def min_value(self) -> float:
        if self.signed:
            return -(2 ** (self.bit_width - 1))
        else:
            return 0.0

    @property
    def n_values(self) -> int:
        return int(2**self.bit_width)

    def cast(self, data: torch.Tensor) -> torch.Tensor:
        return torch.clamp(torch.round(data), min=self.min_value, max=self.max_value)

    @cache
    def get_representable_values(self) -> torch.Tensor:
        return torch.tensor(list(range(int(self.min_value), int(self.max_value) + 1)))


@register_data_format
class Int2(UniformDataFormat):
    name = "int2"
    bit_width = 2
    signed = True


@register_data_format
class Int3(UniformDataFormat):
    name = "int3"
    bit_width = 3
    signed = True


@register_data_format
class Int4(UniformDataFormat):
    name = "int4"
    bit_width = 4
    signed = True


@register_data_format
class Int8(UniformDataFormat):
    name = "int8"
    bit_width = 8
    signed = True
