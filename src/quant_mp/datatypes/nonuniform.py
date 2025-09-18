import torch

from .template import DataFormat, nearest_neighbor_cast, register_data_format


class NonUniformDataFormat(DataFormat):
    signed: bool = True
    bit_width = 4

    def __str__(self) -> str:
        return self.name

    @property
    def max_value(self) -> float:
        return torch.max(self.get_representable_values()).item()

    @property
    def min_value(self) -> float:
        return torch.min(self.get_representable_values()).item()

    @property
    def n_values(self) -> int:
        return int(len(self.get_representable_values()))

    def cast(self, data: torch.Tensor) -> torch.Tensor:
        device = data.device
        dtype = data.dtype
        rv = self.get_values_cached(device, dtype)
        return nearest_neighbor_cast(data, rv)


@register_data_format
class NF4(NonUniformDataFormat):
    name = "nf4"

    def get_representable_values(self) -> torch.Tensor:
        return torch.tensor(
            [
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
                1.0000000,
            ]
        )


@register_data_format
class SF4(NonUniformDataFormat):
    name = "sf4"

    def get_representable_values(self) -> torch.Tensor:
        return torch.tensor(
            [
                -1.000,
                -0.628,
                -0.455,
                -0.334,
                -0.237,
                -0.153,
                -0.075,
                0.000,
                0.066,
                0.133,
                0.205,
                0.284,
                0.376,
                0.491,
                0.657,
                1.000,
            ]
        )
