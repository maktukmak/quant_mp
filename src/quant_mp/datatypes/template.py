from abc import ABC, abstractmethod
from functools import cache
from typing import TypeVar

import torch
from loguru import logger


class DataFormat(ABC):
    name: str
    bit_width: int
    torch_equivalent: torch.dtype | None = None

    def __init__(self, *args, **kwargs) -> None:
        """
        Initializes the DataFormat instance.
        The name and bit_width attributes must be set in subclasses.
        """
        super().__init__(*args, **kwargs)
        # Cache of representable values moved to specific device/dtype
        # Keyed by (str(device), dtype)
        self._rv_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self._validate()

    def _validate(self) -> None:
        """
        Validation script to ensure that the DataFormat instance is correctly initialized.
        """

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"quant_mp.{self.name}"

    def __hash__(self) -> int:
        return hash(str(self))

    @property
    @abstractmethod
    def max_value(self) -> float:
        """
        Returns the maximum representable value for this data format.
        """

    @property
    @abstractmethod
    def min_value(self) -> float:
        """
        Returns the minimum representable value for this data format.
        """

    @property
    def range(self) -> float:
        """
        Returns the range of values supported
        """
        return self.max_value - self.min_value

    @property
    def abs_max_value(self) -> float:
        """
        Returns the absolute maximum representable value for this data format.
        This is the maximum of the absolute values of min_value and max_value.
        """
        return max(abs(self.min_value), abs(self.max_value))

    @property
    @abstractmethod
    def n_values(self) -> int:
        """
        Returns the number of representable values for this data format.
        This is used to determine the quantization level.
        """

    @abstractmethod
    def cast(self, data: torch.Tensor) -> torch.Tensor:
        """
        Casts the data to the nearest representable value in this format.
        """

    def get_output_mask(self, pre_cast_data: torch.Tensor) -> torch.Tensor:
        """
        Returns a mask for the output tensor that indicates which values are valid in this data format.
        This is used in STE (Straight-Through Estimator) to determine which values can be quantized without loss of information.
        """
        return (pre_cast_data <= self.max_value) * (pre_cast_data >= self.min_value)

    @abstractmethod
    @cache
    def get_representable_values(self) -> torch.Tensor:
        """
        Returns all representable values for this data format. Implementation should return sorted values and is cached for performance.
        """

    def get_values_cached(
        self, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Returns representable values materialized on the requested device/dtype, with per-device/dtype caching.
        """
        key = (str(device), dtype)
        rv = self._rv_cache.get(key)
        if rv is None:
            rv = self.get_representable_values().to(device=device, dtype=dtype)
            self._rv_cache[key] = rv
        return rv


DATA_FORMATS: dict[str, DataFormat] = {}
_T = TypeVar("_T", bound=type)


def register_data_format(cls: _T) -> _T:
    """
    Decorator to register a DataFormat class.
    """
    if not issubclass(cls, DataFormat):
        raise TypeError(f"{cls.__name__} must be a subclass of DataFormat")

    try:
        DATA_FORMATS[cls.name] = cls()
    except Exception as e:
        logger.error(f"Failed to register data format {cls.name}: {e}")
        logger.error(
            f"Resulting data format {cls.name} may be faulty. Please check above error."
        )
    return cls


@cache
def get_data_format(name: str) -> DataFormat:
    if name not in DATA_FORMATS:
        raise RuntimeError(
            f"Unrecognized data format name: {name}. Valid data formats: {DATA_FORMATS.keys()}"
        )
    return DATA_FORMATS[name]


def nearest_neighbor_cast(
    data: torch.Tensor, representable_values: torch.Tensor
) -> torch.Tensor:
    """
    Cast `data` to the nearest values from `representable_values` using L1 distance.
    Assumes `representable_values` is already on the same device/dtype as `data`.
    """
    orig_shape = data.shape
    data_flat = data.reshape(-1)
    diffs = (data_flat[:, None] - representable_values[None, :]).abs()
    idx = torch.argmin(diffs, dim=1)
    return representable_values[idx].reshape(orig_shape)
