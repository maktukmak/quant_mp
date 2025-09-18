from dataclasses import dataclass
from typing import Optional

from loguru import logger

from quant_mp.algs.template import Algorithm, get_algorithm
from quant_mp.datatypes.template import DataFormat, get_data_format


@dataclass(frozen=True)
class QuantConfig:
    qval_data_format: DataFormat  # Primary data format for quantizing values.
    qparam_data_format: DataFormat  # Not currently in use. Will implement later
    algorithm: Optional[Algorithm] = None
    symmetric: bool = True
    qblock_size: None | int | str = None

    def __post_init__(self):
        if isinstance(self.qblock_size, str) and self.qblock_size != "channel":
            raise ValueError(
                f'Invalid block size configuration: {self.qblock_size}. Must be None, int, or "channel"'
            )

    @classmethod
    def from_dict(cls, data: dict) -> "QuantConfig":  # pyright: ignore[reportMissingTypeArgument]
        qval_data_format = get_data_format(data.pop("qval_data_format"))
        qparam_data_format = get_data_format(data.pop("qparam_data_format"))
        algorithm_init_kwargs = data.pop("algorithm_init_kwargs", None)
        algorithm_name = data.pop("algorithm", QuantConfig.algorithm)
        if algorithm_name is not None:
            algorithm = get_algorithm(
                algorithm_name, algorithm_init_kwargs=algorithm_init_kwargs
            )
        else:
            algorithm = None

        return cls(
            qval_data_format,
            qparam_data_format,
            algorithm,
            **data,
        )


@dataclass
class QuantModuleConfig:
    activation: Optional[QuantConfig]
    weight: Optional[QuantConfig]

    def __post_init__(self):
        if self.activation is None and self.weight is None:
            logger.warning(
                "Quant module config intialized as full-precision. Consider not passing an empty config."
            )

    @classmethod
    def from_dict(cls, data: dict) -> "QuantModuleConfig":  # pyright: ignore[reportMissingTypeArgument]
        activation = None
        if "activation" in data:
            activation = QuantConfig.from_dict(data["activation"])
        weight = None
        if "weight" in data:
            weight = QuantConfig.from_dict(data["weight"])
        return cls(activation, weight)
