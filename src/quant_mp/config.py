from dataclasses import dataclass
from typing import Optional


@dataclass
class QuantConfig:
    qtype: Optional[str] = None
    qbits: int = 4
    qblock_size: Optional[int | str] = None
    alg: str = "minmax"
    beta: float = 0.0
    format: Optional[str] = "e2m1"
    num_iters: int = 1  # Number of iterations if using iterative fit
    symmetric: bool = True

    @property
    def is_quantized(self) -> bool:
        return self.qtype is not None

    @property
    def alg_requires_grad_params(self) -> bool:
        return False


@dataclass
class QuantLinearConfig:
    label: str
    activation: QuantConfig
    weight: QuantConfig
    grad: QuantConfig
