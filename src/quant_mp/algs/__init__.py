from .analytic import Analytic
from .iterative import Iterative
from .lsq import LSQ
from .minmax import MinMax
from .octav import Octav
from .template import ALGORITHMS, get_algorithm

__all__ = [
    "ALGORITHMS",
    "Analytic",
    "get_algorithm",
    "Iterative",
    "LSQ",
    "MinMax",
    "Octav",
]
