from .float import (
    FloatDataFormat,
    Fp4_e2m1,
    Fp4_e3m0,
    Fp8_e4m3,
    Fp8_e4m3fnuz,
    Fp8_e5m2,
    Fp32,
)
from .int import Int2, Int3, Int4, Int8, UniformDataFormat
from .nonuniform import NF4, SF4

int2 = Int2()
int3 = Int3()
int4 = Int4()
int8 = Int8()

fp4_e3m0 = Fp4_e3m0()
fp4_e2m1 = Fp4_e2m1()
fp8_e5m2 = Fp8_e5m2()
fp8_e4m3 = Fp8_e4m3()
fp8_e4m3fnuz = Fp8_e4m3fnuz()
fp32 = Fp32()

nf4 = NF4()
sf4 = SF4()

__all__ = [
    "UniformDataFormat",
    "FloatDataFormat",
    "int2",
    "int3",
    "int4",
    "int8",
    "fp4_e3m0",
    "fp4_e2m1",
    "fp8_e5m2",
    "fp8_e4m3",
    "fp8_e4m3fnuz",
    "fp32",
    "nf4",
    "sf4",
]
