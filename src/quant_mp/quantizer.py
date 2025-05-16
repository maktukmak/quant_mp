from typing import Callable, Dict, Tuple
import torch

from quant_mp.config import QuantConfig
import numpy as np
from sklearn.cluster import KMeans
from scipy import special
from abc import ABC, abstractmethod

torch.set_printoptions(precision=5)


class QuantizerBase(ABC):
    fit_dispatcher: Dict[
        str,
        Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, torch.Tensor],
        ],
    ]

    def __init__(self, qconfig: QuantConfig, device: torch.device):
        self.qconfig = qconfig
        self.num_bits = qconfig.qbits
        self.alg = qconfig.alg
        self.device = device

        self.n_levels = int(2**self.num_bits)

    def error(self, x, xdeq):
        err = torch.sum(((x - xdeq) ** 2)) / len(x)
        return err

    def q_function(self, x):
        return 0.5 - 0.5 * special.erf(x / np.sqrt(2))

    def gauss_cdf(self, x, m, std):
        return 0.5 * (1 + torch.erf((x - m) / (torch.sqrt(torch.tensor(2.0)) * std)))

    @abstractmethod
    def compute_quant_levels(
        self, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        pass

    @abstractmethod
    def quant(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantizes input with scale and shift. Returns quantized input and clip mask"""
        pass

    @abstractmethod
    def dequant(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        """Dequantizes input with scale and shift. Returns dequantized input."""
        pass

    def fit(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fits with the corresponding algorithm. Returns an updated scale and shift based on input.

        Expected Shapes:
            input: (*, n_blocks)
            scale: (n_blocks)
            shift: (n_blocks)
        """
        return self.fit_dispatcher[self.alg](input, scale, shift)

    def fit_and_quant(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fits and quantizes and"""
        scale, shift = self.fit(input, scale, shift)
        input, mask = self.quant(input, scale, shift)
        return input, scale, shift, mask


# TODO: Validate each configuration
class UniformQuantizer(QuantizerBase):
    def __init__(self, qconfig: QuantConfig, device: torch.device):
        super().__init__(qconfig, device)

        self.fit_dispatcher = {
            "minmax": self.fit_minmax,
            "normal": self.fit_normal,
            "iterative": self.fit_iterative,
        }

        self.k_list = torch.arange(0, self.n_levels - 1).to(torch.int)

        self.G = self.k_list
        if self.qconfig.symmetric:
            self.G = self.k_list - self.n_levels / 2 + 1

        C = np.linspace(1, 100, 10000)
        gres = self.snr(C, 1, self.n_levels)
        self.Copt = C[np.argmax(gres)]

    def compute_quant_levels(self, scale, shift):
        lk = scale * self.k_list + shift
        return lk

    def quant(self, input, scale, shift):
        num_blocks = input.shape[0]
        input = (input - shift.view(num_blocks, -1)) / scale.view(num_blocks, -1)
        mask = (input <= self.n_levels / 2 - 1) * (input >= -self.n_levels / 2 + 1)
        return torch.clamp(
            torch.round(input), -self.n_levels // 2 + 1, self.n_levels // 2 - 1
        ), mask

    def dequant(self, input, scale, shift):
        num_blocks = input.shape[0]
        return scale.view(num_blocks, -1) * input + shift.view(num_blocks, -1)

    def snr(self, C, sigma2, N):
        C = torch.tensor(C)

        z = C**2 / sigma2
        return 1 / (
            2 * (1 + z) * self.q_function(torch.sqrt(z))
            - torch.sqrt(2 * z / np.pi) * torch.exp(-0.5 * z)
            + z / (3 * ((N - 1) ** 2))
        )

    def fit_minmax(
        self, input: torch.Tensor, _scale: torch.Tensor, _shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        if self.qconfig.symmetric:
            max_x = torch.max(torch.abs(input), axis=1)[0]
            scale = 2 * max_x / (self.n_levels - 1)
            shift = torch.zeros_like(scale, device=device)
        else:
            min_x = torch.min(input, axis=1)[0]
            max_x = torch.max(input, axis=1)[0]

            scale = (max_x - min_x) / (self.n_levels - 1)
            shift = min_x + scale / 2
        return scale, shift

    def fit_normal(
        self, input: torch.Tensor, _scale: torch.Tensor, _shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        x_std = torch.std(input, axis=1)
        scale = (2 * self.Copt * x_std) / (self.n_levels - 1)
        if self.qconfig.symmetric:
            shift = torch.zeros_like(scale, device=device)
        else:
            x_mean = torch.mean(input, axis=1)
            shift = x_mean - (self.n_levels / 2 - 1) * scale
        return scale, shift

    def fit_iterative(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = input.shape[0]
        for _ in range(self.qconfig.num_iters):
            xint, _ = self.quant(input, scale, shift)
            num_s = torch.sum((input - shift.view(num_blocks, -1)) * xint, axis=1)
            denum_s = torch.sum(xint**2, axis=1)
            scale = num_s / denum_s
            if not self.qconfig.symmetric:
                num_z = torch.sum(input - scale.view(num_blocks, -1) * xint, axis=1)
                denum_z = len(input)
                shift = num_z / denum_z
        return scale, shift


# TODO: Validate each configuration
class FloatQuantizer(QuantizerBase):
    def __init__(self, qconfig: QuantConfig, device: torch.device):
        super().__init__(qconfig, device)

        self.fit_dispatcher = {
            "cast": self.fit_cast,
            "minmax": self.fit_minmax,
            "normal": self.fit_normal,
            "iterative": self.fit_iterative,
        }

        assert qconfig.format is not None, (
            "Floating point quantizer must specify a format"
        )

        self.format = qconfig.format

        dict_format = {
            (8, "e4m3"): (4, 3, 7, 1),
            (8, "e5m2"): (5, 2, 15, 4),
            (4, "e2m1"): (2, 1, 1, 0),
            (4, "e3m0"): (3, 0, 2, 0),
            (16, "fp"): (5, 10, 15, 0),
            (16, "bf"): (8, 7, 127, 0),
        }
        if (self.num_bits, qconfig.format) not in dict_format:
            raise ValueError(
                f"Floating point format {qconfig.format} with {self.num_bits} bits is not supported."
            )

        # Init quant floating point grid
        self.E, self.M, self.bias, self.c = dict_format[(self.num_bits, self.format)]
        kmax = 2 ** (self.E + self.M) - 1 - self.c
        R = kmax // 2**self.M + (kmax % 2**self.M > 0) * 1 - 1
        R = 2 * R - 1

        Gn = [
            (2 ** (k // 2**self.M))
            * (2 ** (-self.bias))
            * (1 + (k % (2**self.M)) * 2 ** (-self.M))
            for k in range(2**self.M, kmax + 1)
        ]
        Gs = [2 ** (-self.bias) * (k * 2 ** (1 - self.M)) for k in range(1, 2**self.M)]
        Gh = torch.tensor(Gs + Gn)
        self.G = torch.concat((-torch.flip(Gh, [0]), torch.tensor([0.0]), Gh))

        self.vr = torch.tensor(
            [
                2 ** (abs(r - 1 - R // 2) + 1 - self.M - self.bias)
                for r in range(1, R + 1)
            ]
        )
        self.xr = torch.tensor([2 ** (r + 1 - self.bias) for r in range(1, R // 2 + 2)])
        self.xr[-1] = self.G[-1]
        self.xr = torch.concat((-torch.flip(self.xr, [0]), self.xr))

        C = np.linspace(1, 100, 10000)
        gres = self.snr(C, 1.0)
        self.Copt = C[np.argmax(gres)]

    def compute_quant_levels(
        self, scale: torch.Tensor, shift: torch.Tensor
    ) -> torch.Tensor:
        lk = self.G * scale + shift
        return lk

    def quant(self, input, scale, shift):
        num_blocks = input.shape[0]
        input = (input - shift.view(num_blocks, -1)) / scale.view(num_blocks, -1)
        mask = (input <= self.G[-1]) * (input >= self.G[0])
        return self.cast_to_fp(input), mask

    def dequant(self, input, scale, shift):
        num_blocks = input.shape[0]
        return scale.view(num_blocks, -1) * input + shift.view(num_blocks, -1)

    def cast_to_fp(self, x):
        x = torch.clamp(x, self.G[0].to(x.device), self.G[-1].to(x.device))

        v = 2 ** (torch.floor(torch.log2(torch.abs(x))) - self.M)
        v[torch.floor(torch.log2(torch.abs(x)) + self.bias) < 1] = 2 ** (
            1 - self.M - self.bias
        )

        Xf = v * torch.round(x / v)

        return Xf

    def snr(self, C, sigma2):
        Cmax = self.G[-1]
        s = C / Cmax
        sigma2 = torch.tensor(sigma2)
        C = torch.tensor(C)

        res = 2 * (1 + C**2 / sigma2) * self.q_function(C / torch.sqrt(sigma2))
        res += (
            -C
            * torch.sqrt(torch.tensor(2.0) / torch.pi)
            * torch.exp(-0.5 * (C**2) / sigma2)
            / torch.sqrt(sigma2)
        )

        F = self.gauss_cdf(s[:, None] * self.xr[None], 0.0, torch.sqrt(sigma2))
        p = F[:, 1:] - F[:, :-1]

        res += torch.sum(
            ((s[:, None] * self.vr[None]) ** 2) * p / (12 * sigma2), axis=1
        )
        return 1 / res

    def fit_cast(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = input.shape[-1]
        device = input.device
        return torch.ones(num_blocks, device=device), torch.zeros(
            num_blocks, device=device
        )

    def fit_minmax(
        self, input: torch.Tensor, _scale: torch.Tensor, _shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        if self.qconfig.symmetric:
            max_x = torch.max(torch.abs(input), axis=1)[0]
            scale = 2 * max_x / (2 * torch.max(self.G))
            shift = torch.zeros_like(scale, device=device)
        else:
            min_x = torch.min(input, axis=1)[0]
            max_x = torch.max(input, axis=1)[0]
            scale = (max_x - min_x) / (2 * torch.max(self.G))
            shift = min_x + torch.max(self.G) * scale
        return scale, shift

    def fit_normal(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = input.device
        x_std = torch.std(input, axis=1)
        scale = self.Copt * x_std / self.G[-1]
        if self.qconfig.symmetric:
            shift = torch.zeros_like(scale, device=device)
        else:
            x_mean = torch.mean(input, axis=1)
            shift = x_mean

        return scale, shift

    def fit_iterative(
        self, input: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_blocks = input.shape[0]
        for _ in range(self.qconfig.num_iters):
            xfloat, _ = self.quant(input, scale, shift)
            num_s = torch.sum((input - shift.view(num_blocks, -1)) * xfloat, axis=1)
            denum_s = torch.sum(xfloat**2, axis=1)
            scale = num_s / denum_s
            if not self.qconfig.symmetric:
                num_z = torch.sum(input - scale.view(num_blocks, -1) * xfloat, axis=1)
                denum_z = len(input)
                shift = num_z / denum_z
        return scale, shift


def get_quantizer(
    qconfig: QuantConfig, device: torch.device = torch.device("cpu")
) -> QuantizerBase | None:
    if qconfig.qtype:
        quantizers = {
            "uniform": UniformQuantizer,
            "nonuniform": NonUniformQuantizer,
            "float": FloatQuantizer,
        }
        return quantizers[qconfig.qtype](qconfig, device)
    return None
