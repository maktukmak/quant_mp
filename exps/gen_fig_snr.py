import matplotlib.pyplot as plt
import torch

from quant_mp.algs.analytic import snr_float, snr_general, snr_uniform
from quant_mp.datatypes import NF4, SF4, Fp4_e2m1, Fp4_e3m0, Int2, Int3, Int4

C = torch.linspace(1, 10, 100)

label = "4-bit normal-float"
data_format = NF4()
res = snr_general(data_format, C, 1.0)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label)

label = "4-bit student-float"
data_format = SF4()
res = snr_general(data_format, C, 1.0)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label)

label = "4-bit float (E3M0)"
data_format = Fp4_e3m0()
G = data_format.get_representable_values()
xr, vr = data_format.compute_interval_step_size()
res = snr_float(G, xr, vr, C, 1.0)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label)

label = "4-bit float (E2M1)"
data_format = Fp4_e2m1()
G = data_format.get_representable_values()
xr, vr = data_format.compute_interval_step_size()
res = snr_float(G, xr, vr, C, 1.0)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label)

label = "4-bit uniform"
data_format = Int4()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label, linestyle="--")

label = "3-bit uniform"
data_format = Int3()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label, linestyle="--")

label = "2-bit uniform"
data_format = Int2()
G = data_format.get_representable_values()
res = snr_uniform(C, 1.0, data_format.n_values)
print(label + " max snr: ", 10 * torch.log10(torch.max(res)))
plt.plot(C, res, label=label, linestyle="--")


plt.yscale("log")
plt.xlabel("Clipping point")
plt.legend()
plt.grid()
plt.ylabel("SNR")
plt.savefig("snr_vs_clip.jpg", bbox_inches="tight")
plt.show()
