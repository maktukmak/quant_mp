from copy import deepcopy

import torch
from tqdm import tqdm

from quant_mp.config import QuantModuleConfig
from quant_mp.QModules import QConv2d, QLinear


def patch_model(model, config: QuantModuleConfig):
    def replace_layer(module: torch.nn.Module):
        if isinstance(module, torch.nn.Linear):
            target_state_dict = deepcopy(module.state_dict())
            bias = True if module.bias is not None else False
            new_module = QLinear(
                module.in_features,
                module.out_features,
                bias,
                module.weight.device,
                module.weight.dtype,
                config,
            )
            new_module.load_state_dict(target_state_dict, strict=False)

            return new_module
        elif isinstance(module, torch.nn.Conv2d):
            target_state_dict = deepcopy(module.state_dict())
            bias = True if module.bias is not None else False
            new_module = QConv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,  # pyright: ignore[reportArgumentType]
                module.stride,  # pyright: ignore[reportArgumentType]
                module.padding,  # pyright: ignore[reportArgumentType]
                module.dilation,  # pyright: ignore[reportArgumentType]
                module.groups,
                bias,
                module.padding_mode,
                config,
            )
            new_module.load_state_dict(target_state_dict, strict=False)
            return new_module
        else:
            return module

    def recursive_setattr(obj: torch.nn.Module, attr, value):
        attr = attr.split(".", 1)
        if len(attr) == 1:
            setattr(obj, attr[0], value)
        else:
            recursive_setattr(getattr(obj, attr[0]), attr[1], value)

    for name, module in tqdm(tuple(model.named_modules()), desc="Patching Model"):
        if name and not name.endswith("lm_head"):
            recursive_setattr(model, name, replace_layer(module))
