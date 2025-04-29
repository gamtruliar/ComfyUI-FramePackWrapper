# LoRA network module: FramePack専用（musubi tuner準拠）
import math
import re
from typing import Dict, List, Optional, Type, Union
import torch
import torch.nn as nn

FRAMEPACK_TARGET_REPLACE_MODULES = [
    "HunyuanVideoTransformerBlock",
    "HunyuanVideoSingleTransformerBlock",
]

class LoRAModule(torch.nn.Module):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        self.lora_name = lora_name

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            torch.nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            torch.nn.init.zeros_(self.lora_up.weight)
        else:
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down in self.lora_down:
                torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
            for lora_up in self.lora_up:
                torch.nn.init.zeros_(lora_up.weight)

        if isinstance(alpha, torch.Tensor):
            alpha_buf = alpha.detach().clone()
        else:
            alpha_buf = torch.tensor(alpha)
        self.register_buffer("alpha", alpha_buf)

        self.scale = alpha / self.lora_dim

        self.multiplier = multiplier
        self.org_module = org_module
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x):
        org_forwarded = self.org_forward(x)
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)
            return org_forwarded + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return org_forwarded + torch.cat(lxs, dim=-1) * self.multiplier * self.scale

class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        **kwargs,
    ):
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha)
        self.org_module_ref = [org_module]
        self.enabled = True
        self.network = None

    def set_network(self, network):
        self.network = network

    def merge_to(self, sd, dtype, device, non_blocking=False):
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)
        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            down_weight = sd["lora_down.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            up_weight = sd["lora_up.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            if len(weight.size()) == 2:
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                weight = (
                    weight
                    + self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + self.multiplier * conved * self.scale
            org_sd["weight"] = weight.to(org_device, dtype=dtype)
            self.org_module.load_state_dict(org_sd)
        else:
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                down_weight = sd[f"lora_down.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)
                up_weight = sd[f"lora_up.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            org_sd["weight"] = weight.to(org_device, dtype)
            self.org_module.load_state_dict(org_sd)

def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = True,
    **kwargs,
):
    return create_network_from_weights(
        FRAMEPACK_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )

def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = True,
    **kwargs,
):
    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            dim = value.shape[0]
            modules_dim[lora_name] = dim
    module_class = LoRAInfModule if for_inference else LoRAModule
    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network

class LoRANetwork(torch.nn.Module):
    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Optional[List[nn.Module]],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        self.text_encoder_loras = []
        self.unet_loras, _ = self.create_modules(True, prefix, unet, target_replace_modules, module_class, modules_dim, modules_alpha, dropout, rank_dropout, module_dropout, exclude_patterns, include_patterns, verbose)

    def create_modules(
        self,
        is_unet: bool,
        pfx: str,
        root_module: torch.nn.Module,
        target_replace_mods: Optional[List[str]],
        module_class: Type[object],
        modules_dim: Optional[Dict[str, int]],
        modules_alpha: Optional[Dict[str, int]],
        dropout,
        rank_dropout,
        module_dropout,
        exclude_patterns,
        include_patterns,
        verbose,
    ):
        loras = []
        skipped = []
        for name, module in root_module.named_modules():
            # exclude_patternsによる除外判定
            if exclude_patterns is not None:
                excluded = False
                for pattern in exclude_patterns:
                    if re.match(pattern, name):
                        print(f"[LoRA][exclude] skip module: {name} (pattern: {pattern})")
                        excluded = True
                        break
                if excluded:
                    continue
            if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                for child_name, child_module in module.named_modules():
                    # exclude_patternsによる除外判定（子モジュール名にも適用）
                    if exclude_patterns is not None:
                        excluded = False
                        for pattern in exclude_patterns:
                            if re.match(pattern, child_name):
                                print(f"[LoRA][exclude] skip child module: {child_name} (pattern: {pattern})")
                                excluded = True
                                break
                        if excluded:
                            continue
                    is_linear = child_module.__class__.__name__ == "Linear"
                    is_conv2d = child_module.__class__.__name__ == "Conv2d"
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)
                    if is_linear or is_conv2d:
                        original_name = (name + "." if name else "") + child_name
                        lora_name = f"{pfx}.{original_name}".replace(".", "_")
                        dim = None
                        alpha = None
                        if modules_dim is not None:
                            if lora_name in modules_dim:
                                dim = modules_dim[lora_name]
                                alpha = modules_alpha[lora_name]
                        else:
                            if is_linear or is_conv2d_1x1:
                                dim = self.lora_dim
                                alpha = self.alpha
                            elif self.conv_lora_dim is not None:
                                dim = self.conv_lora_dim
                                alpha = self.conv_alpha
                        if dim is None or dim == 0:
                            skipped.append(lora_name)
                            continue
                        lora = module_class(
                            lora_name,
                            child_module,
                            self.multiplier,
                            dim,
                            alpha,
                            dropout=dropout,
                            rank_dropout=rank_dropout,
                            module_dropout=module_dropout,
                        )
                        loras.append(lora)
            if target_replace_mods is None:
                break
        return loras, skipped

    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False):
        for lora in self.unet_loras:
            sd_for_lora = {}
            for key in weights_sd.keys():
                if key.startswith(lora.lora_name):
                    sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
            if len(sd_for_lora) == 0:
                continue
            lora.merge_to(sd_for_lora, dtype, device, non_blocking)