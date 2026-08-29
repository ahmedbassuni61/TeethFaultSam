import math
import torch
import torch.nn as nn
from typing import List, Dict, Optional

class LoRALinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
            
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.rank = rank
        self.alpha = alpha
        
        # Scaling property
        self.scaling = alpha / rank
        
        # LoRA parameters
        self.lora_A = nn.Parameter(torch.empty((rank, self.in_features)))
        self.lora_B = nn.Parameter(torch.empty((self.out_features, rank)))
        
        self._reset_parameters()
        self.merged = False

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
            
        orig_out = self.original_layer(x)
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return orig_out + lora_out

    def merge(self):
        if not self.merged:
            # lora_B @ lora_A => (out_features, rank) @ (rank, in_features) = (out_features, in_features)
            delta_weight = (self.lora_B @ self.lora_A) * self.scaling
            self.original_layer.weight.data += delta_weight
            self.merged = True


def inject_lora(model: nn.Module, target_modules: Optional[List[str]] = None, rank: int = 4, alpha: float = 1.0) -> int:
    """
    Inject LoRA layers into specified Linear modules of a model.
    """
    if target_modules is None:
        target_modules = ['q_proj', 'v_proj']
        
    injected_count = 0
    # Create a list of tuples to avoid modifying dict while iterating
    modules_to_replace = []
    
    for name, module in model.named_modules():
        for target in target_modules:
            if name.endswith(target) and isinstance(module, nn.Linear):
                modules_to_replace.append((name, module))
                break
                
    for name, module in modules_to_replace:
        parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
        child_name = name.rsplit('.', 1)[-1] if '.' in name else name
        
        parent = model
        if parent_name:
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
        
        lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
        setattr(parent, child_name, lora_layer)
        injected_count += 1
                
    print(f"Injected LoRA into {injected_count} modules (rank={rank}, alpha={alpha}).")
    return injected_count


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """
    Return a list of LoRA parameters for the optimizer.
    """
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.extend([module.lora_A, module.lora_B])
    return params


def merge_lora(model: nn.Module):
    """
    Merge all LoRA layers into their original weights.
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()
            merged_count += 1
    print(f"Merged {merged_count} LoRA modules.")


def lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Return a state dict containing only LoRA parameters.
    """
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}
