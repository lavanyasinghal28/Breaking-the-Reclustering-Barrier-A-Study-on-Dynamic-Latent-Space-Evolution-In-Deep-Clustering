import math
import torch
import torch.nn as nn


@torch.no_grad()
def soft_reset_linear_layer(layer: nn.Linear, alpha: float):
    """
    BRB Soft reset for a single linear layer.
    θ_new = α * θ_old + (1 - α) * φ_new
    where φ_new is Kaiming Uniform initialised.
    """
    phi_weight = torch.empty_like(layer.weight.data)
    nn.init.kaiming_uniform_(phi_weight, a=math.sqrt(5))

    layer.weight.data.mul_(alpha)
    layer.weight.data.add_((1 - alpha) * phi_weight)

    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        phi_bias = torch.empty_like(layer.bias.data)
        nn.init.uniform_(phi_bias, -bound, bound)
        layer.bias.data.mul_(alpha)
        layer.bias.data.add_((1 - alpha) * phi_bias)


def apply_soft_reset_to_network(model: nn.Module, alpha: float = 0.5):
    """Apply BRB soft reset to every nn.Linear layer in the model."""
    print(f"Applying BRB Soft Reset with alpha={alpha}...")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            soft_reset_linear_layer(module, alpha)
