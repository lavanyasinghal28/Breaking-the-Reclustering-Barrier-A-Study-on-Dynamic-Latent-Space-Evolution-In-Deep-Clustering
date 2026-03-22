import torch
import torch.nn as nn
import math

@torch.no_grad()
def soft_reset_linear_layer(layer: nn.Linear, alpha: float):
    """
    Applies the BRB Soft reset to a single linear layer because base encoder is feedforward
    formula: θ_new = α * θ_old + (1 - α) * φ_new
    (where φ_new is randomly initialized weights using Kaiming Uniform)
    """

    # 1. Generate randomly initialized weights (φ_new)
    phi_weight = torch.empty_like(layer.weight.data)
    nn.init.kaiming_uniform_(phi_weight, a = math.sqrt(5))

    # 2. Blend the old weights with the new random weights
    layer.weight.data.mul_(alpha)
    layer.weight.data.add_((1-alpha)*phi_weight)

    # 3. exact same for the biases
    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        phi_bias = torch.empty_like(layer.bias.data)
        nn.init.uniform_(phi_bias, -bound, bound)
        layer.bias.data.mul_(alpha)
        layer.bias.data.add_((1-alpha) * phi_bias)

def apply_soft_reset_to_network(model: nn.Module, alpha: float = 0.5):
    """
    Loops through the autoencoder and applies the soft reset to all linear layers.
    """
    print(f"Applying BRB Soft Reset with alpha={alpha}...")
    for module in model.modules():
        if isinstance(module, nn.Linear):
            soft_reset_linear_layer(module, alpha)
