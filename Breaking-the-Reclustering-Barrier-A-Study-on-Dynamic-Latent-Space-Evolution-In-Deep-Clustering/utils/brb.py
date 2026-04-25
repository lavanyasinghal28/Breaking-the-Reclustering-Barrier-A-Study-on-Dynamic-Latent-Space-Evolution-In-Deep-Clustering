import math
from collections import deque
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn


@torch.no_grad()
def _soft_reset_layer(layer: nn.Linear, alpha: float) -> None:
    """
    Soft reset for a single linear layer with scalar alpha.

    theta_new = alpha * theta_old + (1 - alpha) * phi_new
    where phi_new is Kaiming Uniform initialized.
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


@torch.no_grad()
def _soft_reset_layer_per_weight(layer: nn.Linear, alpha_weight: torch.Tensor) -> None:
    """Per-weight soft reset for a single linear layer."""
    phi_weight = torch.empty_like(layer.weight.data)
    nn.init.kaiming_uniform_(phi_weight, a=math.sqrt(5))
    layer.weight.data.mul_(alpha_weight).add_((1.0 - alpha_weight) * phi_weight)

    if layer.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(layer.weight)
        bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
        alpha_bias = alpha_weight.mean().item()
        phi_bias = torch.empty_like(layer.bias.data)
        nn.init.uniform_(phi_bias, -bound, bound)
        layer.bias.data.mul_(alpha_bias).add_((1.0 - alpha_bias) * phi_bias)


def _get_linear_layers(model: nn.Module, skip_embedding_layer: bool = True):
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if skip_embedding_layer and len(layers) > 1:
        layers = layers[:-1]
    return layers


def apply_soft_reset_to_network(model: nn.Module, alpha: float = 0.5,
                                skip_embedding_layer: bool = True) -> None:
    """Original BRB reset with one alpha for all layers."""
    for layer in _get_linear_layers(model, skip_embedding_layer=skip_embedding_layer):
        _soft_reset_layer(layer, alpha)


def ldar_reset(model: nn.Module, alpha_base: float = 0.8, gamma: float = 1.0,
               skip_embedding_layer: bool = True) -> None:
    """
    Layer-Depth Adaptive Reset.

    Earlier layers are preserved more; deeper layers are reset more.
    """
    linear_layers = _get_linear_layers(model, skip_embedding_layer=skip_embedding_layer)
    total = len(linear_layers)
    if total == 0:
        return

    for idx, layer in enumerate(linear_layers):
        depth_frac = idx / max(total - 1, 1)
        alpha_layer = 1.0 - (1.0 - alpha_base) * (depth_frac ** gamma)
        _soft_reset_layer(layer, alpha_layer)


class LSARScheduler:
    """Loss-Sensitive Annealed Reset scheduler."""

    def __init__(
        self,
        alpha_max: float = 0.90,
        alpha_min: float = 0.75,
        window: int = 5,
        eps: float = 1e-3,
        stagnation_factor: float = 0.88,
        total_epochs: int = 100,
        skip_embedding_layer: bool = True,
    ) -> None:
        self.alpha_max = alpha_max
        self.alpha_min = alpha_min
        self.window = window
        self.eps = eps
        self.stagnation_factor = stagnation_factor
        self.total_epochs = total_epochs
        self.skip_embedding_layer = skip_embedding_layer

        self._loss_history = deque(maxlen=window)
        self._last_alpha: Optional[float] = None

    def _cosine_alpha(self, epoch: int) -> float:
        progress = min(epoch, self.total_epochs) / max(self.total_epochs, 1)
        # cosine in [1, 0] gives alpha in [alpha_max, alpha_min]
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.alpha_min + (self.alpha_max - self.alpha_min) * cosine

    def _is_stagnating(self) -> bool:
        if len(self._loss_history) < self.window:
            return False
        return (self._loss_history[0] - self._loss_history[-1]) < self.eps

    def record_loss(self, cluster_loss_value: float) -> None:
        self._loss_history.append(float(cluster_loss_value))

    def maybe_reset(self, model: nn.Module, epoch: int, cluster_loss_value: float) -> float:
        self.record_loss(cluster_loss_value)

        alpha = self._cosine_alpha(epoch)
        if self._is_stagnating():
            alpha = max(0.5, alpha * self.stagnation_factor)

        for layer in _get_linear_layers(model, skip_embedding_layer=self.skip_embedding_layer):
            _soft_reset_layer(layer, alpha)

        self._last_alpha = alpha
        return alpha

    @property
    def last_alpha(self) -> Optional[float]:
        return self._last_alpha


def compute_diagonal_fisher_for_dcn_encoder(
    autoencoder: nn.Module,
    centers: torch.Tensor,
    dataloader,
    n_batches: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate diagonal Fisher for encoder params using DCN cluster loss."""
    if device is None:
        device = next(autoencoder.parameters()).device

    fisher: Dict[str, torch.Tensor] = {}
    autoencoder.eval()
    seen = 0

    for batch in dataloader:
        if seen >= n_batches:
            break

        batch_x = batch[0] if isinstance(batch, (list, tuple)) else batch
        batch_x = batch_x.to(device)

        autoencoder.zero_grad(set_to_none=True)
        z = autoencoder.encode(batch_x)
        dists = torch.cdist(z, centers.detach())
        min_dists = dists.min(dim=1).values
        cluster_loss = (min_dists ** 2).mean()
        cluster_loss.backward()

        for name, param in autoencoder.named_parameters():
            if param.grad is None:
                continue
            sq_grad = param.grad.detach().pow(2)
            fisher[name] = fisher[name] + sq_grad if name in fisher else sq_grad.clone()

        seen += 1

    denom = max(seen, 1)
    return {k: v / denom for k, v in fisher.items()}


def fwr_reset(
    model: nn.Module,
    fisher: Dict[str, torch.Tensor],
    lambda_: float = 3.0,
    alpha_floor: float = 0.3,
    skip_embedding_layer: bool = True,
) -> None:
    """Fisher-Weighted Reset with per-weight alpha maps."""
    named_linear_layers = [(name, m) for name, m in model.named_modules() if isinstance(m, nn.Linear)]
    if skip_embedding_layer and len(named_linear_layers) > 1:
        named_linear_layers = named_linear_layers[:-1]

    for layer_name, layer in named_linear_layers:
        weight_key = f"{layer_name}.weight"
        if weight_key in fisher:
            fisher_w = fisher[weight_key].to(layer.weight.device)
            fisher_mean = fisher_w.mean().clamp(min=1e-10)
            alpha_weight = torch.sigmoid(lambda_ * fisher_w / fisher_mean)
            alpha_weight = alpha_weight.clamp(min=alpha_floor, max=1.0)
        else:
            alpha_weight = torch.full_like(layer.weight, 0.8)

        _soft_reset_layer_per_weight(layer, alpha_weight)


def fwr_reset_with_estimation(
    autoencoder: nn.Module,
    centers: torch.Tensor,
    dataloader,
    n_batches: int = 10,
    lambda_: float = 3.0,
    alpha_floor: float = 0.3,
    skip_embedding_layer: bool = True,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Estimate Fisher and apply FWR in one call."""
    fisher = compute_diagonal_fisher_for_dcn_encoder(
        autoencoder=autoencoder,
        centers=centers,
        dataloader=dataloader,
        n_batches=n_batches,
        device=device,
    )
    fwr_reset(
        model=autoencoder,
        fisher=fisher,
        lambda_=lambda_,
        alpha_floor=alpha_floor,
        skip_embedding_layer=skip_embedding_layer,
    )
    return fisher


def apply_reset_method(
    method_name: str,
    autoencoder: nn.Module,
    epoch: int,
    cluster_loss_value: float,
    train_loader,
    centers: Optional[torch.Tensor],
    device: torch.device,
    ldar_kwargs: Optional[dict] = None,
    lsar_scheduler: Optional[LSARScheduler] = None,
    fwr_kwargs: Optional[dict] = None,
) -> Dict[str, float]:
    """Unified dispatch for reset strategies used by training loops."""
    method = method_name.lower()
    info: Dict[str, float] = {}

    if method == "brb":
        alpha = float((ldar_kwargs or {}).get("alpha", 0.7))
        apply_soft_reset_to_network(autoencoder, alpha=alpha)
        info["alpha"] = alpha
        return info

    if method == "ldar":
        kwargs = ldar_kwargs or {}
        alpha_base = float(kwargs.get("alpha_base", 0.8))
        gamma = float(kwargs.get("gamma", 1.0))
        ldar_reset(autoencoder, alpha_base=alpha_base, gamma=gamma)
        info["alpha_base"] = alpha_base
        info["gamma"] = gamma
        return info

    if method == "lsar":
        if lsar_scheduler is None:
            lsar_scheduler = LSARScheduler(total_epochs=100)
        alpha = lsar_scheduler.maybe_reset(
            model=autoencoder,
            epoch=epoch,
            cluster_loss_value=cluster_loss_value,
        )
        info["alpha"] = float(alpha)
        return info

    if method == "fwr":
        if centers is None:
            raise ValueError("FWR requires centers tensor for Fisher estimation.")
        kwargs = fwr_kwargs or {}
        n_batches = int(kwargs.get("n_batches", 8))
        lambda_ = float(kwargs.get("lambda_", 3.0))
        alpha_floor = float(kwargs.get("alpha_floor", 0.3))
        fwr_reset_with_estimation(
            autoencoder=autoencoder,
            centers=centers,
            dataloader=train_loader,
            n_batches=n_batches,
            lambda_=lambda_,
            alpha_floor=alpha_floor,
            device=device,
        )
        info["lambda"] = lambda_
        info["alpha_floor"] = alpha_floor
        info["fisher_batches"] = float(n_batches)
        return info

    raise ValueError(f"Unknown reset method: {method_name}")
