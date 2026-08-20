import math
from typing import Any, Tuple

import torch
import torch.nn as nn

from .bounds import Box
from .network import SequentialNetwork
from .templates import Template, TemplateCollection
from .verifier import ZonotopeVerifier


def transform_template(
    template: Template,
    target_activation: torch.Tensor,
    transfer_radius: float,
) -> Template:
    """Expand a template around the target activation."""

    transfer_radius = _validate_transfer_radius(transfer_radius)
    if target_activation.shape != template.region.lower.shape:
        raise ValueError("target activation and template shapes differ")
    if (
        target_activation.dtype != template.region.lower.dtype
        or target_activation.device != template.region.lower.device
    ):
        raise ValueError("target activation and template must share dtype and device")

    target_lower = target_activation - transfer_radius
    target_upper = target_activation + transfer_radius
    region = Box(
        torch.minimum(template.region.lower, target_lower),
        torch.maximum(template.region.upper, target_upper),
    )
    return Template(template.layer, region, template.patch, template.template_radius)


def transfer_templates(
    source_verifier: ZonotopeVerifier,
    target_verifier: ZonotopeVerifier,
    templates: TemplateCollection,
    image: torch.Tensor,
    transfer_radius: float,
) -> TemplateCollection:
    """Transform generated templates for the target network."""

    _validate_compatible_networks(source_verifier.network, target_verifier.network)
    transfer_radius = _validate_transfer_radius(transfer_radius)
    if image.ndim == 0 or image.shape[0] != 1:
        raise ValueError("template transfer expects one input")

    activations = {}
    transferred = TemplateCollection()
    with torch.no_grad():
        for template in templates.templates:
            if template.layer not in activations:
                activations[template.layer] = target_verifier.network.activation_at(
                    image, template.layer
                )
            candidate = transform_template(
                template, activations[template.layer], transfer_radius
            )
            transferred.add(candidate)
    return transferred


def _validate_transfer_radius(transfer_radius: float) -> float:
    transfer_radius = float(transfer_radius)
    if not math.isfinite(transfer_radius) or transfer_radius < 0:
        raise ValueError("transfer radius must be finite and nonnegative")
    return transfer_radius


def _validate_compatible_networks(
    source: SequentialNetwork, target: SequentialNetwork
) -> None:
    if len(source.layers) != len(target.layers):
        raise ValueError("source and target networks must have the same architecture")
    for source_layer, target_layer in zip(source.layers, target.layers):
        if _layer_signature(source_layer) != _layer_signature(target_layer):
            raise ValueError("source and target networks must have the same architecture")


def _layer_signature(layer: nn.Module) -> Tuple[Any, ...]:
    if isinstance(layer, nn.Linear):
        return (type(layer), layer.in_features, layer.out_features, layer.bias is not None)
    if isinstance(layer, nn.Conv2d):
        return (
            type(layer),
            layer.in_channels,
            layer.out_channels,
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.dilation,
            layer.groups,
            layer.padding_mode,
            layer.bias is not None,
        )
    if isinstance(layer, nn.Flatten):
        return (type(layer), layer.start_dim, layer.end_dim)
    if isinstance(layer, nn.AvgPool2d):
        return (
            type(layer),
            layer.kernel_size,
            layer.stride,
            layer.padding,
            layer.ceil_mode,
            layer.count_include_pad,
            layer.divisor_override,
        )
    if isinstance(layer, (nn.ReLU, nn.Identity)):
        return (type(layer),)
    raise TypeError(f"unsupported architecture layer: {type(layer).__name__}")
