from dataclasses import dataclass
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import torch

from .bounds import Box, Zonotope
from .verifier import ZonotopeVerifier


@dataclass
class Template:
    layer: int
    region: Box
    patch: Tuple[int, int, int]
    template_radius: float


class TemplateCollection:
    """Box templates indexed by their zero-based sequential layer."""

    def __init__(self, templates: Iterable[Template] = ()):
        self.templates = []
        for template in templates:
            self.add(template)

    def add(self, template: Template) -> None:
        if template.region.lower.shape[0] != 1:
            raise ValueError("stored templates must have batch size one")
        self.templates.append(template)

    def at_layer(self, layer: int) -> List[Template]:
        return [template for template in self.templates if template.layer == layer]

    def matches(self, layer: int, region: Union[Box, Zonotope]) -> torch.Tensor:
        if isinstance(region, Zonotope):
            region = region.to_box()
        if not isinstance(region, Box):
            raise TypeError("region must be a Box or Zonotope")

        candidates = self.at_layer(layer)
        if not candidates:
            return torch.zeros(
                region.lower.shape[0], dtype=torch.bool, device=region.lower.device
            )
        template_box = Box(
            torch.cat([template.region.lower for template in candidates], dim=0),
            torch.cat([template.region.upper for template in candidates], dim=0),
        )
        return template_box.contains_each(region).any(dim=0)

    def __len__(self) -> int:
        return len(self.templates)


def template_radius_candidates(minimum: float = 1 / 256) -> Tuple[float, ...]:
    if minimum <= 0 or minimum > 1:
        raise ValueError("minimum template radius must be in (0, 1]")
    values = []
    current = 1.0
    while current >= minimum:
        values.append(current)
        current *= 0.5
    values.append(0.0)
    return tuple(values)


def generate_templates(
    verifier: ZonotopeVerifier,
    image: torch.Tensor,
    label: int,
    layer: int,
    patch_size: int,
    template_radii: Sequence[float] = template_radius_candidates(),
    input_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> TemplateCollection:
    """Generate patch-partitioned box templates from raw pixel coordinates."""

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("template generation expects one NCHW image")
    if patch_size <= 0:
        raise ValueError("patch size must be positive")
    height, width = image.shape[-2:]
    if height % patch_size or width % patch_size:
        raise ValueError("patch size must evenly divide image height and width")
    if layer < 0 or layer >= len(verifier.network.layers):
        raise ValueError("template layer is outside the network")

    candidates = _prepare_radius_candidates(template_radii)
    transform = input_transform if input_transform is not None else _identity
    templates = TemplateCollection()
    for row in range(0, height, patch_size):
        for column in range(0, width, patch_size):
            template = _generate_patch_template(
                verifier,
                image,
                label,
                layer,
                row,
                column,
                patch_size,
                candidates,
                transform,
            )
            if template is not None:
                templates.add(template)
    return templates


def template_is_valid(
    verifier: ZonotopeVerifier, template: Template, label: int
) -> bool:
    output = verifier.propagate(template.region, start_layer=template.layer + 1)
    return bool(verifier.verify(output, label).item())


def _prepare_radius_candidates(values: Sequence[float]) -> Tuple[float, ...]:
    candidates = {float(value) for value in values}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in candidates):
        raise ValueError("template radii must be finite and in [0, 1]")
    candidates.add(0.0)
    return tuple(sorted(candidates, reverse=True))


def _generate_patch_template(
    verifier: ZonotopeVerifier,
    image: torch.Tensor,
    label: int,
    layer: int,
    row: int,
    column: int,
    patch_size: int,
    candidates: Sequence[float],
    input_transform: Callable[[torch.Tensor], torch.Tensor],
) -> Union[Template, None]:
    for template_radius in candidates:
        lower, upper = _patch_bounds(
            image, row, column, patch_size, template_radius
        )
        lower = input_transform(lower)
        upper = input_transform(upper)
        prefix = verifier.propagate_bounds(lower, upper, end_layer=layer + 1)
        template = Template(
            layer=layer,
            region=prefix.to_box(),
            patch=(row, column, patch_size),
            template_radius=template_radius,
        )
        if template_is_valid(verifier, template, label):
            return template
    return None


def _patch_bounds(
    image: torch.Tensor,
    row: int,
    column: int,
    patch_size: int,
    template_radius: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lower = image.clone()
    upper = image.clone()
    patch = image[
        :, :, row : row + patch_size, column : column + patch_size
    ]
    lower[:, :, row : row + patch_size, column : column + patch_size] = (
        patch - template_radius
    ).clamp_min(0)
    upper[:, :, row : row + patch_size, column : column + patch_size] = (
        patch + template_radius
    ).clamp_max(1)
    return lower, upper


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value
