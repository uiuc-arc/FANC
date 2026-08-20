from dataclasses import dataclass

import torch

from .bounds import Box, Zonotope
from .templates import TemplateCollection, template_is_valid
from .verifier import ZonotopeVerifier


@dataclass
class TemplateVerificationResult:
    verified: torch.Tensor
    matched: torch.Tensor


class TemplateVerifier:
    """Validate and snapshot templates before verifying input regions."""

    def __init__(
        self,
        verifier: ZonotopeVerifier,
        templates: TemplateCollection,
        label: int,
        layer: int,
    ):
        if layer < 0 or layer >= len(verifier.network.layers):
            raise ValueError("template layer is outside the network")
        if any(template.layer != layer for template in templates.templates):
            raise ValueError("all templates must belong to the requested layer")
        self._verifier = verifier
        self._label = label
        self._layer = layer
        self.candidate_template_count = len(templates)
        valid = [
            template
            for template in templates.templates
            if template_is_valid(verifier, template, label)
        ]
        self._valid_template_count = len(valid)
        self._template_box = None
        if valid:
            self._template_box = Box(
                torch.cat(
                    [template.region.lower for template in valid], dim=0
                ).detach().clone(),
                torch.cat(
                    [template.region.upper for template in valid], dim=0
                ).detach().clone(),
            )

    @property
    def valid_template_count(self) -> int:
        return self._valid_template_count

    def verify(
        self, lower: torch.Tensor, upper: torch.Tensor
    ) -> TemplateVerificationResult:
        intermediate = self._verifier.propagate_bounds(
            lower, upper, end_layer=self._layer + 1
        )
        if self._template_box is None:
            matched = torch.zeros(
                intermediate.center.shape[0],
                dtype=torch.bool,
                device=intermediate.center.device,
            )
        else:
            matched = self._template_box.contains_each(intermediate.to_box()).any(
                dim=0
            )
        verified = matched.clone()
        remaining = ~matched
        if remaining.any():
            remainder = Zonotope(
                intermediate.center[remaining], intermediate.generators[remaining]
            )
            output = self._verifier.propagate(
                remainder, start_layer=self._layer + 1
            )
            verified[remaining] = self._verifier.verify(output, self._label)
        return TemplateVerificationResult(verified=verified, matched=matched)


def verify_with_templates(
    verifier: ZonotopeVerifier,
    lower: torch.Tensor,
    upper: torch.Tensor,
    label: int,
    templates: TemplateCollection,
    layer: int,
) -> torch.Tensor:
    """Verify regions using validated template matches and DeepZ fallback."""

    template_verifier = TemplateVerifier(verifier, templates, label, layer)
    return template_verifier.verify(lower, upper).verified
