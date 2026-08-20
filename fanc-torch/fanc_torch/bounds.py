from typing import Tuple

import torch


def _reduce_nonbatch(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 1:
        return value
    return value.flatten(start_dim=1).all(dim=1)


def _diagonal_generators(value: torch.Tensor) -> torch.Tensor:
    flattened = value.flatten(start_dim=1)
    active = torch.nonzero((flattened != 0).any(dim=0), as_tuple=False).flatten()
    generators = value.new_zeros(value.shape[0], active.numel(), flattened.shape[1])
    if active.numel():
        errors = torch.arange(active.numel(), device=value.device)
        generators[:, errors, active] = flattened[:, active]
    return generators.reshape(value.shape[0], active.numel(), *value.shape[1:])


def _remove_zero_generators(generators: torch.Tensor) -> torch.Tensor:
    active = (generators != 0).flatten(start_dim=2).any(dim=2).any(dim=0)
    return generators[:, active]


class Zonotope:
    """A batched affine form with one shared error-term axis per sample."""

    def __init__(self, center: torch.Tensor, generators: torch.Tensor):
        if center.ndim < 2:
            raise ValueError("center must include batch and feature dimensions")
        if generators.ndim != center.ndim + 1:
            raise ValueError("generators must add one error-term dimension")
        if generators.shape[0] != center.shape[0]:
            raise ValueError("center and generators must have the same batch size")
        if generators.shape[2:] != center.shape[1:]:
            raise ValueError("center and generator feature shapes must match")
        if generators.device != center.device or generators.dtype != center.dtype:
            raise ValueError("center and generators must share device and dtype")
        if not torch.is_floating_point(center):
            raise TypeError("zonotopes must use a floating-point dtype")

        self.center = center
        self.generators = generators

    @classmethod
    def from_bounds(cls, lower: torch.Tensor, upper: torch.Tensor) -> "Zonotope":
        Box._validate_bounds(lower, upper)
        center = (lower + upper) * 0.5
        generators = _diagonal_generators((upper - lower) * 0.5)
        return cls(center, generators)

    @property
    def lower(self) -> torch.Tensor:
        return self.center - self.generators.abs().sum(dim=1)

    @property
    def upper(self) -> torch.Tensor:
        return self.center + self.generators.abs().sum(dim=1)

    @property
    def unstable(self) -> torch.Tensor:
        return (self.lower < 0) & (self.upper > 0)

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lower, self.upper

    def clone(self) -> "Zonotope":
        return Zonotope(self.center.clone(), self.generators.clone())

    def to_box(self) -> "Box":
        return Box(self.lower, self.upper)

    def scale(self, factor: torch.Tensor) -> "Zonotope":
        try:
            factor = torch.broadcast_to(factor, self.center.shape)
        except RuntimeError as error:
            raise ValueError("factor must broadcast to the center shape") from error
        if not torch.isfinite(factor).all() or (factor < 0).any():
            raise ValueError("factor must be finite and nonnegative")
        self.generators = self.generators * factor.unsqueeze(1)
        return self

    def is_finite(self) -> torch.Tensor:
        finite = torch.isfinite(self.center) & torch.isfinite(self.generators).all(dim=1)
        return _reduce_nonbatch(finite)


class Box:
    """A batched axis-aligned region."""

    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self._validate_bounds(lower, upper)
        self.lower = lower
        self.upper = upper

    @staticmethod
    def _validate_bounds(lower: torch.Tensor, upper: torch.Tensor) -> None:
        if lower.ndim < 2:
            raise ValueError("bounds must include batch and feature dimensions")
        if lower.shape != upper.shape:
            raise ValueError("lower and upper bounds must have the same shape")
        if lower.device != upper.device or lower.dtype != upper.dtype:
            raise ValueError("lower and upper bounds must share device and dtype")
        if not torch.is_floating_point(lower):
            raise TypeError("bounds must use a floating-point dtype")
        if (lower > upper).any():
            raise ValueError("lower bounds must not exceed upper bounds")

    @property
    def center(self) -> torch.Tensor:
        return (self.lower + self.upper) * 0.5

    @property
    def generators(self) -> torch.Tensor:
        return _diagonal_generators((self.upper - self.lower) * 0.5)

    @property
    def unstable(self) -> torch.Tensor:
        return (self.lower < 0) & (self.upper > 0)

    def bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lower, self.upper

    def clone(self) -> "Box":
        return Box(self.lower.clone(), self.upper.clone())

    def to_zonotope(self) -> Zonotope:
        return Zonotope(self.center, self.generators)

    def scale(self, factor: torch.Tensor) -> "Box":
        try:
            factor = torch.broadcast_to(factor, self.center.shape)
        except RuntimeError as error:
            raise ValueError("factor must broadcast to the box shape") from error
        if not torch.isfinite(factor).all() or (factor < 0).any():
            raise ValueError("factor must be finite and nonnegative")
        center = self.center
        self.lower = center + factor * (self.lower - center)
        self.upper = center + factor * (self.upper - center)
        return self

    def contains(self, other: "Box") -> torch.Tensor:
        if self.lower.shape != other.lower.shape:
            raise ValueError("boxes must have the same shape")
        if self.lower.device != other.lower.device or self.lower.dtype != other.lower.dtype:
            raise ValueError("boxes must share device and dtype")
        contained = (self.lower <= other.lower) & (self.upper >= other.upper)
        return _reduce_nonbatch(contained)

    def contains_each(self, other: "Box") -> torch.Tensor:
        if self.lower.shape[1:] != other.lower.shape[1:]:
            raise ValueError("template and region feature shapes must match")
        if self.lower.device != other.lower.device or self.lower.dtype != other.lower.dtype:
            raise ValueError("templates and regions must share device and dtype")
        template_lower = self.lower.flatten(start_dim=1).unsqueeze(1)
        template_upper = self.upper.flatten(start_dim=1).unsqueeze(1)
        region_lower = other.lower.flatten(start_dim=1).unsqueeze(0)
        region_upper = other.upper.flatten(start_dim=1).unsqueeze(0)
        return (
            (template_lower <= region_lower) & (template_upper >= region_upper)
        ).all(dim=2)

    def is_finite(self) -> torch.Tensor:
        finite = torch.isfinite(self.lower) & torch.isfinite(self.upper)
        return _reduce_nonbatch(finite)
