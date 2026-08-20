from typing import Optional, Union

import torch
import torch.nn as nn

from .bounds import Box, Zonotope, _diagonal_generators, _remove_zero_generators
from .network import SequentialNetwork


class ZonotopeVerifier:
    """A DeepZ-style verifier for sequential PyTorch networks."""

    def __init__(self, network: SequentialNetwork):
        self.network = network.freeze()

    def propagate_bounds(
        self,
        lower: torch.Tensor,
        upper: torch.Tensor,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> Zonotope:
        return self.propagate(
            Zonotope.from_bounds(lower, upper),
            start_layer=start_layer,
            end_layer=end_layer,
        )

    def propagate(
        self,
        region: Union[Box, Zonotope],
        start_layer: int = 0,
        end_layer: Optional[int] = None,
    ) -> Zonotope:
        if isinstance(region, Box):
            region = region.to_zonotope()
        if not isinstance(region, Zonotope):
            raise TypeError("region must be a Box or Zonotope")

        layer_count = len(self.network.layers)
        stop = layer_count if end_layer is None else end_layer
        if start_layer < 0 or stop < start_layer or stop > layer_count:
            raise ValueError("invalid layer range")

        current = region
        for layer_index in range(start_layer, stop):
            current = self.apply_layer(current, layer_index)
        return current

    def apply_layer(self, region: Zonotope, layer_index: int) -> Zonotope:
        layer = self.network.layers[layer_index]
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            return self._apply_affine(region, layer_index)
        if isinstance(layer, nn.Flatten):
            if layer.start_dim != 1 or layer.end_dim != -1:
                raise TypeError("only default Flatten is supported")
            return Zonotope(
                layer(region.center),
                region.generators.flatten(start_dim=2),
            )
        if isinstance(layer, nn.ReLU):
            return self._apply_relu(region)
        if isinstance(layer, nn.AvgPool2d):
            return self._apply_linear_module(region, layer)
        if isinstance(layer, nn.Identity):
            return region.clone()
        raise TypeError(f"unsupported layer {layer_index}: {type(layer).__name__}")

    def verify(
        self, output: Zonotope, labels: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        if output.center.ndim != 2:
            raise ValueError("classification output must have shape [batch, classes]")

        batch_size, class_count = output.center.shape
        label_tensor = torch.as_tensor(labels, device=output.center.device, dtype=torch.long)
        if label_tensor.ndim == 0:
            label_tensor = label_tensor.expand(batch_size)
        if label_tensor.shape != (batch_size,):
            raise ValueError("labels must contain one class index per sample")
        if ((label_tensor < 0) | (label_tensor >= class_count)).any():
            raise ValueError("label index is outside the output range")

        batch_indices = torch.arange(batch_size, device=output.center.device)
        true_center = output.center[batch_indices, label_tensor]
        true_generators = output.generators[batch_indices, :, label_tensor]
        center_difference = output.center - true_center.unsqueeze(1)
        generator_difference = output.generators - true_generators.unsqueeze(2)
        upper_difference = center_difference + generator_difference.abs().sum(dim=1)
        upper_difference[batch_indices, label_tensor] = float("-inf")
        return (upper_difference < 0).all(dim=1)

    def verify_bounds(
        self, lower: torch.Tensor, upper: torch.Tensor, labels: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        return self.verify(self.propagate_bounds(lower, upper), labels)

    def _apply_affine(self, region: Zonotope, layer_index: int) -> Zonotope:
        layer = self.network.layers[layer_index]
        center = layer(region.center)
        batch_size, error_count = region.generators.shape[:2]
        if error_count == 0:
            return Zonotope(
                center,
                region.generators.new_zeros(batch_size, 0, *center.shape[1:]),
            )
        generator_inputs = region.generators.reshape(
            batch_size * error_count, *region.generators.shape[2:]
        )
        generators = self.network.apply_without_bias(layer_index, generator_inputs)
        generators = generators.reshape(batch_size, error_count, *center.shape[1:])
        return Zonotope(center, generators)

    @staticmethod
    def _apply_linear_module(region: Zonotope, layer: nn.Module) -> Zonotope:
        center = layer(region.center)
        batch_size, error_count = region.generators.shape[:2]
        if error_count == 0:
            return Zonotope(
                center,
                region.generators.new_zeros(batch_size, 0, *center.shape[1:]),
            )
        generator_inputs = region.generators.reshape(
            batch_size * error_count, *region.generators.shape[2:]
        )
        generators = layer(generator_inputs)
        generators = generators.reshape(batch_size, error_count, *center.shape[1:])
        return Zonotope(center, generators)

    @staticmethod
    def _apply_relu(region: Zonotope) -> Zonotope:
        lower, upper = region.bounds()
        positive = lower >= 0
        negative = upper <= 0
        unstable = ~(positive | negative)

        slope = torch.zeros_like(region.center)
        slope[positive] = 1
        slope[unstable] = upper[unstable] / (upper[unstable] - lower[unstable])

        intercept = torch.zeros_like(region.center)
        intercept[unstable] = (
            -upper[unstable] * lower[unstable]
            / (2 * (upper[unstable] - lower[unstable]))
        )

        center = slope * region.center + intercept
        generators = slope.unsqueeze(1) * region.generators
        new_errors = _diagonal_generators(intercept)
        generators = _remove_zero_generators(
            torch.cat((generators, new_errors), dim=1)
        )
        return Zonotope(center, generators)
