from typing import Iterable, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional


class SequentialNetwork(nn.Module):
    """A sequential PyTorch model with the affine operations FANC requires."""

    def __init__(self, layers: Union[nn.Sequential, Iterable[nn.Module]]):
        super().__init__()
        self.layers = layers if isinstance(layers, nn.Sequential) else nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

    def activation_at(self, inputs: torch.Tensor, layer_index: int) -> torch.Tensor:
        if layer_index < 0 or layer_index >= len(self.layers):
            raise ValueError("activation layer is outside the network")
        current = inputs
        for index in range(layer_index + 1):
            current = self.layers[index](current)
        return current

    def apply_without_bias(self, layer_index: int, inputs: torch.Tensor) -> torch.Tensor:
        layer = self.layers[layer_index]
        if isinstance(layer, nn.Linear):
            return functional.linear(inputs, layer.weight)
        if isinstance(layer, nn.Conv2d):
            return functional.conv2d(
                inputs,
                layer.weight,
                stride=layer.stride,
                padding=layer.padding,
                dilation=layer.dilation,
                groups=layer.groups,
            )
        raise TypeError(f"layer {layer_index} is not an affine layer")

    def freeze(self) -> "SequentialNetwork":
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self
