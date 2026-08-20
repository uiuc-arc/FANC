from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .network import SequentialNetwork


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NETWORK_NAMES = (
    "fcnn7",
    "fconv4",
    "fcnn7_cifar",
    "fconv4_cifar",
)
TARGET_VARIANTS = ("quant8", "quant16", "float16")


def load_bundled_network(
    name: str,
    target_variant: Optional[str] = None,
) -> SequentialNetwork:
    """Load a bundled source network or target variant as a PyTorch network."""

    builders: Dict[str, Callable[[], nn.Sequential]] = {
        "fcnn7": lambda: _fcnn7(784),
        "fconv4": _fconv4,
        "fcnn7_cifar": lambda: _fcnn7(3072),
        "fconv4_cifar": _fconv4_cifar,
    }
    if name not in builders:
        raise ValueError(f"unknown bundled network: {name}")
    if target_variant is not None and target_variant not in TARGET_VARIANTS:
        raise ValueError(f"unknown target variant: {target_variant}")

    directory = REPOSITORY_ROOT / "proof_transfer" / "nets"
    suffix = "" if target_variant is None else f"_{target_variant}"
    path = directory / f"{name}{suffix}.onnx"
    model = builders[name]()
    _load_onnx_initializers(model, path)
    return SequentialNetwork(model).freeze()


def _load_onnx_initializers(model: nn.Sequential, path: Path) -> None:
    try:
        import onnx
        from onnx import numpy_helper
    except ImportError as error:
        raise ImportError("loading bundled networks requires onnx") from error

    graph = onnx.load(str(path)).graph
    initializers = {
        item.name: torch.from_numpy(numpy_helper.to_array(item).copy())
        for item in graph.initializer
    }
    state = model.state_dict()
    missing = [name for name in state if name not in initializers]
    if missing:
        raise ValueError(f"ONNX network is missing parameters: {missing}")
    model.load_state_dict(
        {name: initializers[name].to(dtype=value.dtype) for name, value in state.items()}
    )


def _fcnn7(input_features: int) -> nn.Sequential:
    layers = [nn.Flatten(), nn.Linear(input_features, 200)]
    for _ in range(6):
        layers.extend((nn.ReLU(), nn.Linear(200, 200)))
    layers.extend((nn.ReLU(), nn.Linear(200, 10)))
    return nn.Sequential(*layers)


def _fconv4() -> nn.Sequential:
    layers = [
        nn.Conv2d(1, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(392, 256),
    ]
    layers.extend(_linear_tail(256))
    return nn.Sequential(*layers)


def _fconv4_cifar() -> nn.Sequential:
    layers = [
        nn.Conv2d(3, 4, 13, stride=1, padding=6),
        nn.ReLU(),
        nn.Conv2d(4, 4, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4, 8, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 8, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(512, 256),
    ]
    layers.extend(_linear_tail(256))
    return nn.Sequential(*layers)


def _linear_tail(width: int) -> Tuple[nn.Module, ...]:
    layers = []
    for _ in range(4):
        layers.extend((nn.ReLU(), nn.Linear(width, width)))
    layers.extend((nn.ReLU(), nn.Linear(width, 10)))
    return tuple(layers)
