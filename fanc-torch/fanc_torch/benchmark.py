import csv
from dataclasses import dataclass
import hashlib
import itertools
import math
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import onnx

from .models import load_bundled_network


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class NetworkConfig:
    network: str
    dataset: str
    shape: Tuple[int, int, int]
    patch_size: int
    template_layer: int


@dataclass
class DatasetInputs:
    raw: torch.Tensor
    normalized: torch.Tensor
    labels: torch.Tensor


NETWORK_CONFIGS: Dict[str, NetworkConfig] = {
    "fcnn7": NetworkConfig("fcnn7", "mnist", (1, 28, 28), 7, 4),
    "fconv4": NetworkConfig("fconv4", "mnist", (1, 28, 28), 14, 6),
    "fcnn7_cifar": NetworkConfig(
        "fcnn7_cifar", "cifar10", (3, 32, 32), 16, 6
    ),
    "fconv4_cifar": NetworkConfig(
        "fconv4_cifar", "cifar10", (3, 32, 32), 32, 5
    ),
}
ATTACKS = ("patch", "l0_random", "l0_center")


def load_dataset(network: str) -> DatasetInputs:
    config = _network_config(network)
    directory = REPOSITORY_ROOT / "data"
    path = directory / f"{config.dataset}_test.csv"
    with path.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("dataset must contain at least one row")
    expected_columns = math.prod(config.shape) + 1
    if any(len(row) != expected_columns for row in rows):
        raise ValueError(f"dataset rows must contain {expected_columns} columns")
    try:
        labels = torch.tensor([int(row[0]) for row in rows], dtype=torch.long)
        flat = torch.tensor(
            [[float(value) for value in row[1:]] for row in rows],
            dtype=torch.float32,
        )
    except ValueError as error:
        raise ValueError("dataset values must be numeric") from error
    if ((labels < 0) | (labels > 9)).any():
        raise ValueError("dataset labels must be in [0, 9]")
    if not torch.isfinite(flat).all() or ((flat < 0) | (flat > 255)).any():
        raise ValueError("dataset pixels must be finite values in [0, 255]")
    flat /= 255
    if config.dataset == "mnist":
        raw = flat.reshape(-1, 1, 28, 28)
    else:
        raw = flat.reshape(-1, 32, 32, 3).permute(0, 3, 1, 2)
    return DatasetInputs(
        raw=raw,
        normalized=normalize_inputs(raw, config.dataset),
        labels=labels,
    )


def normalize_inputs(inputs: torch.Tensor, dataset: str) -> torch.Tensor:
    if dataset == "mnist":
        return inputs.clone()
    if dataset != "cifar10":
        raise ValueError(f"unknown dataset: {dataset}")
    means = inputs.new_tensor((0.4914, 0.4822, 0.4465)).reshape(1, 3, 1, 1)
    deviations = inputs.new_tensor((0.2023, 0.1994, 0.2010)).reshape(1, 3, 1, 1)
    return (inputs - means) / deviations


def build_experiment(
    network: str,
    image_count: int,
    attacks: Sequence[str],
    seed: int,
    specification_limit: Optional[int] = None,
) -> Dict[str, object]:
    if not _is_int(image_count) or image_count <= 0:
        raise ValueError("image count must be positive")
    if not _is_int(seed):
        raise ValueError("seed must be an integer")
    attacks = list(attacks)
    if (
        not attacks
        or any(not isinstance(attack, str) for attack in attacks)
        or len(set(attacks)) != len(attacks)
    ):
        raise ValueError("attacks must be nonempty and unique")
    if any(attack not in ATTACKS for attack in attacks):
        raise ValueError("unsupported experiment attack")
    if specification_limit is not None and (
        not _is_int(specification_limit) or specification_limit <= 0
    ):
        raise ValueError("specification limit must be positive")
    config = _network_config(network)
    dataset = load_dataset(network)
    model = load_bundled_network(network)
    with torch.no_grad():
        predictions = model(dataset.normalized).argmax(dim=1)
    eligible = (
        torch.nonzero(predictions == dataset.labels, as_tuple=False)
        .flatten()
        .tolist()
    )
    random.Random(seed).shuffle(eligible)
    selected = eligible[:image_count]
    if len(selected) != image_count:
        raise ValueError(
            "not enough images are correctly classified by the original network"
        )

    data_path = _data_path(config)
    network_path = _network_path(network)
    experiment = {
        "network": network,
        "seed": seed,
        "attacks": attacks,
        "specification_limit": specification_limit,
        "data_sha256": _sha256(data_path),
        "network_sha256": _sha256(network_path),
        "images": selected,
    }
    return experiment


def validate_experiment(experiment: Dict[str, object]) -> DatasetInputs:
    expected_fields = {
        "network",
        "seed",
        "attacks",
        "specification_limit",
        "data_sha256",
        "network_sha256",
        "images",
    }
    if set(experiment) != expected_fields:
        raise ValueError("experiment fields do not match the expected format")
    network = experiment.get("network")
    if not isinstance(network, str):
        raise ValueError("experiment network must be a string")
    config = _network_config(network)
    seed = experiment.get("seed")
    if not _is_int(seed):
        raise ValueError("experiment seed must be an integer")
    attacks = experiment.get("attacks")
    if (
        not isinstance(attacks, list)
        or not attacks
        or any(not isinstance(attack, str) for attack in attacks)
        or len(set(attacks)) != len(attacks)
        or any(attack not in ATTACKS for attack in attacks)
    ):
        raise ValueError("experiment attacks must be supported, nonempty, and unique")
    specification_limit = experiment.get("specification_limit")
    if specification_limit is not None and (
        not _is_int(specification_limit) or specification_limit <= 0
    ):
        raise ValueError("experiment specification limit must be positive")
    images = experiment.get("images")
    if (
        not isinstance(images, list)
        or not images
        or any(not _is_int(index) for index in images)
        or len(set(images)) != len(images)
    ):
        raise ValueError("experiment images must be nonempty, unique integers")
    if experiment.get("data_sha256") != _sha256(_data_path(config)):
        raise ValueError(
            "experiment dataset hash does not match the local dataset file"
        )
    if experiment.get("network_sha256") != network_file_sha256(config.network):
        raise ValueError("experiment network hash does not match the local model file")
    dataset = load_dataset(config.network)
    if any(index < 0 or index >= len(dataset.labels) for index in images):
        raise ValueError("experiment image index is outside the dataset")
    return dataset


def network_file_sha256(network: str, target_variant: Optional[str] = None) -> str:
    return _sha256(_network_path(network, target_variant))


def timing_summary(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    baseline_times = []
    fanc_times = []
    for result in results:
        run_baseline = sum(
            float(attack["timings"]["baseline_verification_seconds"])
            for attack in result["attacks"].values()
        )
        run_fanc = sum(
            float(attack["timings"]["fanc_verification_seconds"])
            for attack in result["attacks"].values()
        )
        baseline_times.append(run_baseline)
        fanc_times.append(run_fanc)
    return {
        "baseline_verification_seconds": _statistics(baseline_times),
        "fanc_verification_seconds": _statistics(fanc_times),
    }


def summarize_results(
    results: Sequence[Dict[str, object]],
    configuration: Dict[str, object],
    template_generation_seconds: Optional[float] = None,
    template_transformation_seconds: Optional[float] = None,
    template_validation_seconds: Optional[float] = None,
) -> Dict[str, object]:
    if not results:
        raise ValueError("at least one result is required")
    first = results[0]
    expected_metrics = _aggregate_metrics(first)
    for result in results[1:]:
        if _aggregate_metrics(result) != expected_metrics:
            raise ValueError("repeated runs produced different verification counts")
    timings = timing_summary(results)
    if template_generation_seconds is not None:
        timings["template_generation_seconds"] = float(template_generation_seconds)
    if template_transformation_seconds is not None:
        timings["template_transformation_seconds"] = float(
            template_transformation_seconds
        )
    if template_validation_seconds is not None:
        timings["template_validation_seconds"] = float(template_validation_seconds)
    return {
        "repetitions": len(results),
        "configuration": dict(configuration),
        "attacks": expected_metrics,
        "timings": timings,
        "environment": runtime_environment(),
    }


def runtime_environment() -> Dict[str, object]:
    revision, tracked_changes = _code_state()
    return {
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "onnx": onnx.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "device": "cpu",
        "dtype": "float32",
        "code_revision": revision,
        "tracked_changes": tracked_changes,
    }


def _code_state() -> Tuple[Optional[str], Optional[bool]]:
    root = REPOSITORY_ROOT
    try:
        revision = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain", "--untracked-files=no"),
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision, bool(status)


def _aggregate_metrics(result: Dict[str, object]) -> Dict[str, object]:
    return {
        name: {
            key: int(values[key])
            for key in (
                "specifications",
                "baseline_verified",
                "fanc_verified",
                "template_matches",
            )
        }
        for name, values in result["attacks"].items()
    }


def specification_descriptors(
    attack: str, height: int, width: int, seed: int, image_index: int
) -> List[Dict[str, object]]:
    rng = random.Random(_attack_seed(seed, image_index, attack))
    if attack == "patch":
        descriptors = [
            {"row": row, "column": column, "height": 2, "width": 2}
            for row in range(height - 1)
            for column in range(width - 1)
        ]
        rng.shuffle(descriptors)
        return descriptors
    if attack == "l0_random":
        return [
            {
                "pixels": [
                    [index // width, index % width]
                    for index in rng.sample(range(height * width), 3)
                ]
            }
            for _ in range(1000)
        ]
    if attack == "l0_center":
        center_rows = range((height - 5) // 2, (height - 5) // 2 + 5)
        center_columns = range((width - 4) // 2, (width - 4) // 2 + 4)
        pixels = list(itertools.product(center_rows, center_columns))
        descriptors = [
            {"pixels": [list(pixel) for pixel in combination]}
            for combination in itertools.combinations(pixels, 3)
        ]
        rng.shuffle(descriptors)
        return descriptors
    raise ValueError(f"unsupported attack: {attack}")


def specification_bounds(
    raw_image: torch.Tensor,
    descriptors: Sequence[Dict[str, object]],
    dataset: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if raw_image.shape[0] != 1:
        raise ValueError("specification generation expects one image")
    lower = raw_image.expand(len(descriptors), *raw_image.shape[1:]).clone()
    upper = lower.clone()
    for batch_index, descriptor in enumerate(descriptors):
        if "pixels" in descriptor:
            pixels = descriptor["pixels"]
        else:
            row = int(descriptor["row"])
            column = int(descriptor["column"])
            patch_height = int(descriptor["height"])
            patch_width = int(descriptor["width"])
            pixels = itertools.product(
                range(row, row + patch_height),
                range(column, column + patch_width),
            )
        for row, column in pixels:
            lower[batch_index, :, int(row), int(column)] = 0
            upper[batch_index, :, int(row), int(column)] = 1
    return normalize_inputs(lower, dataset), normalize_inputs(upper, dataset)


def _network_config(network: str) -> NetworkConfig:
    if network not in NETWORK_CONFIGS:
        raise ValueError(f"unknown experiment network: {network}")
    return NETWORK_CONFIGS[network]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _attack_seed(seed: int, image_index: int, attack: str) -> int:
    attack_code = sum((index + 1) * ord(value) for index, value in enumerate(attack))
    return seed * 1_000_003 + image_index * 97 + attack_code


def _data_path(config: NetworkConfig) -> Path:
    directory = REPOSITORY_ROOT / "data"
    return directory / f"{config.dataset}_test.csv"


def _network_path(
    network: str,
    target_variant: Optional[str] = None,
) -> Path:
    directory = REPOSITORY_ROOT / "proof_transfer" / "nets"
    suffix = "" if target_variant is None else f"_{target_variant}"
    return directory / f"{network}{suffix}.onnx"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _statistics(values: Sequence[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }
