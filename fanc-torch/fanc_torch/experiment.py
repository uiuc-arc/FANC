from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch

from .algorithm import TemplateVerifier
from .benchmark import (
    NETWORK_CONFIGS,
    NetworkConfig,
    network_file_sha256,
    normalize_inputs,
    specification_bounds,
    specification_descriptors,
    validate_experiment,
)
from .models import load_bundled_network
from .templates import (
    TemplateCollection,
    generate_templates,
    template_radius_candidates,
)
from .transfer import transfer_templates
from .verifier import ZonotopeVerifier


@dataclass(frozen=True)
class SourceImage:
    index: int
    label: int
    raw: torch.Tensor
    normalized: torch.Tensor
    templates: TemplateCollection


@dataclass(frozen=True)
class PreparedSource:
    experiment: Dict[str, object]
    config: NetworkConfig
    verifier: ZonotopeVerifier
    images: Tuple[SourceImage, ...]
    threads: int
    template_generation_seconds: float


@dataclass(frozen=True)
class TargetImage:
    source: SourceImage
    template_verifier: TemplateVerifier


@dataclass(frozen=True)
class VerificationProgress:
    image: int
    images: int
    specification: str
    processed: int
    total: int
    baseline_verified: int
    fanc_verified: int
    template_matches: int


@dataclass(frozen=True)
class PreparedTarget:
    source: PreparedSource
    target_variant: str
    verifier: ZonotopeVerifier
    images: Tuple[TargetImage, ...]
    transfer_radius: float
    template_transformation_seconds: float
    template_validation_seconds: float


def prepare_source(
    experiment: Dict[str, object],
    threads: int = 1,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PreparedSource:
    dataset = validate_experiment(experiment)
    if threads <= 0:
        raise ValueError("thread count must be positive")
    if torch.get_num_threads() != threads:
        raise ValueError("configure PyTorch thread count before running the experiment")

    config = NETWORK_CONFIGS[str(experiment["network"])]
    source_network = load_bundled_network(config.network)
    source_verifier = ZonotopeVerifier(source_network)

    images = []
    template_generation_seconds = 0.0
    image_indices = list(experiment["images"])
    selected = torch.tensor(image_indices, dtype=torch.long)
    with torch.no_grad():
        predictions = source_network(dataset.normalized[selected]).argmax(dim=1)
    if not torch.equal(predictions, dataset.labels[selected]):
        raise ValueError(
            "experiment images must be correctly classified by the original network"
        )
    for position, index in enumerate(image_indices, start=1):
        if progress is not None:
            progress(position, len(image_indices))
        index = int(index)
        label = int(dataset.labels[index])
        raw = dataset.raw[index : index + 1]
        normalized = dataset.normalized[index : index + 1]
        started = perf_counter()
        templates = generate_templates(
            source_verifier,
            raw,
            label,
            layer=config.template_layer,
            patch_size=config.patch_size,
            template_radii=template_radius_candidates(),
            input_transform=lambda value: normalize_inputs(value, config.dataset),
        )
        template_generation_seconds += perf_counter() - started
        images.append(SourceImage(index, label, raw, normalized, templates))
    return PreparedSource(
        experiment,
        config,
        source_verifier,
        tuple(images),
        threads,
        template_generation_seconds,
    )


def prepare_target(
    source: PreparedSource,
    target_variant: str,
    transfer_radius: float,
    progress: Optional[Callable[[int, int], None]] = None,
) -> PreparedTarget:
    target_network = load_bundled_network(source.config.network, target_variant)
    target_verifier = ZonotopeVerifier(target_network)
    images = []
    template_transformation_seconds = 0.0
    template_validation_seconds = 0.0
    for position, image in enumerate(source.images, start=1):
        if progress is not None:
            progress(position, len(source.images))
        started = perf_counter()
        transferred = transfer_templates(
            source.verifier,
            target_verifier,
            image.templates,
            image.normalized,
            transfer_radius=transfer_radius,
        )
        template_transformation_seconds += perf_counter() - started
        started = perf_counter()
        template_verifier = TemplateVerifier(
            target_verifier,
            transferred,
            image.label,
            source.config.template_layer,
        )
        template_validation_seconds += perf_counter() - started
        images.append(TargetImage(image, template_verifier))
    return PreparedTarget(
        source,
        target_variant,
        target_verifier,
        tuple(images),
        float(transfer_radius),
        template_transformation_seconds,
        template_validation_seconds,
    )


def run_experiment(
    target: PreparedTarget,
    batch_size: int,
    progress: Optional[Callable[[VerificationProgress], None]] = None,
    fanc_first: bool = False,
) -> Dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if torch.get_num_threads() != target.source.threads:
        raise ValueError("configure PyTorch thread count before running the experiment")

    experiment = target.source.experiment
    config = target.source.config
    image_results = []
    for position, image in enumerate(target.images, start=1):
        image_results.append(
            _run_image(
                image,
                config,
                target.verifier,
                tuple(experiment["attacks"]),
                int(experiment["seed"]),
                experiment["specification_limit"],
                batch_size,
                fanc_first,
                progress,
                position,
                len(target.images),
            )
        )

    attacks = {}
    for image in image_results:
        for name in sorted(image["attacks"]):
            attack = image["attacks"][name]
            totals = attacks.setdefault(
                name,
                {
                    "specifications": 0,
                    "baseline_verified": 0,
                    "fanc_verified": 0,
                    "template_matches": 0,
                    "timings": {
                        "baseline_verification_seconds": 0.0,
                        "fanc_verification_seconds": 0.0,
                    },
                },
            )
            for key in (
                "specifications",
                "baseline_verified",
                "fanc_verified",
                "template_matches",
            ):
                totals[key] += int(attack[key])
            for key in totals["timings"]:
                totals["timings"][key] += float(attack["timings"][key])
    result = {
        "measurement_order": "fanc_first" if fanc_first else "baseline_first",
        "attacks": attacks,
    }
    return result


def verification_configuration(
    target: PreparedTarget,
    batch_size: int,
) -> Dict[str, object]:
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    return {
        "network": target.source.config.network,
        "target_variant": target.target_variant,
        "target_network_sha256": network_file_sha256(
            target.source.config.network, target.target_variant
        ),
        "transfer_radius": target.transfer_radius,
        "batch_size": batch_size,
        "threads": target.source.threads,
    }


def _run_image(
    image: TargetImage,
    config: NetworkConfig,
    target_verifier: ZonotopeVerifier,
    attack_names: Sequence[str],
    seed: int,
    specification_limit: Optional[int],
    batch_size: int,
    fanc_first: bool,
    progress: Optional[Callable[[VerificationProgress], None]],
    image_position: int,
    image_count: int,
) -> Dict[str, object]:
    attacks = {}
    for name in attack_names:
        descriptors = specification_descriptors(
            name, config.shape[1], config.shape[2], seed, image.source.index
        )
        if specification_limit is not None:
            descriptors = descriptors[: int(specification_limit)]
        attacks[name] = _run_attack(
            descriptors,
            image.source.raw,
            config.dataset,
            image.source.label,
            target_verifier,
            image.template_verifier,
            batch_size,
            fanc_first,
            progress,
            image_position,
            image_count,
            name,
        )
    return {
        "index": image.source.index,
        "attacks": attacks,
    }


def _run_attack(
    descriptors: Sequence[Dict[str, object]],
    raw_image: torch.Tensor,
    dataset: str,
    label: int,
    target_verifier: ZonotopeVerifier,
    template_verifier: TemplateVerifier,
    batch_size: int,
    fanc_first: bool,
    progress: Optional[Callable[[VerificationProgress], None]],
    image_position: int,
    image_count: int,
    specification: str,
) -> Dict[str, object]:
    specification_count = 0
    baseline_verified = 0
    fanc_verified = 0
    template_matches = 0
    baseline_verification_seconds = 0.0
    fanc_verification_seconds = 0.0

    def report_progress() -> None:
        if progress is not None:
            progress(
                VerificationProgress(
                    image=image_position,
                    images=image_count,
                    specification=specification,
                    processed=specification_count,
                    total=len(descriptors),
                    baseline_verified=baseline_verified,
                    fanc_verified=fanc_verified,
                    template_matches=template_matches,
                )
            )

    report_progress()
    if descriptors:
        warmup = descriptors[:batch_size]
        lower, upper = specification_bounds(raw_image, warmup, dataset)
        target_verifier.verify_bounds(lower, upper, label)
        template_verifier.verify(lower, upper)

    for start in range(0, len(descriptors), batch_size):
        batch = descriptors[start : start + batch_size]
        lower, upper = specification_bounds(raw_image, batch, dataset)

        if fanc_first:
            started = perf_counter()
            fanc = template_verifier.verify(lower, upper)
            fanc_verification_seconds += perf_counter() - started
            started = perf_counter()
            baseline = target_verifier.verify_bounds(lower, upper, label)
            baseline_verification_seconds += perf_counter() - started
        else:
            started = perf_counter()
            baseline = target_verifier.verify_bounds(lower, upper, label)
            baseline_verification_seconds += perf_counter() - started
            started = perf_counter()
            fanc = template_verifier.verify(lower, upper)
            fanc_verification_seconds += perf_counter() - started

        baseline_values = baseline.tolist()
        fanc_values = fanc.verified.tolist()
        template_match_values = fanc.matched.tolist()
        specification_count += len(batch)
        baseline_verified += sum(bool(value) for value in baseline_values)
        fanc_verified += sum(bool(value) for value in fanc_values)
        template_matches += sum(bool(value) for value in template_match_values)
        report_progress()

    return {
        "specifications": specification_count,
        "baseline_verified": baseline_verified,
        "fanc_verified": fanc_verified,
        "template_matches": template_matches,
        "timings": {
            "baseline_verification_seconds": baseline_verification_seconds,
            "fanc_verification_seconds": fanc_verification_seconds,
        },
    }
