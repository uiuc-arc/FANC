import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Optional, Sequence

import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from fanc_torch.benchmark import (
    ATTACKS,
    NETWORK_CONFIGS,
    build_experiment,
    summarize_results,
)
from fanc_torch.experiment import (
    VerificationProgress,
    prepare_source,
    prepare_target,
    run_experiment,
    verification_configuration,
)
from fanc_torch.models import TARGET_VARIANTS
from fanc_torch.templates import template_radius_candidates


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a deterministic FANC-Torch configuration.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="result directory (default: output)",
    )
    parser.add_argument(
        "--network",
        choices=sorted(NETWORK_CONFIGS),
        default="fcnn7",
        help="bundled original network (default: fcnn7)",
    )
    parser.add_argument(
        "--target-variant",
        choices=TARGET_VARIANTS,
        default="quant8",
        help="bundled target network variant (default: quant8)",
    )
    parser.add_argument(
        "--attack",
        action="append",
        choices=ATTACKS,
        help="perturbation family; repeat as needed (default: all)",
    )
    parser.add_argument(
        "--images",
        type=_positive_int,
        default=2,
        help="images correctly classified by the original network (default: 2)",
    )
    parser.add_argument(
        "--spec-limit",
        type=_positive_int,
        help="maximum specifications per attack and image (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=2022, help="selection seed (default: 2022)"
    )
    parser.add_argument(
        "--transfer-radius",
        type=_nonnegative_float,
        default=0.0,
        help="transfer radius around the target network activation (default: 0)",
    )
    parser.add_argument(
        "--batch-size",
        type=_positive_int,
        default=8,
        help="specifications verified together (default: 8)",
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=1,
        help="PyTorch CPU threads (default: 1)",
    )
    parser.add_argument(
        "--repetitions",
        type=_positive_int,
        default=1,
        help="runs using the same experiment (default: 1)",
    )
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict:
    attacks = tuple(args.attack) if args.attack else ATTACKS
    _prepare_output(args.output)
    _print_parameters(args, attacks)
    print("experiment setup: started", flush=True)
    experiment = build_experiment(
        args.network,
        args.images,
        attacks,
        args.seed,
        specification_limit=args.spec_limit,
    )
    _write_json(args.output / "experiment.json", experiment)
    selected_images = ", ".join(str(index) for index in experiment["images"])
    print(f"experiment setup: complete (images: {selected_images})", flush=True)

    torch.set_num_threads(args.threads)
    print("template generation: started", flush=True)
    source = prepare_source(
        experiment,
        threads=args.threads,
        progress=lambda current, total: print(
            f"template generation: image {current}/{total}", flush=True
        ),
    )
    generated_templates = sum(len(image.templates) for image in source.images)
    print(
        f"template generation: complete ({generated_templates} templates)",
        flush=True,
    )
    print("template transfer and validation: started", flush=True)
    target = prepare_target(
        source,
        target_variant=args.target_variant,
        transfer_radius=args.transfer_radius,
        progress=lambda current, total: print(
            f"template transfer and validation: image {current}/{total}",
            flush=True,
        ),
    )
    candidate_templates = sum(
        image.template_verifier.candidate_template_count for image in target.images
    )
    valid_templates = sum(
        image.template_verifier.valid_template_count for image in target.images
    )
    print(
        "template transfer and validation: complete "
        f"({valid_templates}/{candidate_templates} valid templates)",
        flush=True,
    )
    results = []
    for repetition in range(1, args.repetitions + 1):
        print(
            f"verification repetition {repetition}/{args.repetitions}: started",
            flush=True,
        )
        progress = _VerificationProgressDisplay(repetition, args.repetitions)
        try:
            result = run_experiment(
                target,
                batch_size=args.batch_size,
                progress=progress,
                fanc_first=repetition % 2 == 0,
            )
        finally:
            progress.close()
        results.append(result)
        _write_json(args.output / f"result-{repetition}.json", result)
        print(
            f"verification repetition {repetition}/{args.repetitions}: complete",
            flush=True,
        )

    summary = summarize_results(
        results,
        verification_configuration(target, args.batch_size),
        template_generation_seconds=source.template_generation_seconds,
        template_transformation_seconds=target.template_transformation_seconds,
        template_validation_seconds=target.template_validation_seconds,
    )
    _write_json(args.output / "summary.json", summary)
    return summary


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return parsed


def _print_parameters(args: argparse.Namespace, attacks: Sequence[str]) -> None:
    config = NETWORK_CONFIGS[args.network]
    template_radii = ", ".join(
        f"{radius:g}" for radius in template_radius_candidates()
    )
    specification_limit = args.spec_limit if args.spec_limit is not None else "all"
    values = (
        ("network", args.network),
        ("target variant", args.target_variant),
        ("attacks", ", ".join(attacks)),
        ("images", args.images),
        ("specification limit", specification_limit),
        ("seed", args.seed),
        ("template patch size", config.patch_size),
        ("template layer", config.template_layer),
        ("template radius candidates", template_radii),
        ("transfer radius", args.transfer_radius),
        ("verification batch size", args.batch_size),
        ("threads", args.threads),
        ("repetitions", args.repetitions),
        ("output", args.output),
    )
    print("parameters:")
    for name, value in values:
        print(f"  {name}: {value}")


def _write_json(path: Path, value: dict) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _prepare_output(path: Path) -> None:
    if path.is_symlink():
        raise ValueError("output path must be a directory")
    if not path.exists():
        path.mkdir(parents=True)
        return
    if not path.is_dir():
        raise ValueError("output path must be a directory")

    entries = list(path.iterdir())
    if any(
        not entry.is_file() or not _is_result_file(entry.name) for entry in entries
    ):
        raise ValueError("output directory contains files not created by this runner")
    for entry in entries:
        entry.unlink()


def _is_result_file(name: str) -> bool:
    if name in {"experiment.json", "summary.json"}:
        return True
    prefix = "result-"
    suffix = ".json"
    if not name.startswith(prefix) or not name.endswith(suffix):
        return False
    repetition = name[len(prefix) : -len(suffix)]
    return repetition.isdigit() and int(repetition) > 0


class _VerificationProgressDisplay:
    def __init__(self, repetition: int, repetitions: int):
        self.repetition = repetition
        self.repetitions = repetitions
        self.key = None
        self.bar = None

    def __call__(self, progress: VerificationProgress) -> None:
        key = (progress.image, progress.specification)
        if key != self.key:
            self.close()
            self.key = key
            self.bar = tqdm(
                total=progress.total,
                desc=(
                    f"repetition {self.repetition}/{self.repetitions} "
                    f"image {progress.image}/{progress.images} "
                    f"{progress.specification}"
                ),
                unit="spec",
            )
        self.bar.update(progress.processed - self.bar.n)
        self.bar.set_postfix(
            {
                "baseline": progress.baseline_verified,
                "FANC": progress.fanc_verified,
                "template_matches": progress.template_matches,
            },
            refresh=True,
        )

    def close(self) -> None:
        if self.bar is not None:
            self.bar.close()
            self.bar = None


def _print_summary(summary: dict, output: Path) -> None:
    print(f"repetitions: {summary['repetitions']}")
    for attack, metrics in summary["attacks"].items():
        print(
            f"{attack}: baseline "
            f"{metrics['baseline_verified']}/{metrics['specifications']}, "
            f"FANC {metrics['fanc_verified']}/{metrics['specifications']}, "
            "template matches "
            f"{metrics['template_matches']}/{metrics['specifications']}"
        )
    timings = summary["timings"]
    print(f"template generation: {timings['template_generation_seconds']:.6f}s")
    print(
        "template transformation: "
        f"{timings['template_transformation_seconds']:.6f}s"
    )
    print(f"template validation: {timings['template_validation_seconds']:.6f}s")
    for label, key in (
        ("baseline verification", "baseline_verification_seconds"),
        ("FANC verification", "fanc_verification_seconds"),
    ):
        values = timings[key]
        print(f"{label}: {values['mean']:.6f}s ± {values['std']:.6f}s")
    print(f"results: {output}")


def main() -> None:
    args = parse_args()
    summary = run(args)
    _print_summary(summary, args.output)


if __name__ == "__main__":
    main()
