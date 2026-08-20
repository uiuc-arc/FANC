from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fanc_torch.experiment import VerificationProgress
from scripts.run_suite import (
    ATTACKS,
    _VerificationProgressDisplay,
    _prepare_output,
    _print_parameters,
    _print_summary,
    _write_json,
    parse_args,
)


class RunSuiteArgumentTests(unittest.TestCase):
    def test_defaults_process_all_specifications(self):
        args = parse_args([])

        self.assertEqual(args.output, Path("output"))
        self.assertEqual(args.network, "fcnn7")
        self.assertEqual(args.target_variant, "quant8")
        self.assertIsNone(args.attack)
        self.assertEqual(args.images, 2)
        self.assertIsNone(args.spec_limit)
        self.assertEqual(args.transfer_radius, 0)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.threads, 1)
        self.assertEqual(args.repetitions, 1)

class RunSuiteOutputTests(unittest.TestCase):
    def test_progress_distinguishes_baseline_and_fanc(self):
        with patch("scripts.run_suite.tqdm") as create_bar:
            bar = create_bar.return_value
            bar.n = 0
            display = _VerificationProgressDisplay(1, 2)

            display(
                VerificationProgress(
                    image=1,
                    images=1,
                    specification="patch",
                    processed=2,
                    total=3,
                    baseline_verified=1,
                    fanc_verified=2,
                    template_matches=1,
                )
            )

            create_bar.assert_called_once_with(
                total=3,
                desc="repetition 1/2 image 1/1 patch",
                unit="spec",
            )
            bar.set_postfix.assert_called_once_with(
                {"baseline": 1, "FANC": 2, "template_matches": 1},
                refresh=True,
            )
            display.close()

    def test_parameters_print_all_as_default_specification_limit(self):
        args = parse_args([])

        output = StringIO()
        with redirect_stdout(output):
            _print_parameters(args, ATTACKS)

        self.assertIn("  specification limit: all\n", output.getvalue())

    def test_parameters_print_effective_configuration(self):
        args = parse_args(
            [
                "--output",
                "output/custom",
                "--network",
                "fcnn7",
                "--target-variant",
                "float16",
                "--attack",
                "patch",
                "--images",
                "3",
                "--spec-limit",
                "12",
                "--seed",
                "7",
                "--transfer-radius",
                "0.001",
                "--batch-size",
                "4",
                "--threads",
                "2",
                "--repetitions",
                "3",
            ]
        )

        output = StringIO()
        with redirect_stdout(output):
            _print_parameters(args, tuple(args.attack))

        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "parameters:",
                "  network: fcnn7",
                "  target variant: float16",
                "  attacks: patch",
                "  images: 3",
                "  specification limit: 12",
                "  seed: 7",
                "  template patch size: 7",
                "  template layer: 4",
                "  template radius candidates: "
                "1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, "
                "0.0078125, 0.00390625, 0",
                "  transfer radius: 0.001",
                "  verification batch size: 4",
                "  threads: 2",
                "  repetitions: 3",
                "  output: output/custom",
            ],
        )

    def test_summary_prints_counts_and_phase_timings_without_speedup(self):
        summary = {
            "repetitions": 1,
            "attacks": {
                "patch": {
                    "specifications": 3,
                    "baseline_verified": 2,
                    "fanc_verified": 3,
                    "template_matches": 1,
                }
            },
            "timings": {
                "template_generation_seconds": 1.0,
                "template_transformation_seconds": 0.5,
                "template_validation_seconds": 0.25,
                "baseline_verification_seconds": {"mean": 2.0, "std": 0.2},
                "fanc_verification_seconds": {"mean": 1.0, "std": 0.1},
            },
        }

        output = StringIO()
        with redirect_stdout(output):
            _print_summary(summary, Path("output"))

        text = output.getvalue()
        self.assertIn("repetitions: 1\n", text)
        self.assertNotIn("consistent", text)
        self.assertIn(
            "patch: baseline 2/3, FANC 3/3, template matches 1/3",
            text,
        )
        self.assertIn("template generation: 1.000000s", text)
        self.assertIn("template transformation: 0.500000s", text)
        self.assertIn("template validation: 0.250000s", text)
        self.assertIn("baseline verification: 2.000000s ± 0.200000s", text)
        self.assertIn("FANC verification: 1.000000s ± 0.100000s", text)
        self.assertNotIn("speedup", text)

    def test_existing_runner_results_are_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            _prepare_output(output)
            for name in (
                "experiment.json",
                "result-1.json",
                "result-3.json",
                "summary.json",
            ):
                _write_json(output / name, {})

            _prepare_output(output)

            self.assertEqual(list(output.iterdir()), [])

    def test_unrelated_output_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result = output / "result-1.json"
            unrelated = output / "notes.txt"
            _write_json(result, {})
            unrelated.write_text("keep me")

            with self.assertRaisesRegex(ValueError, "not created by this runner"):
                _prepare_output(output)

            self.assertEqual(unrelated.read_text(), "keep me")
            self.assertEqual(json.loads(result.read_text()), {})

    def test_json_is_written_without_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            path = output / "result.json"

            _write_json(path, {"value": 1})

            self.assertEqual(json.loads(path.read_text()), {"value": 1})
            self.assertEqual(list(output.iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
