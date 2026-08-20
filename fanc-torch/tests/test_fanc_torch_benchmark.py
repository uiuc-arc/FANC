from copy import deepcopy
import unittest

import torch

from fanc_torch.benchmark import (
    build_experiment,
    load_dataset,
    specification_bounds,
    specification_descriptors,
    timing_summary,
    validate_experiment,
)

class BenchmarkInputsTests(unittest.TestCase):
    def test_cifar_is_converted_from_hwc_and_normalized(self):
        dataset = load_dataset("fcnn7_cifar")

        self.assertEqual(dataset.raw.shape, (100, 3, 32, 32))
        expected = (dataset.raw[0, 0, 0, 0] - 0.4914) / 0.2023
        torch.testing.assert_close(dataset.normalized[0, 0, 0, 0], expected)

    def test_patch_bounds_modify_only_requested_pixels(self):
        image = torch.full((1, 1, 4, 4), 0.5)
        descriptors = [{"row": 1, "column": 2, "height": 2, "width": 2}]

        lower, upper = specification_bounds(image, descriptors, "mnist")

        self.assertEqual(int((lower == 0).sum()), 4)
        self.assertEqual(int((upper == 1).sum()), 4)
        self.assertEqual(int((lower == 0.5).sum()), 12)


class ExperimentDescriptionTests(unittest.TestCase):
    def test_same_seed_produces_identical_experiment(self):
        first = build_experiment("fcnn7", 2, ("patch", "l0_random"), 17, 5)
        second = build_experiment("fcnn7", 2, ("patch", "l0_random"), 17, 5)

        self.assertEqual(first, second)

    def test_out_of_range_image_is_rejected(self):
        experiment = build_experiment("fcnn7", 1, ("patch",), 17, 1)
        modified = deepcopy(experiment)
        modified["images"][0] = 100

        with self.assertRaisesRegex(ValueError, "outside the dataset"):
            validate_experiment(modified)

    def test_expected_specification_counts(self):
        self.assertEqual(len(specification_descriptors("patch", 28, 28, 1, 0)), 729)
        self.assertEqual(
            len(specification_descriptors("l0_random", 28, 28, 1, 0)), 1000
        )
        self.assertEqual(
            len(specification_descriptors("l0_center", 28, 28, 1, 0)), 1140
        )


class TimingSummaryTests(unittest.TestCase):
    def test_repeated_timings_use_sample_standard_deviation(self):
        first = {
            "attacks": {
                "patch": {
                    "timings": {
                        "baseline_verification_seconds": 4,
                        "fanc_verification_seconds": 2,
                    }
                }
            }
        }
        second = deepcopy(first)
        second["attacks"]["patch"]["timings"] = {
            "baseline_verification_seconds": 6,
            "fanc_verification_seconds": 3,
        }

        summary = timing_summary([first, second])

        self.assertAlmostEqual(
            summary["baseline_verification_seconds"]["std"], 2**0.5
        )


if __name__ == "__main__":
    unittest.main()
