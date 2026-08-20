from copy import deepcopy
import unittest

from fanc_torch.benchmark import summarize_results


def _result():
    result = {
        "attacks": {
            "patch": {
                "specifications": 2,
                "baseline_verified": 2,
                "fanc_verified": 2,
                "template_matches": 1,
                "timings": {
                    "baseline_verification_seconds": 4,
                    "fanc_verification_seconds": 2,
                },
            }
        },
    }
    return result


class SummaryTests(unittest.TestCase):
    def test_counts_and_timing_are_aggregated_per_repetition(self):
        summary = summarize_results(
            [_result()],
            {"network": "network"},
            template_generation_seconds=3,
            template_transformation_seconds=1,
            template_validation_seconds=0.5,
        )

        self.assertNotIn("consistent", summary)
        self.assertEqual(summary["repetitions"], 1)
        self.assertEqual(summary["configuration"], {"network": "network"})
        self.assertEqual(summary["attacks"]["patch"]["template_matches"], 1)
        self.assertEqual(summary["timings"]["template_generation_seconds"], 3)
        self.assertEqual(
            summary["timings"]["template_transformation_seconds"], 1
        )
        self.assertEqual(summary["timings"]["template_validation_seconds"], 0.5)
        self.assertEqual(
            summary["timings"]["baseline_verification_seconds"]["mean"], 4
        )
        self.assertEqual(summary["timings"]["fanc_verification_seconds"]["mean"], 2)
        self.assertEqual(summary["environment"]["device"], "cpu")
        self.assertEqual(summary["environment"]["dtype"], "float32")
        self.assertIn("code_revision", summary["environment"])
        self.assertIn("tracked_changes", summary["environment"])

    def test_inconsistent_counts_are_rejected(self):
        first = _result()
        second = deepcopy(first)
        second["attacks"]["patch"]["fanc_verified"] = 1

        with self.assertRaisesRegex(ValueError, "different verification counts"):
            summarize_results([first, second], {"network": "network"})


if __name__ == "__main__":
    unittest.main()
