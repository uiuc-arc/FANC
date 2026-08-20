import unittest
from unittest.mock import patch

import torch

from fanc_torch.benchmark import build_experiment
from fanc_torch.experiment import (
    prepare_source,
    prepare_target,
    run_experiment,
    verification_configuration,
)
from fanc_torch.templates import generate_templates, template_is_valid


class ExperimentTests(unittest.TestCase):
    def test_prepared_source_is_reused_across_targets_and_runs(self):
        experiment = build_experiment("fcnn7", 1, ("patch",), 17, 1)

        threads = torch.get_num_threads()
        progress = []
        with patch(
            "fanc_torch.experiment.generate_templates", wraps=generate_templates
        ) as generate:
            source = prepare_source(experiment, threads=threads)
        with patch(
            "fanc_torch.algorithm.template_is_valid", wraps=template_is_valid
        ) as validate:
            target = prepare_target(source, "quant8", transfer_radius=0)
            second_target = prepare_target(source, "float16", transfer_radius=0)
            first = run_experiment(
                target,
                batch_size=1,
                progress=progress.append,
            )
            second = run_experiment(target, batch_size=1, fanc_first=True)

        self.assertEqual(generate.call_count, 1)
        self.assertIs(second_target.source, source)
        self.assertEqual(validate.call_count, 2 * len(source.images[0].templates))
        for key in (
            "specifications",
            "baseline_verified",
            "fanc_verified",
            "template_matches",
        ):
            self.assertEqual(
                first["attacks"]["patch"][key],
                second["attacks"]["patch"][key],
            )
        self.assertEqual(first["measurement_order"], "baseline_first")
        self.assertEqual(second["measurement_order"], "fanc_first")
        attack = first["attacks"]["patch"]
        self.assertEqual(attack["specifications"], 1)
        self.assertIn("baseline_verified", attack)
        self.assertIn("template_matches", attack)
        configuration = verification_configuration(target, batch_size=1)
        self.assertEqual(configuration["transfer_radius"], 0)
        self.assertEqual(configuration["batch_size"], 1)
        self.assertNotIn("configuration", first)
        self.assertEqual(len(progress), 2)
        self.assertEqual(progress[0].processed, 0)
        final_progress = progress[-1]
        self.assertEqual(final_progress.image, 1)
        self.assertEqual(final_progress.images, 1)
        self.assertEqual(final_progress.specification, "patch")
        self.assertEqual(final_progress.processed, 1)
        self.assertEqual(final_progress.total, 1)
        self.assertEqual(
            final_progress.baseline_verified, attack["baseline_verified"]
        )
        self.assertEqual(final_progress.fanc_verified, attack["fanc_verified"])
        self.assertEqual(
            final_progress.template_matches, attack["template_matches"]
        )

    def test_cifar_experiment_runs_in_normalized_coordinates(self):
        experiment = build_experiment("fcnn7_cifar", 1, ("patch",), 17, 1)
        threads = torch.get_num_threads()

        source = prepare_source(experiment, threads=threads)
        target = prepare_target(source, "quant8", transfer_radius=0)
        result = run_experiment(target, batch_size=1)

        self.assertEqual(
            verification_configuration(target, batch_size=1)["network"],
            "fcnn7_cifar",
        )
        self.assertEqual(result["attacks"]["patch"]["specifications"], 1)


if __name__ == "__main__":
    unittest.main()
