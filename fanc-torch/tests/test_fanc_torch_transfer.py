import unittest

import torch
import torch.nn as nn

from fanc_torch import (
    Box,
    SequentialNetwork,
    Template,
    TemplateCollection,
    ZonotopeVerifier,
    template_is_valid,
    transfer_templates,
    transform_template,
)


def make_network(first_weight: float, competitor_bias: float) -> SequentialNetwork:
    first = nn.Linear(1, 1)
    output = nn.Linear(1, 2)
    with torch.no_grad():
        first.weight.fill_(first_weight)
        first.bias.zero_()
        output.weight.copy_(torch.tensor([[1.0], [0.0]]))
        output.bias.copy_(torch.tensor([0.0, competitor_bias]))
    return SequentialNetwork([first, nn.ReLU(), output])


class TemplateTransformationTests(unittest.TestCase):
    def test_transformation_is_hull_with_target_radius_box(self):
        template = Template(
            1,
            Box(torch.tensor([[0.8]]), torch.tensor([[1.0]])),
            (0, 0, 1),
            0.25,
        )

        transformed = transform_template(template, torch.tensor([[0.5]]), 0.1)

        torch.testing.assert_close(transformed.region.lower, torch.tensor([[0.4]]))
        torch.testing.assert_close(transformed.region.upper, torch.tensor([[1.0]]))
        self.assertEqual(transformed.layer, template.layer)
        self.assertEqual(transformed.patch, template.patch)
        self.assertEqual(transformed.template_radius, template.template_radius)

class TemplateTransferTests(unittest.TestCase):
    def setUp(self):
        self.source = ZonotopeVerifier(make_network(1.0, 0.7))
        self.template = Template(
            1,
            Box(torch.tensor([[0.8]]), torch.tensor([[1.0]])),
            (0, 0, 1),
            0.25,
        )
        self.templates = TemplateCollection([self.template])
        self.image = torch.tensor([[1.0]])

    def test_valid_transformed_template_is_available_for_target_validation(self):
        target = ZonotopeVerifier(make_network(0.9, 0.2))

        transferred = transfer_templates(
            self.source,
            target,
            self.templates,
            self.image,
            transfer_radius=0.1,
        )

        self.assertEqual(len(transferred), 1)
        self.assertTrue(template_is_valid(target, transferred.templates[0], 0))
        torch.testing.assert_close(
            transferred.templates[0].region.lower, torch.tensor([[0.8]])
        )

    def test_invalid_transformed_template_is_left_for_target_validation(self):
        target = ZonotopeVerifier(make_network(0.5, 0.7))

        transferred = transfer_templates(
            self.source,
            target,
            self.templates,
            self.image,
            transfer_radius=0.1,
        )

        self.assertEqual(len(transferred), 1)

    def test_architecture_mismatch_is_rejected(self):
        target = ZonotopeVerifier(
            SequentialNetwork([nn.Linear(1, 2), nn.ReLU(), nn.Linear(2, 2)])
        )

        with self.assertRaisesRegex(ValueError, "same architecture"):
            transfer_templates(
                self.source,
                target,
                self.templates,
                self.image,
                transfer_radius=0.0,
            )

if __name__ == "__main__":
    unittest.main()
