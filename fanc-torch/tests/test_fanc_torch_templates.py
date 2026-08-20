import unittest

import torch
import torch.nn as nn

from fanc_torch import (
    SequentialNetwork,
    ZonotopeVerifier,
    generate_templates,
    template_radius_candidates,
    template_is_valid,
)


class TemplateGenerationTests(unittest.TestCase):
    def test_candidates_descend_to_zero(self):
        self.assertEqual(
            template_radius_candidates(0.2), (1.0, 0.5, 0.25, 0.0)
        )

    def test_generation_creates_one_template_per_patch(self):
        flatten = nn.Flatten()
        linear = nn.Linear(16, 2)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([1.0, 0.0]))
        verifier = ZonotopeVerifier(SequentialNetwork([flatten, linear]))
        image = torch.full((1, 1, 4, 4), 0.5)

        templates = generate_templates(
            verifier, image, label=0, layer=0, patch_size=2, template_radii=(1.0,)
        )

        self.assertEqual(len(templates), 4)
        self.assertEqual(
            [template.patch for template in templates.templates],
            [(0, 0, 2), (0, 2, 2), (2, 0, 2), (2, 2, 2)],
        )
        self.assertTrue(
            all(template.template_radius == 1.0 for template in templates.templates)
        )
        self.assertTrue(
            all(template_is_valid(verifier, template, 0) for template in templates.templates)
        )

    def test_largest_individually_verified_candidate_is_selected(self):
        flatten = nn.Flatten()
        linear = nn.Linear(1, 2)
        with torch.no_grad():
            linear.weight.copy_(torch.tensor([[1.0], [0.0]]))
            linear.bias.copy_(torch.tensor([0.0, 0.6]))
        verifier = ZonotopeVerifier(SequentialNetwork([flatten, linear]))
        image = torch.tensor([[[[0.9]]]])

        templates = generate_templates(
            verifier,
            image,
            label=0,
            layer=0,
            patch_size=1,
            template_radii=(0.1, 0.5, 0.25),
        )

        self.assertEqual(len(templates), 1)
        self.assertEqual(templates.templates[0].template_radius, 0.25)

    def test_input_transform_is_applied_after_raw_pixel_clamping(self):
        flatten = nn.Flatten()
        linear = nn.Linear(1, 2)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([1.0, 0.0]))
        verifier = ZonotopeVerifier(SequentialNetwork([flatten, linear]))

        templates = generate_templates(
            verifier,
            torch.tensor([[[[0.25]]]]),
            label=0,
            layer=0,
            patch_size=1,
            template_radii=(0.25,),
            input_transform=lambda value: value * 2 - 1,
        )

        self.assertEqual(len(templates), 1)
        torch.testing.assert_close(
            templates.templates[0].region.lower, torch.tensor([[-1.0]])
        )
        torch.testing.assert_close(
            templates.templates[0].region.upper, torch.tensor([[0.0]])
        )

    def test_misclassified_point_produces_no_template(self):
        flatten = nn.Flatten()
        linear = nn.Linear(1, 2)
        with torch.no_grad():
            linear.weight.zero_()
            linear.bias.copy_(torch.tensor([0.0, 1.0]))
        verifier = ZonotopeVerifier(SequentialNetwork([flatten, linear]))

        templates = generate_templates(
            verifier,
            torch.tensor([[[[0.5]]]]),
            label=0,
            layer=0,
            patch_size=1,
            template_radii=(1.0,),
        )

        self.assertEqual(len(templates), 0)

if __name__ == "__main__":
    unittest.main()
