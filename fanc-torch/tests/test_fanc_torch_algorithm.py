import unittest

import torch
import torch.nn as nn

from fanc_torch import (
    Box,
    SequentialNetwork,
    Template,
    TemplateCollection,
    TemplateVerifier,
    ZonotopeVerifier,
    verify_with_templates,
)


def make_verifier() -> ZonotopeVerifier:
    first = nn.Linear(1, 1)
    output = nn.Linear(1, 2)
    with torch.no_grad():
        first.weight.fill_(1.0)
        first.bias.zero_()
        output.weight.copy_(torch.tensor([[1.0], [0.0]]))
        output.bias.copy_(torch.tensor([0.0, 0.6]))
    return ZonotopeVerifier(SequentialNetwork([first, nn.ReLU(), output]))


def make_template(lower: float, upper: float, layer: int = 1) -> Template:
    return Template(
        layer,
        Box(torch.tensor([[lower]]), torch.tensor([[upper]])),
        (0, 0, 1),
        0.25,
    )


class TemplateVerificationTests(unittest.TestCase):
    def setUp(self):
        self.verifier = make_verifier()

    def test_invalid_templates_are_filtered_before_containment(self):
        templates = TemplateCollection([make_template(0.0, 1.0)])
        lower = torch.tensor([[0.4]])
        upper = torch.tensor([[0.5]])

        verified = verify_with_templates(
            self.verifier, lower, upper, 0, templates, layer=1
        )

        torch.testing.assert_close(verified, torch.tensor([False]))

    def test_misses_have_exactly_the_baseline_result(self):
        lower = torch.tensor([[0.8], [0.4], [-0.2]])
        upper = torch.tensor([[0.9], [0.5], [0.7]])
        baseline = self.verifier.verify_bounds(lower, upper, 0)

        result = verify_with_templates(
            self.verifier,
            lower,
            upper,
            0,
            TemplateCollection(),
            layer=1,
        )

        torch.testing.assert_close(result, baseline)

    def test_template_verifier_reports_matches_separately_from_fallbacks(self):
        template_verifier = TemplateVerifier(
            self.verifier,
            TemplateCollection([make_template(0.75, 1.0)]),
            label=0,
            layer=1,
        )

        result = template_verifier.verify(
            torch.tensor([[0.8], [0.65]]),
            torch.tensor([[0.9], [0.7]]),
        )

        torch.testing.assert_close(result.matched, torch.tensor([True, False]))
        torch.testing.assert_close(result.verified, torch.tensor([True, True]))

    def test_template_verifier_ignores_later_collection_changes(self):
        templates = TemplateCollection([make_template(0.75, 1.0)])
        template_verifier = TemplateVerifier(
            self.verifier, templates, label=0, layer=1
        )

        templates.add(make_template(0.0, 1.0))
        result = template_verifier.verify(
            torch.tensor([[0.4]]), torch.tensor([[0.5]])
        )

        self.assertEqual(template_verifier.candidate_template_count, 1)
        self.assertEqual(template_verifier.valid_template_count, 1)
        self.assertFalse(hasattr(template_verifier, "templates"))
        torch.testing.assert_close(result.matched, torch.tensor([False]))
        torch.testing.assert_close(result.verified, torch.tensor([False]))

    def test_template_verifier_copies_validated_bounds(self):
        template = make_template(0.75, 1.0)
        template_verifier = TemplateVerifier(
            self.verifier, TemplateCollection([template]), label=0, layer=1
        )

        template.region.lower.fill_(0.0)
        result = template_verifier.verify(
            torch.tensor([[0.4]]), torch.tensor([[0.5]])
        )

        torch.testing.assert_close(result.matched, torch.tensor([False]))
        torch.testing.assert_close(result.verified, torch.tensor([False]))


if __name__ == "__main__":
    unittest.main()
