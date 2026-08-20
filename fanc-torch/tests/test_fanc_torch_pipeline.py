import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from fanc_torch import (
    SequentialNetwork,
    ZonotopeVerifier,
    generate_templates,
    template_is_valid,
    transfer_templates,
    verify_with_templates,
)


def make_network(input_weight: float) -> SequentialNetwork:
    hidden = nn.Linear(1, 1)
    output = nn.Linear(1, 2)
    with torch.no_grad():
        hidden.weight.fill_(input_weight)
        hidden.bias.zero_()
        output.weight.copy_(torch.tensor([[1.0], [0.0]]))
        output.bias.copy_(torch.tensor([0.0, 0.2]))
    return SequentialNetwork([nn.Flatten(), hidden, nn.ReLU(), output])


class FancTorchPipelineTests(unittest.TestCase):
    def test_generate_transfer_and_verify_pipeline(self):
        source = ZonotopeVerifier(make_network(1.0))
        target = ZonotopeVerifier(make_network(0.9))
        image = torch.tensor([[[[0.8]]]])
        templates = generate_templates(
            source,
            image,
            label=0,
            layer=2,
            patch_size=1,
            template_radii=(0.1,),
        )

        transferred = transfer_templates(
            source,
            target,
            templates,
            image,
            transfer_radius=0.05,
        )
        with patch(
            "fanc_torch.algorithm.template_is_valid", wraps=template_is_valid
        ) as validate:
            verified = verify_with_templates(
                target,
                torch.tensor([[[[0.75]]]]),
                torch.tensor([[[[0.80]]]]),
                label=0,
                templates=transferred,
                layer=2,
            )

        self.assertEqual(len(templates), 1)
        self.assertEqual(len(transferred), 1)
        self.assertEqual(validate.call_count, 1)
        torch.testing.assert_close(verified, torch.tensor([True]))


if __name__ == "__main__":
    unittest.main()
