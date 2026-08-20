import itertools
import unittest

import torch
import torch.nn as nn

from fanc_torch import SequentialNetwork, Zonotope, ZonotopeVerifier


def enumerate_corners(lower, upper):
    corners = []
    for choices in itertools.product((0, 1), repeat=lower.numel()):
        mask = torch.tensor(choices, dtype=torch.bool).reshape_as(lower)
        corners.append(torch.where(mask, upper, lower))
    return torch.stack(corners)


class ZonotopeVerifierTests(unittest.TestCase):
    def test_linear_relu_bounds_contain_all_input_corners(self):
        first = nn.Linear(2, 3)
        second = nn.Linear(3, 2)
        with torch.no_grad():
            first.weight.copy_(torch.tensor([[1.0, -1.0], [0.5, 2.0], [-2.0, 0.5]]))
            first.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
            second.weight.copy_(torch.tensor([[1.0, -0.5, 0.25], [-1.0, 1.0, 0.5]]))
            second.bias.copy_(torch.tensor([0.0, 0.2]))
        network = SequentialNetwork([first, nn.ReLU(), second])
        verifier = ZonotopeVerifier(network)
        lower = torch.tensor([[-0.4, -0.2]])
        upper = torch.tensor([[0.6, 0.8]])

        abstract_output = verifier.propagate_bounds(lower, upper)
        concrete_outputs = network(enumerate_corners(lower[0], upper[0]))

        self.assertTrue((concrete_outputs >= abstract_output.lower[0]).all())
        self.assertTrue((concrete_outputs <= abstract_output.upper[0]).all())

    def test_continuation_matches_single_pass(self):
        torch.manual_seed(0)
        network = SequentialNetwork(
            [nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 2)]
        )
        verifier = ZonotopeVerifier(network)
        region = Zonotope.from_bounds(
            torch.tensor([[-0.2, 0.1]]), torch.tensor([[0.4, 0.7]])
        )

        prefix = verifier.propagate(region, end_layer=2)
        continued = verifier.propagate(prefix, start_layer=2)
        complete = verifier.propagate(region)

        torch.testing.assert_close(continued.center, complete.center)
        torch.testing.assert_close(continued.generators, complete.generators)

    def test_convolution_point_matches_concrete_network(self):
        convolution = nn.Conv2d(1, 2, kernel_size=2)
        network = SequentialNetwork([convolution, nn.ReLU(), nn.Flatten(), nn.Linear(8, 2)])
        verifier = ZonotopeVerifier(network)
        inputs = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3) / 10

        output = verifier.propagate_bounds(inputs, inputs)

        torch.testing.assert_close(output.center, network(inputs))

    def test_verify_proves_a_dominant_label(self):
        network = SequentialNetwork([nn.Linear(2, 2)])
        with torch.no_grad():
            network.layers[0].weight.copy_(torch.tensor([[1.0, 1.0], [-1.0, -1.0]]))
            network.layers[0].bias.copy_(torch.tensor([2.0, -2.0]))
        verifier = ZonotopeVerifier(network)
        lower = torch.tensor([[0.0, 0.0], [-3.0, -3.0]])
        upper = torch.tensor([[0.1, 0.1], [3.0, 3.0]])

        verified = verifier.verify_bounds(lower, upper, torch.tensor([0, 0]))

        torch.testing.assert_close(verified, torch.tensor([True, False]))

    def test_relu_allocates_errors_only_for_unstable_neurons(self):
        verifier = ZonotopeVerifier(SequentialNetwork([nn.ReLU()]))
        lower = torch.tensor([[-1.0, 1.0, -2.0]])
        upper = torch.tensor([[1.0, 1.0, -1.0]])

        output = verifier.propagate_bounds(lower, upper)

        self.assertEqual(output.generators.shape[1], 2)
        self.assertTrue((output.lower <= torch.tensor([[0.0, 1.0, 0.0]])).all())
        self.assertTrue((output.upper >= torch.tensor([[1.0, 1.0, 0.0]])).all())


if __name__ == "__main__":
    unittest.main()
