import unittest

import torch

from fanc_torch import Box, Zonotope


class BoundsTests(unittest.TestCase):
    def test_contains_each_compares_all_templates_and_regions(self):
        templates = Box(
            torch.tensor([[0.0, 0.0], [0.5, 0.5]]),
            torch.tensor([[2.0, 2.0], [1.5, 1.5]]),
        )
        regions = Box(
            torch.tensor([[0.75, 0.75], [-1.0, -1.0]]),
            torch.tensor([[1.25, 1.25], [0.0, 0.0]]),
        )

        matched = templates.contains_each(regions)

        torch.testing.assert_close(
            matched, torch.tensor([[True, False], [True, False]])
        )

    def test_bounds_round_trip_through_zonotope(self):
        lower = torch.tensor([[[-1.0, 0.0], [2.0, 3.0]]])
        upper = torch.tensor([[[1.0, 4.0], [4.0, 3.0]]])

        zonotope = Zonotope.from_bounds(lower, upper)
        box = zonotope.to_box()

        torch.testing.assert_close(box.lower, lower)
        torch.testing.assert_close(box.upper, upper)
        self.assertEqual(zonotope.generators.shape, (1, 3, 2, 2))

    def test_batched_generators_use_union_of_varying_dimensions(self):
        lower = torch.zeros(2, 3)
        upper = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 2.0]])

        zonotope = Zonotope.from_bounds(lower, upper)

        self.assertEqual(zonotope.generators.shape, (2, 2, 3))
        torch.testing.assert_close(zonotope.lower, lower)
        torch.testing.assert_close(zonotope.upper, upper)


if __name__ == "__main__":
    unittest.main()
