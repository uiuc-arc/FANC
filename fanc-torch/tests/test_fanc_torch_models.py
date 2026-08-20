import unittest

import torch

from fanc_torch import NETWORK_NAMES, TARGET_VARIANTS, load_bundled_network


class BundledNetworkTests(unittest.TestCase):
    def test_all_bundled_networks_load_and_run(self):
        inputs = {
            "fcnn7": torch.zeros(1, 1, 28, 28),
            "fconv4": torch.zeros(1, 1, 28, 28),
            "fcnn7_cifar": torch.zeros(1, 3, 32, 32),
            "fconv4_cifar": torch.zeros(1, 3, 32, 32),
        }
        for name in NETWORK_NAMES:
            for target_variant in (None, *TARGET_VARIANTS):
                with self.subTest(name=name, target_variant=target_variant):
                    network = load_bundled_network(name, target_variant)
                    self.assertEqual(network(inputs[name]).shape, (1, 10))

if __name__ == "__main__":
    unittest.main()
