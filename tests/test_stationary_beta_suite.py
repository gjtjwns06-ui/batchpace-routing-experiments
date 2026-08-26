import unittest

import numpy as np

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON
from src.stationary_beta_environments import (
    generate_stationary_beta_paths,
    stationary_beta_specs,
)


class StationaryBetaSuiteTest(unittest.TestCase):
    def test_exactly_40_unique_parameter_pairs(self) -> None:
        specs = stationary_beta_specs()
        self.assertEqual(len(specs), 40)
        self.assertEqual(len({spec.environment_id for spec in specs}), 40)
        self.assertEqual(len({(spec.alpha, spec.beta) for spec in specs}), 40)

    def test_paths_are_stationary_single_beta(self) -> None:
        spec = stationary_beta_specs()[10]
        data = generate_stationary_beta_paths(spec, seed=17, paths=100)
        expected_shape = (100, HORIZON, BATCH_SIZE)
        self.assertEqual(data["gain"].shape, expected_shape)
        self.assertTrue(np.all((data["gain"] >= 0) & (data["gain"] <= 1)))
        self.assertTrue(np.all(data["phase"] == 0))
        self.assertTrue(np.all(data["latent"] == 0))
        first_half = data["gain"][:, : HORIZON // 2].mean()
        second_half = data["gain"][:, HORIZON // 2 :].mean()
        self.assertLess(abs(first_half - second_half), 0.02)
        self.assertLess(abs(data["gain"].mean() - spec.theoretical_mean), 0.02)


if __name__ == "__main__":
    unittest.main()
