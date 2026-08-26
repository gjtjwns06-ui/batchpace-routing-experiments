from __future__ import annotations

import unittest

import numpy as np

from src.benchmark_suite_environments import generate_synthetic_paths, synthetic_specs
from src.benchmark_suite_policies import (
    build_batch_dp,
    build_sequential_dp,
    offline_oracle,
    run_batchpace,
    run_equal_split,
    run_fixed_threshold,
    run_robust_batch,
    run_sequential,
)


class BenchmarkSuiteTest(unittest.TestCase):
    def test_exactly_40_distinct_synthetic_environments(self) -> None:
        specs = synthetic_specs()
        self.assertEqual(len(specs), 40)
        self.assertEqual(len({x.environment_id for x in specs}), 40)

    def test_shift_and_shape(self) -> None:
        sample = generate_synthetic_paths(synthetic_specs()[0], seed=7, paths=3)
        for key in ("gain", "score", "uncertainty", "phase", "latent"):
            self.assertEqual(sample[key].shape, (3, 24, 20))
        self.assertTrue(np.all(sample["phase"][:, :12] == 0))
        self.assertTrue(np.all(sample["phase"][:, 12:] == 1))

    def test_all_deterministic_policies_respect_budget(self) -> None:
        rng = np.random.default_rng(5)
        calibration = rng.random((20, 4, 5))
        score = rng.random(20)
        gain = rng.uniform(-1, 1, 20)
        budget = 7
        batch_dp = build_batch_dp(calibration, budget)
        sequential_dp = build_sequential_dp(calibration, budget)
        outcomes = [
            run_batchpace(score, gain, 5, budget, batch_dp),
            run_sequential(score, gain, 5, budget, sequential_dp),
            run_fixed_threshold(score, gain, 5, budget, 0.5),
            run_equal_split(score, gain, 5, budget),
            run_robust_batch(score - 0.2, gain, 5, budget),
            offline_oracle(gain, budget, 5),
        ]
        for outcome in outcomes:
            self.assertLessEqual(outcome.budget_used, budget)
            self.assertEqual(len(outcome.allocations), 4)
            self.assertAlmostEqual(outcome.gain, gain[outcome.selected].sum())


if __name__ == "__main__":
    unittest.main()

