from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON
from src.stationary_llm_broad_environments import (
    broad_stationary_llm_specs,
    sample_broad_stationary_paths,
)


class BroadStationaryLlmEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for category_index in range(14):
            for split in ("train", "calibration", "test"):
                for item in range(4 + category_index):
                    rows.append({
                        "category": f"category_{category_index:02d}",
                        "split": split,
                    })
        self.frame = pd.DataFrame(rows)

    def test_one_environment_per_broad_category(self) -> None:
        specs = broad_stationary_llm_specs(self.frame)
        self.assertEqual(len(specs), 14)
        self.assertEqual(len({spec.category for spec in specs}), 14)
        self.assertTrue(all(spec.regime == "iid_stationary" for spec in specs))
        self.assertTrue(all("_to_" not in spec.environment_id for spec in specs))

    def test_all_batches_use_one_unchanging_category(self) -> None:
        spec = broad_stationary_llm_specs(self.frame)[5]
        indices = sample_broad_stationary_paths(
            self.frame, spec, "test", seed=2026, paths=2
        )
        self.assertEqual(indices.shape, (2, HORIZON, BATCH_SIZE))
        categories = self.frame.iloc[indices.reshape(-1)]["category"].unique()
        np.testing.assert_array_equal(categories, [spec.category])


if __name__ == "__main__":
    unittest.main()

