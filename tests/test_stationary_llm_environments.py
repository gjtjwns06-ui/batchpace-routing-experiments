from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.benchmark_suite_environments import BATCH_SIZE, HORIZON
from src.stationary_llm_environments import (
    sample_stationary_llm_paths,
    stationary_llm_specs,
)


class StationaryLlmEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        rows = []
        for source_index in range(45):
            for split in ("train", "calibration", "test"):
                for item in range(3 + (source_index % 4)):
                    rows.append({
                        "src": f"stemez-Subject{source_index:02d}",
                        "category": f"category_{source_index % 5}",
                        "split": split,
                        "gain": float(item % 2),
                    })
        self.frame = pd.DataFrame(rows)

    def test_selects_exactly_40_unique_single_subject_specs(self) -> None:
        specs = stationary_llm_specs(self.frame, limit=40)
        self.assertEqual(len(specs), 40)
        self.assertEqual(len({spec.source for spec in specs}), 40)
        self.assertTrue(all(spec.regime == "iid_stationary" for spec in specs))
        self.assertTrue(all("_to_" not in spec.environment_id for spec in specs))

    def test_every_batch_uses_the_same_subject_pool(self) -> None:
        spec = stationary_llm_specs(self.frame, limit=40)[0]
        indices = sample_stationary_llm_paths(
            self.frame, spec, "test", seed=1234, paths=3
        )
        self.assertEqual(indices.shape, (3, HORIZON, BATCH_SIZE))
        sources = self.frame.iloc[indices.reshape(-1)]["src"].unique()
        splits = self.frame.iloc[indices.reshape(-1)]["split"].unique()
        np.testing.assert_array_equal(sources, [spec.source])
        np.testing.assert_array_equal(splits, ["test"])


if __name__ == "__main__":
    unittest.main()

