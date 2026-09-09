from __future__ import annotations

import math
import unittest

import numpy as np

from lapo_value_model.procvlm_metrics import (
    _confusion,
    _occupied_bins,
    epr_at_tau,
    mcc_from_confusion,
    progress_scores,
)


def brute_epr(scores: np.ndarray, tau: float = 0.5) -> dict[str, float | int]:
    values = np.sort(np.clip(np.asarray(scores, dtype=np.float64), 0.0, 1.0))
    feasible = [
        k for k in range(1, int(math.floor(len(values) / tau)) + 1)
        if _occupied_bins(values, k) / k >= tau
    ]
    k = max(feasible)
    return {"k": k, "epr": math.log2(k)}


class EPRTests(unittest.TestCase):
    def test_segmented_search_matches_brute_force(self) -> None:
        rng = np.random.default_rng(7)
        cases = [rng.random(size) for size in range(1, 40)]
        cases.extend(
            [
                np.asarray([0.0, 0.0, 1.0, 1.0]),
                np.asarray([0.49, 0.51]),
                np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
            ]
        )
        for values in cases:
            actual = epr_at_tau(values, block_size=7)
            expected = brute_epr(values)
            self.assertEqual(actual["k"], expected["k"])
            self.assertAlmostEqual(actual["epr"], expected["epr"])

    def test_resolution_ordering(self) -> None:
        constant = np.full(100, 0.4)
        discrete = np.tile(np.asarray([0.1, 0.3, 0.6, 0.9]), 25)
        continuous = np.linspace(0.0, 1.0, 100)
        self.assertEqual(epr_at_tau(constant)["k"], 2)
        self.assertGreater(epr_at_tau(discrete)["epr"], epr_at_tau(constant)["epr"])
        self.assertGreater(epr_at_tau(continuous)["epr"], epr_at_tau(discrete)["epr"])

    def test_value_to_progress_transform(self) -> None:
        values = np.asarray([-2.0, -1.0, -0.5, 0.0, 1.0])
        np.testing.assert_array_equal(progress_scores(values), [0.0, 0.0, 0.5, 1.0, 1.0])
        in_range = np.linspace(-1.0, 0.0, 31)
        self.assertEqual(
            epr_at_tau(progress_scores(in_range))["k"],
            epr_at_tau(in_range + 1.0)["k"],
        )


class MCCTests(unittest.TestCase):
    def test_perfect_inverse_and_degenerate(self) -> None:
        truth = np.asarray([False, False, True, True])
        self.assertEqual(mcc_from_confusion(_confusion(truth, truth)), 1.0)
        self.assertEqual(mcc_from_confusion(_confusion(truth, ~truth)), -1.0)
        self.assertEqual(
            mcc_from_confusion(_confusion(truth, np.zeros_like(truth))),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
