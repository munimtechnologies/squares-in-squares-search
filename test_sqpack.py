"""Tests for sqpack. MV2 of Munim, Inc. (made with AI agents working in his name). Apache-2.0."""
import math
import unittest

import numpy as np

import sqpack


class TestGeometry(unittest.TestCase):
    def test_rotated_corner(self):
        cx, cy = sqpack.corners(np.array([0.0]), np.array([0.0]), np.array([math.pi / 4]))
        self.assertLess(abs(cx[0, 0]), 1e-12)
        self.assertLess(abs(cy[0, 0] - math.sqrt(0.5)), 1e-12)

    def test_overlap_rejected(self):
        rep = sqpack.verify([1.0, 1.0], [1.0, 1.0], [0.0, 0.3], 3.0)
        self.assertFalse(rep["ok"])
        self.assertLess(rep["min_gap"], 0.0)

    def test_grid_touching_is_strictly_rejected_then_legalised(self):
        x = np.array([0.5, 1.5, 0.5, 1.5])
        y = np.array([0.5, 0.5, 1.5, 1.5])
        th = np.zeros(4)
        self.assertFalse(sqpack.verify(x, y, th, 2.0, 0.0)["ok"])
        out = sqpack.legalize(x, y, th)
        self.assertIsNotNone(out)
        x2, y2, th2, L, rep = out
        self.assertTrue(rep["ok"])
        self.assertLess(abs(L - 2.0), 1e-9)
        self.assertTrue(sqpack.verify(x2, y2, th2, L, 1e-12)["ok"])

    def test_tables_complete(self):
        for n in sqpack.CALIBRATION + sqpack.TARGETS:
            self.assertIn(n, sqpack.RECORDS)
            self.assertLess(sqpack.RECORDS[n], math.ceil(math.sqrt(n)))


class TestObjective(unittest.TestCase):
    def test_separated_pair_has_zero_penalty(self):
        x = np.array([0.5, 2.0])
        y = np.array([0.5, 0.5])
        th = np.zeros(2)
        I, J = sqpack.all_pairs(2)
        phi, t = sqpack.init_lines(x, y, th, I, J)
        z = np.concatenate([x, y, th, [10.0], phi, t])
        f, _ = sqpack.fg(z, 2, I, J, 1.0)
        self.assertAlmostEqual(f, 10.0, places=12)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        n = 3
        I, J = sqpack.all_pairs(n)
        P = len(I)
        z = np.concatenate([rng.uniform(0, 2, n), rng.uniform(0, 2, n), rng.uniform(0, 1.5, n),
                            [1.5], rng.uniform(0, 6.28, P), rng.uniform(0, 2, P)])
        f, g = sqpack.fg(z, n, I, J, 1.0)
        self.assertGreater(np.max(np.abs(g)), 1e-3)
        h = 1e-6
        num = np.zeros_like(z)
        for k in range(len(z)):
            zp = z.copy()
            zp[k] += h
            zm = z.copy()
            zm[k] -= h
            num[k] = (sqpack.fg(zp, n, I, J, 1.0)[0] - sqpack.fg(zm, n, I, J, 1.0)[0]) / (2 * h)
        self.assertLess(float(np.max(np.abs(num - g))), 1e-5 * (1.0 + float(np.max(np.abs(g)))))


class TestSearch(unittest.TestCase):
    def test_two_squares_reach_side_two_and_never_below(self):
        r = sqpack.search(2, 6.0, seed=1, record=2.0)
        self.assertIsNotNone(r["best_L"])
        self.assertGreaterEqual(r["best_L"], 2.0 - 1e-9)
        self.assertLessEqual(r["best_L"], 2.0 + 1e-3)
        self.assertTrue(r["verify"]["ok"])
        p = r["packing"]
        self.assertTrue(sqpack.verify(p["x"], p["y"], p["th"], p["L"], 1e-12)["ok"])


if __name__ == "__main__":
    unittest.main()
