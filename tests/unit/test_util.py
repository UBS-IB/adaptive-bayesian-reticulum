from unittest import TestCase

import numpy as np
from numpy.ma.testutils import assert_almost_equal

from reticulum.util import pick_proportional, sigmoid


class UtilTest(TestCase):
    def test_pick_proportional(self) -> None:
        np.random.seed(666)

        n = 100000

        rel_probs = np.array([1, 4])
        s = 0
        for i in range(n):
            s += pick_proportional(rel_probs)

        s /= n
        self.assertAlmostEqual(s, 0.8, delta=0.002)

        rel_probs = np.array([2, 2])
        s = 0
        for i in range(n):
            s += pick_proportional(rel_probs)

        s /= n
        self.assertAlmostEqual(s, 0.5, delta=0.002)

        rel_probs = np.array([-3, -1])
        s = 0
        for i in range(n):
            s += pick_proportional(rel_probs)

        s /= n
        self.assertAlmostEqual(s, 0.25, delta=0.002)

    def test_sigmoid(self) -> None:
        np.seterr(all='raise')

        # scalar
        self.assertEqual(sigmoid(0), 0.5)
        assert_almost_equal(sigmoid(-708), 0.0, decimal=300)
        self.assertEqual(sigmoid(-709), 0.0)
        self.assertEqual(sigmoid(-1e99), 0.0)

        self.assertEqual(sigmoid(40), 1.0)
        self.assertEqual(sigmoid(1e90), 1.0)

        # array
        assert_almost_equal(sigmoid(np.array([-708])), np.array([0.0]), decimal=300)
        assert_almost_equal(sigmoid(np.array([0.0])), np.array([0.5]), decimal=300)
        assert_almost_equal(sigmoid(np.array([40])), np.array([1.0]), decimal=300)

        assert_almost_equal(sigmoid(np.array([-708, 0.0])), np.array([0.0, 0.5]), decimal=300)
        assert_almost_equal(sigmoid(np.array([-709, 0.0])), np.array([0.0, 0.5]), decimal=300)

        assert_almost_equal(sigmoid(np.array([-708, 40])), np.array([0.0, 1.0]), decimal=300)
        assert_almost_equal(sigmoid(np.array([-709, 40])), np.array([0.0, 1.0]), decimal=300)

        assert_almost_equal(sigmoid(np.array([0.0, 40])), np.array([0.5, 1.0]), decimal=300)

        assert_almost_equal(sigmoid(np.array([-708, 0.0, 40])), np.array([0.0, 0.5, 1.0]), decimal=300)
        assert_almost_equal(sigmoid(np.array([-709, 0.0, 40])), np.array([0.0, 0.5, 1.0]), decimal=300)
        assert_almost_equal(sigmoid(np.array([-709, -708, 0.0, 40])), np.array([0.0, 0.0, 0.5, 1.0]), decimal=300)
