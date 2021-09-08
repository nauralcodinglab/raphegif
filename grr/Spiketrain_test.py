import unittest

import numpy as np
from numpy import testing as npt

from grr import Spiketrain as spk


class TestPSTH:
    def test_even_kernel_odd_train(self):
        """Even length kernel, odd length spktrain."""
        spktrain = [0, 0, 0, 1, 0, 0, 2, 0, 1]
        dt = 1000.0  # 1000 ms = 1 s
        kernel_width = 4000.0  # 4000 ms = 4s
        expected = [0, 0, 0.25, 0.25, 0.25, 0.75, 0.5, 0.75, 0.75]
        npt.assert_array_equal(
            expected, self.psth_func(spktrain, kernel_width, 1, dt=dt)
        )

    def test_odd_kernel_odd_train(self):
        """Odd length kernel, odd length spktrain."""
        spktrain = [0, 0, 0, 1, 0, 0, 2, 0, 1]
        dt = 1000.0  # 1000 ms = 1 s
        kernel_width = 3000.0  # 3000 ms = 3s
        expected = [0, 0, 0.33, 0.33, 0.33, 0.67, 0.67, 1.0, 0.33]
        npt.assert_allclose(
            expected,
            self.psth_func(spktrain, kernel_width, 1, dt=dt),
            atol=0.01,
        )

    def test_even_kernel_even_train(self):
        spktrain = [0, 0, 0, 1, 0, 0, 2, 0, 1, 0]
        dt = 1000.0  # 1000 ms = 1 s
        kernel_width = 4000.0  # 4000 ms = 4s
        expected = [0, 0.0, 0.25, 0.25, 0.25, 0.75, 0.5, 0.75, 0.75, 0.25]
        npt.assert_array_equal(
            expected, self.psth_func(spktrain, kernel_width, 1, dt=dt)
        )

    def test_odd_kernel_even_train(self):
        """Odd length kernel, even length spktrain."""
        spktrain = [0, 0, 0, 1, 0, 0, 2, 0, 1, 0]
        dt = 1000.0  # 1000 ms = 1 s
        kernel_width = 3000.0  # 3000 ms = 3s
        expected = [0, 0, 0.33, 0.33, 0.33, 0.67, 0.67, 1.0, 0.33, 0.33]
        npt.assert_allclose(
            expected,
            self.psth_func(spktrain, kernel_width, 1, dt=dt),
            atol=0.01,
        )


class SparsePSTH(TestPSTH, unittest.TestCase):
    def setUp(self):
        self.psth_func = spk._sparse_PSTH


class DensePSTH(TestPSTH, unittest.TestCase):
    def setUp(self):
        self.psth_func = spk._dense_PSTH


class ComparePSTHLongSpktrain(unittest.TestCase):
    """Ensure PSTH methods always produce same results on a long spiketrain."""

    def setUp(self):
        self.spktrain = np.zeros(10000)
        self.spktrain[200] = 2
        self.spktrain[500] = 1
        self.spktrain[550] = 1
        self.spktrain[7000] = 1
        self.spktrain[7103] = 1
        self.spktrain[7600] = 3

    def test_small_dt_short_kernel_one_neuron(self):
        dense = spk._dense_PSTH(self.spktrain, 10.0, 1, 0.1)
        sparse = spk._sparse_PSTH(self.spktrain, 10.0, 1, 0.1)
        npt.assert_allclose(dense, sparse)

    def test_large_dt_short_kernel_one_neuron(self):
        dense = spk._dense_PSTH(self.spktrain, 10.0, 1, 10.0)
        sparse = spk._sparse_PSTH(self.spktrain, 10.0, 1, 10.0)
        npt.assert_allclose(dense, sparse)

    def test_small_dt_long_kernel_one_neuron(self):
        dense = spk._dense_PSTH(self.spktrain, 100.0, 1, 0.1)
        sparse = spk._sparse_PSTH(self.spktrain, 100.0, 1, 0.1)
        npt.assert_allclose(dense, sparse)

    def test_small_dt_medium_kernel_few_neurons(self):
        dense = spk._dense_PSTH(self.spktrain, 40.0, 13, 0.1)
        sparse = spk._sparse_PSTH(self.spktrain, 40.0, 13, 0.1)
        npt.assert_allclose(dense, sparse)


if __name__ == '__main__':
    unittest.main()
