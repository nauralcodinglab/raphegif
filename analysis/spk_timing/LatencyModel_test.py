import unittest

import numpy as np
import numpy.testing as npt
from sklearn.model_selection import ParameterGrid

from grr.AugmentedGIF import AugmentedGIF
import LatencyModel as lmod


class Test_IALatencyModel(unittest.TestCase):

    def test_timed_integrate_crosses_thresh(self):
        """Ensure `_timed_integrate_to_bound` returns sub max-time values."""
        iamod = lmod.IALatencyModel(0.1)

        param_range = {
            'V0': [-70.],
            'Vin': [-20., 10.],
            'thresh': [-55., -45.],
            'tau': [50., 100.],
            'ga': [0.],
            'tauh': [100.],
            'max_time': [2000.],
            'time_step': [iamod.dt]
        }
        for param_set in ParameterGrid(param_range):
            t = iamod._timed_integrate_to_bound(**param_set)
            self.assertTrue(
                t < param_set['max_time'] - param_set['time_step'],
                '`_timed_integrate_to_bound` does not reach bound for '
                'params {}.'.format(param_set)
            )

        param_range = {
            'V0': [-70.],
            'Vin': [-20., 10.],
            'thresh': [-55., -45.],
            'tau': [50., 100.],
            'ga': [10.],
            'tauh': [100.],
            'max_time': [2000.],
            'time_step': [iamod.dt]
        }
        for param_set in ParameterGrid(param_range):
            t = iamod._timed_integrate_to_bound(**param_set)
            self.assertTrue(
                t < param_set['max_time'] - param_set['time_step'],
                '`_timed_integrate_to_bound` does not reach bound for '
                'params {}.'.format(param_set)
            )

    def test_timed_integrate_time_step(self):
        """Ensure `_timed_integrate_to_bound` time_step minimally changes output."""
        iamod = lmod.IALatencyModel(0.1)

        time_params = {
            'max_time': np.linspace(5, 5000., 3),
            'time_step': np.linspace(0.05, 0.5, 3)
        }
        for param_set in ParameterGrid(time_params):
            t = iamod._timed_integrate_to_bound(
                0., -1., 1., 1., 1., 1., **param_set
            )
            npt.assert_allclose(
                t, param_set['max_time'], 1e-7, param_set['time_step'],
                err_msg='`max_time` not respected for params {}'.format(
                    param_set
                )
            )

    def test_predict_vs_OhmicLatencyModel(self):
        """Ensure consistency of `predict` with `OhmicLatencyModel`.

        When `IALatencyModel.predict` is passed 0 for ga,
        results should be the same as for `OhmicLatencyModel`

        """
        dt = 0.001
        ohmic_params = {
            'tau': np.linspace(10., 100., 3),
            'thresh': [-50., -42.5],
            'V0': [-70., -63.2],
            'Vin': [-10., -5.3]
        }
        iamod = lmod.IALatencyModel(dt)
        omod = lmod.OhmicLatencyModel(dt)
        for params in ParameterGrid(ohmic_params):
            t_ohm = omod.predict(**params)
            t_ia = iamod.predict(
                ga=0., tauh=1., max_time=1e3, **params
            )
            npt.assert_allclose(
                t_ia, t_ohm, rtol=5e-2, atol=dt,
                err_msg='IALatencyModel not equivalent to OhmicLatencyModel '
                'for {}'.format(params)
            )

if __name__ == '__main__':
    unittest.main()
