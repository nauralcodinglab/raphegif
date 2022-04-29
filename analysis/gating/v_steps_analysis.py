#%% IMPORT MODULES

from __future__ import division

import os
from textwrap import dedent

import pandas as pd
import h5py
from ezephys.rectools import ABFLoader

from grr.CurveFit import extract_gating_data, fit_gating_curve


#%% LOAD DATA

OUTPUT_PATH = os.path.join(os.getenv('DATA_PATH'), 'processed', '5HT')
if not os.path.isdir(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH, mode=755)

# Load gating data
rec_index = pd.read_csv(
    os.path.join(os.getenv('DATA_PATH'), 'raw', '5HT', 'gating', 'index.csv')
)

l = ABFLoader()
ACCESS_RESISTANCE_CUTOFF = 30.0
INCLUSION_QUERY = (
    'access_resistance_megaohm <= @ACCESS_RESISTANCE_CUTOFF and include'
)
gating = l.load(
    [
        os.path.join(os.getenv('DATA_PATH'), 'raw', '5HT', 'gating', fname)
        for fname in rec_index.query(INCLUSION_QUERY)['filename']
    ]
)
del l

#%% PROCESS RAW GATING DATA

pdata = extract_gating_data(
    gating,
    (0, 2000),
    (3500, 4000),
    (26140, 26160),
    (56130, 56160),
    (55000, 56000),
    'time_step',
)

with h5py.File(
    os.path.join(OUTPUT_PATH, 'gating_pdata_room_temp.h5'), 'w'
) as f:
    f.attrs['description'] = dedent(
        """
        Voltage-dependence of whole-cell currents in 5-HT neurons at room
        temperature.

        5-HT neurons express the transient voltage-dependent potassium current
        I_A as well as another non-inactivating voltage-dependent potassium
        current.

        The arrays in this file contain measurements of the amount of whole
        cell that can be used to construct voltage-dependence curves for the
        activation and inactivation gates of I_A and the activation gate of
        the non-inactivating current (`activation_peak`, `inactivation_peak`, and `steady_state`
        datasets, respectively). The dimensionality of each array is
        `[channel, measurement, cell]`, where `channel == 0` is the current in
        pA and `channel == 1` is the voltage at which the current was recorded
        in mV. `plt.plot(d[1, ...] d[0, ...])` can be used to plot the gating
        curves of all cells, where `d` is one of the arrays in this file.

        """[1:]
    )

    for k, v in pdata.iteritems():
        f.create_dataset(k, data=v)

    f.close()

#%% FIG SIGMOID CURVES TO GATING DATA

params = {}
fittedpts = {}
initial_param_guess = {
    'activation_peak': [12, 1, -30],
    'inactivation_peak': [12, -1, -60],
    'steady_state': [12, 1, -25],
}

for k, v in pdata.iteritems():
    p, f = fit_gating_curve(v, initial_param_guess[k])
    params[k] = p
    fittedpts[k] = f

del k, v, p, f, initial_param_guess

with h5py.File(
    os.path.join(OUTPUT_PATH, 'gating_fittedpts_room_temp.h5'), 'w'
) as f:
    f.attrs['description'] = dedent(
        """
        Voltage-dependence of whole-cell currents in 5-HT neurons at room
        temperature.

        5-HT neurons express the transient voltage-dependent potassium current
        I_A as well as another non-inactivating voltage-dependent potassium
        current.

        The arrays in this file contain sigmoid curves fitted to measurements
        of current over voltage for several cells.

        """[1:]
    )

    for k, v in fittedpts.iteritems():
        f.create_dataset(k, data=v)

    f.close()

with open(os.path.join(OUTPUT_PATH, 'gating_params.csv'), 'w') as f:
    included_recs = rec_index.query(INCLUSION_QUERY)
    f.write(
        (
            '# num_cells={}\n'
            '# access_resistance_cutoff_megaohm={}\n'
            '# access_resistance_mean_megaohm={:.2f}\n'
            '# access_resistance_std_megaohm={:.2f}\n'
            '# m="I_A activation gate"\n'
            '# h="I_A inactivation gate"\n'
            '# n="Non-inactivating current activation gate"\n'
            '# A="Sigmoid scaling factor"\n'
            '# k="Sigmoid slope"\n'
            '# V_half="Sigmoid location (half-activation voltage in mV)"\n'
        ).format(
            included_recs.shape[0],
            ACCESS_RESISTANCE_CUTOFF,
            included_recs['access_resistance_megaohm'].mean(),
            included_recs['access_resistance_megaohm'].std(),
        )
    )
    pd.DataFrame(
        {
            'm': params['activation_peak'],
            'h': params['inactivation_peak'],
            'n': params['steady_state'],
        },
        index=('A', 'k', 'V_half'),
    ).to_csv(f)
    f.close()
