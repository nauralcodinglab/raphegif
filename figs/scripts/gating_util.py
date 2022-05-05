import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from grr.Tools import raiseExpectedGot, assertAllAlmostSame
from grr.cell_class import max_normalize_channel


def mark_time_interval(interval, time_unit, dt, ax=None, **pltargs):
    """Mark a time interval on a plot."""
    if ax is None:
        ax = plt.gca()
    if time_unit in {'time_step', 'timestep'}:
        interval = np.array(interval) * dt
    elif time_unit != 'ms':
        raiseExpectedGot("'time_step' or 'ms'", "'time_unit'", time_unit)
    ax.axvspan(*interval, **pltargs)


def save_gating_params(
    f,
    params,
    analyzed_recordings,
    access_resistance_cutoff_megaohm,
    print_metadata=True,
):
    """Save gating curve parameters and metadata to CSV.

    Parameters
    ----------
    f: str or file
    params: pd.DataFrame
    analyzed_recordings: pd.DataFrame
    access_resistance_cutoff_megaohm: float
        Access resistance (Ra) inclusion criterion; recordings with access
        resistance above this value are excluded.
    print_metadata: bool

    See Also
    --------
    load_gating_params()

    """
    metadata = (
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
        analyzed_recordings.shape[0],
        access_resistance_cutoff_megaohm,
        analyzed_recordings['access_resistance_megaohm'].mean(),
        analyzed_recordings['access_resistance_megaohm'].std(),
    )

    if not isinstance(f, file):
        f = open(f, 'w')
    try:
        f.write(metadata)
        params.to_csv(f)
    finally:
        f.close()

    if print_metadata:
        print(metadata)


def load_gating_params(f):
    """Load parameters of sigmoid gating curves from CSV.

    See Also
    --------
    save_gating_params()

    """
    return pd.read_csv(f, index_col=0, comment='#')


def get_plot_coordinates_with_error(pdata, error_type='sem'):
    """Compute data point positions and error bars for gating plot."""
    if error_type not in ['std', 'sem']:
        raiseExpectedGot('`std` or `sem`', 'argument `error_type`', error_type)

    # Ensure voltages are same for all observations.
    for i in range(pdata.shape[1]):
        assertAllAlmostSame(
            pdata[1, i, :]
        )  # Voltage channel should be [1, :, :]

    x = pdata[1, ...].mean(axis=1)
    y_mean = max_normalize_channel(pdata[0, ...]).mean(axis=1)
    y_std = max_normalize_channel(pdata[0, ...]).std(axis=1)
    if error_type == 'sem':
        y_err = y_std / np.sqrt(pdata.shape[2])
    elif error_type == 'std':
        y_err = y_std
    else:
        raise RuntimeError('Unexpectedly reached end of switch.')

    return {'x': x, 'y': y_mean, 'yerr': y_err}
