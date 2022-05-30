"""Utilities for gain analysis of GIFnet simulations.

See Also
--------
gain_illustration.ipynb
gain_illustration_temp_comparison.ipynb

"""
import re
import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from grr.Tools import timeToIndex, raiseExpectedGot
from grr.GainEstimator import GainEstimator
from grr.CurveFit import plot_linear_fit

from common import savefig


def infer_gaba_condition(dir_name):
    """Extract GABA condition from path to data file.

    Determine the manipulation applied to GABA cells in a GIF network
    simulation (e.g., knockout, endocannabinoid-like weakening of inputs, etc.)
    based on the path to the data file.

    """
    regex_match = re.search(r'GABA_(.*)', dir_name)
    if regex_match is not None:
        condition = regex_match.groups()[-1]
    elif 'endocannabinoid' == dir_name:
        condition = 'endocannabinoid'
    else:
        raise ValueError('Unrecognized GABA condition {}'.format(dir_name))
    return condition


def get_step_amplitudes(simulation, step_start_time, dt=0.1):
    """Extract amplitudes of current steps applied to GIF networks."""
    step_start_ind = timeToIndex(step_start_time, dt)[0]
    baseline = simulation['ser/examples/I'][..., :step_start_ind].mean()
    return (
        simulation['ser/examples/I'][..., step_start_ind:]
        .mean(axis=2)
        .mean(axis=1)
        - baseline
    )


def get_pointwise_gain(
    psth_arr,
    step_amplitudes,
    baseline_start_time,
    step_start_time,
    cov=False,
    dt=0.1,
):
    """
    Returns
    -------
    (gain, intercept) if cov = False, or (gain, intercept,
    parameter_covariance_matrix) if cov=True.

    """
    assert np.ndim(psth_arr) == 3
    assert np.ndim(step_amplitudes) == 1
    assert np.shape(psth_arr)[1] == len(step_amplitudes)

    step_start_ind = timeToIndex(step_start_time, dt)[0]
    step_response = psth_arr[..., step_start_ind:]
    mean_step_response = step_response.mean(axis=0)

    baseline_activity = psth_arr[
        ..., timeToIndex(baseline_start_time, dt)[0] : step_start_ind
    ]
    mean_baseline_activity = baseline_activity.mean(axis=0).mean(
        axis=1
    )  # Average over sweeps and time.
    assert np.ndim(mean_baseline_activity) == 1
    assert len(mean_baseline_activity) == mean_step_response.shape[0]

    coeffs, V = np.polyfit(
        step_amplitudes,
        mean_step_response - mean_baseline_activity[:, np.newaxis],
        deg=1,
        cov=cov,
    )

    if cov:
        return coeffs[0, :], coeffs[1, :], V
    else:
        return coeffs[0, :], coeffs[1, :]


def plot_lines_cmap(x, y, cmap, start=0.0, stop=1.0, ax=None, **pltargs):
    """Plot multiple lines with a color gradient across lines.

    Parameters
    ----------
    x: 1D array-like
    y: 2D array-like
        y-coordinates of multiple lines where each column is one line.
        Normalized column index is passed to cmap to determine the colour of
        each line.
    cmap: matplotlib colormap
    start, stop: float
        Normalize the column indices passed to cmap to this range.
    ax: matplotlib.Axes or None
        Axes on which to plot the lines.
    pltargs
        Keyword arguments passed to plt.plot().

    """
    if ax is None:
        ax = plt.gca()

    colors = cmap(np.linspace(start, stop, np.shape(y)[1]))
    for i in range(np.shape(y)[1]):
        ax.plot(x, y[:, i], color=colors[i], **pltargs)


def select_PSTH_dataset(
    dframe, circuit, condition, mod_type, psth_type='Principal PSTH'
):
    return dframe.loc[
        (dframe['Circuit'] == circuit)
        & (dframe['Condition'] == condition)
        & (dframe['Mod type'] == mod_type),
        psth_type,
    ].item()


class GainSimulationVisualizer(object):
    def __init__(
        self,
        psth_dataframe,
        step_amplitudes,
        baseline_interval,
        stimulus_interval,
        dt=0.1,
    ):
        """Initialize GainSimulationVisualizer.

        Arguments
        ---------
        psth_dataframe : pandas.DataFrame
        step_amplitudes : 1D float array
        baseline_interval, stimulus_interval : pair of floats
            Time window to extract baseline/stimulus (ms).
        dt : float, default 0.1
            Time step (ms).

        """
        self.psth_dataframe = psth_dataframe
        self.step_amplitudes = step_amplitudes
        self.baseline_interval = baseline_interval
        self.stimulus_interval = stimulus_interval
        self.dt = dt

    def plot_gain(
        self,
        circuit,
        condition,
        mod_type,
        psth_type,
        ax=None,
        label=None,
        **pltargs
    ):
        """Plot gain over time.

        Wraps select_PSTH_dataset.

        """
        if ax is None:
            ax = plt.gca()

        gain_estimator = self._fit_gain_estimator(
            circuit, condition, mod_type, psth_type
        )

        ax.fill_between(
            self._get_time_support(),
            gain_estimator.gain - gain_estimator.gainUncertainty,
            gain_estimator.gain + gain_estimator.gainUncertainty,
            alpha=0.7,
            lw=0,
            **pltargs
        )
        ax.plot(
            self._get_time_support(),
            gain_estimator.gain,
            label=label,
            **pltargs
        )

    def plot_intercept(
        self,
        circuit,
        condition,
        mod_type,
        psth_type,
        ax=None,
        label=None,
        **pltargs
    ):
        """Plot intercept over time.

        Wraps select_PSTH_dataset.

        """
        if ax is None:
            ax = plt.gca()

        gain_estimator = self._fit_gain_estimator(
            circuit, condition, mod_type, psth_type
        )

        ax.fill_between(
            self._get_time_support(),
            gain_estimator.intercept - gain_estimator.interceptUncertainty,
            gain_estimator.intercept + gain_estimator.interceptUncertainty,
            alpha=0.7,
            lw=0,
            **pltargs
        )
        ax.plot(
            self._get_time_support(),
            gain_estimator.intercept,
            label=label,
            **pltargs
        )

    def _fit_gain_estimator(self, circuit, condition, mod_type, psth_type):
        dset = select_PSTH_dataset(
            self.psth_dataframe, circuit, condition, mod_type, psth_type
        )
        gain_estimator = GainEstimator(self.dt)
        gain_estimator.fit(
            dset,
            self.step_amplitudes,
            self.baseline_interval,
            self.stimulus_interval,
        )
        return gain_estimator

    def plot_psth(
        self,
        circuit,
        condition,
        mod_type,
        psth_type,
        sweeps,
        cmap,
        ax=None,
        label=None,
        **pltargs
    ):
        dset = select_PSTH_dataset(
            self.psth_dataframe, circuit, condition, mod_type, psth_type
        )
        psth_mean = dset[..., self._get_stimulus_slice()].mean(axis=0)
        psth_std = dset[..., self._get_stimulus_slice()].std(axis=0)

        if ax is None:
            ax = plt.gca()

        colors = cmap(np.linspace(0.3, 1.0, psth_mean.shape[0]))
        labeled_flag = False
        for i in sweeps:
            ax.fill_between(
                self._get_time_support(),
                psth_mean[i, :] - psth_std[i, :],
                psth_mean[i, :] + psth_std[i, :],
                color=colors[i],
                alpha=0.7,
                lw=0,
                **pltargs
            )
            if not labeled_flag:
                ax.plot(
                    self._get_time_support(),
                    psth_mean[i, :],
                    color=colors[i],
                    label=label,
                    **pltargs
                )
                labeled_flag = True
            else:
                ax.plot(
                    self._get_time_support(),
                    psth_mean[i, :],
                    color=colors[i],
                    **pltargs
                )

    def _get_stimulus_slice(self):
        return slice(*timeToIndex(self.stimulus_interval, self.dt))

    def _get_time_support(self):
        return np.arange(
            0.0,
            (self.stimulus_interval[1] - self.stimulus_interval[0])
            - 0.5 * self.dt,
            self.dt,
        )


class PSTHToGainIllustration(object):
    def __init__(
        self,
        gain_simulation_visualizer,
        marked_times=[],
        marker_fmt_strings=[],
        marker_labels=[],
        markeredgecolor='none',
        markeredgewidth=1,
        ebar_params={
            'ls': 'none',
            'capsize': 3,
            'ecolor': 'k',
            'markersize': 3,
            'elinewidth': 1.5,
        },
    ):
        if len(marked_times) != len(marker_fmt_strings):
            raiseExpectedGot(
                'equal number of values',
                'arguments `marked_times` and `marker_fmt_strings`',
                'lists of length {} and {}'.format(
                    len(marked_times), len(marker_fmt_strings)
                ),
            )
        if len(marker_labels) != 0 and len(marker_labels) != len(marked_times):
            raiseExpectedGot(
                'list of same length as `marked_times`, or empty list',
                'argument `marker_labels`',
                'list of length {}'.format(len(marker_labels)),
            )

        self._visualizer = gain_simulation_visualizer
        self._marked_times = marked_times
        self._marker_fmt_strings = marker_fmt_strings
        self._marker_labels = marker_labels
        self.markeredgecolor = markeredgecolor
        self.markeredgewidth = markeredgewidth
        self.ebar_params = ebar_params

    @property
    def _step_amplitudes(self):
        return self._visualizer.step_amplitudes

    def select_PSTH_dataset(self, circuit, condition, mod_type, psth_type):
        self._dataset_identifiers = {
            'circuit': circuit,
            'condition': condition,
            'mod_type': mod_type,
            'psth_type': psth_type,
        }
        self._psth_dataset = select_PSTH_dataset(
            self._visualizer.psth_dataframe, **self._dataset_identifiers
        )

    def plot_PSTH_to_gain(self, cmap, color, label, file_name=None):
        plt.figure(figsize=(6, 1.5))

        plt.subplot(131)
        self.plot_marked_PSTH(cmap)
        plt.ylabel('{} pop. firing rate\n(Hz/neuron)'.format(label))
        plt.xlabel('Time from step onset (ms)')
        plt.legend(markerscale=1)

        plt.subplot(132)
        self.plot_fi_curves(
            fit_pltargs={'zorder': -1, 'lw': 1, 'ls': '--', 'color': 'gray'}
        )
        plt.ylabel('{} pop. firing rate\n(Hz/neuron)'.format(label))
        plt.xlabel('Step amplitude (nA)')

        plt.subplot(133)
        self.plot_marked_gain(color)
        plt.ylabel('Gain\n(Hz neuron$^{-1}$ nA$^{-1}$)')
        plt.xlabel('Time from step onset (ms)')

        sns.despine(trim=True)

        plt.tight_layout()

        if file_name is not None:
            savefig(file_name)

    def plot_marked_PSTH(self, cmap, ax=None):
        if ax is None:
            ax = plt.gca()

        self._visualizer.plot_psth(
            sweeps=range(0, 10, 3),
            cmap=cmap,
            ax=ax,
            **self._dataset_identifiers
        )
        if len(self._marker_labels) != 0:
            for time, fmt_string, label in zip(
                self._marked_times,
                self._marker_fmt_strings,
                self._marker_labels,
            ):
                ax.plot(
                    [time - self._visualizer.stimulus_interval[0]] * 4,
                    self._psth_dataset[
                        :, ::3, timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    fmt_string,
                    label=label,
                    markeredgecolor=self.markeredgecolor,
                    markeredgewidth=self.markeredgewidth,
                )
        else:
            for time, fmt_string in zip(
                self._marked_times, self._marker_fmt_strings
            ):
                ax.plot(
                    [time - self._visualizer.stimulus_interval[0]] * 4,
                    self._psth_dataset[
                        :, ::3, timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    fmt_string,
                    markeredgecolor=self.markeredgecolor,
                    markeredgewidth=self.markeredgewidth,
                )

    def plot_fi_curves(
        self, curves='all', fitted=True, fit_pltargs={}, ax=None
    ):
        """Plot population f/I curves.

        Arguments
        ---------
        curves: iterable of ints or `all`
            Plot f/I curves corresponding to these indices of `marked_times`.
        fitted: boolean, default True
            Whether to plot a linear fit to each f/I curve.
        fit_pltargs: dict
            Formatting parameters for linear fits. Passed to ax.plot() as keyword
            arguments.
        ax: matplotlib Axes
            Defaults to current axes.

        """
        if ax is None:
            ax = plt.gca()

        if curves == 'all':
            curves = range(len(self._marker_labels))

        ebar_params = copy.deepcopy(self.ebar_params)
        line_style = ebar_params.pop('ls', None)
        if line_style is None:
            ebar_params.pop('linestyle', None)

        for ind in curves:
            time = self._marked_times[ind]
            fmt_string = self._marker_fmt_strings[ind]

            if len(self._marker_labels) != 0:
                ax.plot(
                    self._step_amplitudes,
                    self._psth_dataset[
                        ..., timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    fmt_string,
                    markeredgecolor=self.markeredgecolor,
                    markeredgewidth=self.markeredgewidth,
                    markersize=ebar_params.get('markersize', 3),
                    label=self._marker_labels[ind],
                    ls='none',
                    zorder=2,
                )
            else:
                ax.plot(
                    self._step_amplitudes,
                    self._psth_dataset[
                        ..., timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    fmt_string,
                    markeredgecolor=self.markeredgecolor,
                    markeredgewidth=self.markeredgewidth,
                    markersize=ebar_params.get('markersize', 3),
                    ls='none',
                    zorder=2,
                )

            err_fmt_string = copy.deepcopy(fmt_string)
            err_fmt_string = (
                err_fmt_string.replace('.', '')
                .replace('o', '')
                .replace('s', '')
                .replace('^', '')
            )

            ax.errorbar(
                x=self._step_amplitudes,
                y=self._psth_dataset[
                    ..., timeToIndex(time, self._visualizer.dt)[0]
                ].mean(axis=0),
                yerr=self._psth_dataset[
                    ..., timeToIndex(time, self._visualizer.dt)[0]
                ].std(axis=0),
                fmt=err_fmt_string,
                marker=None,
                mew=1,
                ls='none',
                zorder=1,
                **ebar_params
            )

            if line_style is not None:
                ax.plot(
                    self._step_amplitudes,
                    self._psth_dataset[
                        ..., timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    fmt_string,
                    ls=line_style,
                    marker=None,
                    zorder=-1,
                )

        if fitted:
            for ind in curves:
                time = self._marked_times[ind]
                plot_linear_fit(
                    x=self._step_amplitudes,
                    y=self._psth_dataset[
                        ..., timeToIndex(time, self._visualizer.dt)[0]
                    ].mean(axis=0),
                    ax=ax,
                    **fit_pltargs
                )

    def plot_marked_gain(self, color, ax=None):
        if ax is None:
            ax = plt.gca()

        self._visualizer.plot_gain(
            ax=ax, color=color, **self._dataset_identifiers
        )

        gain = GainEstimator(self._visualizer.dt)
        gain.fit(
            self._psth_dataset,
            self._step_amplitudes,
            self._visualizer.baseline_interval,
            self._visualizer.stimulus_interval,
        )

        for time, fmt_string in zip(
            self._marked_times, self._marker_fmt_strings
        ):
            ax.plot(
                [time - self._visualizer.stimulus_interval[0]],
                [
                    gain.gain[
                        timeToIndex(
                            time - self._visualizer.stimulus_interval[0],
                            self._visualizer.dt,
                        )[0]
                    ]
                ],
                fmt_string,
                markeredgecolor=self.markeredgecolor,
                markeredgewidth=self.markeredgewidth,
            )

    def copy(self):
        return copy.deepcopy(self)
