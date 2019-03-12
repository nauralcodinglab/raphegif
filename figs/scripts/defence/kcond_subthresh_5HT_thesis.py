#%% IMPORT MODULES

from __future__ import division

import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as mpl
import scipy.stats as stats

import sys
sys.path.append('./src/')

sys.path.append('analysis/subthresh_mod_selection')

import src.pltools as pltools


#%% LOAD DATA

PICKLE_PATH = './figs/figdata/'

with open(PICKLE_PATH + 'ohmic_mod.pyc', 'rb') as f:
    ohmic_mod_coeffs = pickle.load(f)

with open(PICKLE_PATH + 'gk1_mod.pyc', 'rb') as f:
    gk1_mod_coeffs = pickle.load(f)

with open(PICKLE_PATH + 'gk2_mod.pyc', 'rb') as f:
    gk2_mod_coeffs = pickle.load(f)

with open(PICKLE_PATH + 'full_mod.pyc', 'rb') as f:
    full_mod_coeffs = pickle.load(f)


#%% DEFINE FUNCTIONS FOR MAKING COMPLICATED PLOTS

def binned_e_comparison_plot(null_mod, alt_mod, null_mod_label = None, alt_mod_label = None,
                             null_color = (0.1, 0.1, 0.1), alt_color = (0.8, 0.2, 0.2),
                             wilcoxon = True,
                             ax = None):

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    plt.plot(null_mod['binned_e2_centres'], null_mod['binned_e2_values'],
    '-', color = null_color, linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmedian(null_mod['binned_e2_centres'], axis = 1),
    np.nanmedian(null_mod['binned_e2_values'], axis = 1),
    '-', color = null_color, label = null_mod_label)
    plt.plot(alt_mod['binned_e2_centres'], alt_mod['binned_e2_values'],
    '-', color = alt_color, linewidth = 0.5, alpha = 0.3)
    plt.plot(np.nanmedian(alt_mod['binned_e2_centres'], axis = 1),
    np.nanmedian(alt_mod['binned_e2_values'], axis = 1),
    '-', color = alt_color, label = alt_mod_label)


def single_bin_e_comparison_plot(null_mod, alt_mod, V, null_mod_label = None, alt_mod_label = None,
                                 null_markeredgecolor = (0.2, 0.2, 0.2), null_markerfacecolor = (0.5, 0.5, 0.5),
                                 alt_markeredgecolor = (0.2, 0.2, 0.2), alt_markerfacecolor = (0.9, 0.5, 0.5),
                                 ax = None):

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    if not np.array_equal(null_mod['binned_e2_centres'], alt_mod['binned_e2_centres']):
        raise ValueError('Voltage bin centres seem to differ between null and alt models.')

    ind_of_vbin = np.where((np.abs(null_mod['binned_e2_centres'] - V) < 1e-3).all(axis = 1))[0]

    if len(ind_of_vbin) < 1:
        raise ValueError('No bins centred around approximately {}'.format(V))
    elif len(ind_of_vbin) > 1:
        raise ValueError('Multiple bins apparently centred around {}. This can happen if your bins are smaller than 10^-3 in units of the argument `V`.'.format(V))

    ind_of_vbin = ind_of_vbin[0]

    y = np.concatenate((null_mod['binned_e2_values'][ind_of_vbin, :][np.newaxis, :],
    alt_mod['binned_e2_values'][ind_of_vbin, :][np.newaxis, :]),
    axis = 0)
    x = np.zeros_like(y)
    x[1, :] = 1

    plt.plot(x, y, '-', color = 'gray', linewidth = 0.5, alpha = 0.5)
    plt.plot(x[0, :], y[0, :],
    'o', markerfacecolor = null_markerfacecolor, markeredgecolor = null_markeredgecolor, markersize = 5)
    plt.plot(x[1, :], y[1, :],
    'o', markerfacecolor = alt_markerfacecolor, markeredgecolor = alt_markeredgecolor, markersize = 5)
    plt.text(0.5, plt.ylim()[1] * 1.05,
    pltools.p_to_string(stats.wilcoxon(y[0, :], y[1, :])[1]),
    horizontalalignment = 'center', verticalalignment = 'center')
    plt.xlim(-0.2, 1.2)
    plt.ylim(plt.ylim()[0], plt.ylim()[1] * 1.2)
    plt.xticks([0, 1], [null_mod_label, alt_mod_label])


def testset_traces_plot(null_mod, alt_mod, cell_no = 0, null_mod_label = None, alt_mod_label = None,
                        real_color = (0, 0, 0), null_color = (0.1, 0.1, 0.9),
                        alt_color = (0.9, 0.1, 0.1), dt = 0.1, ax = None):

    """
    Compare testset traces for one cell.

    Note: averages sweeps from testset before plotting (looks nicer)

    Inputs:

        null_mod/alt_mod: `x_mod_coeffs`-style dicts

        cell: int
        --  Index of cell from null_mod/alt_mod for which to plot traces.
    """

    if ax is None:
        ax = plt.gca()

    if null_mod_label is None:
        null_mod_label = ''
    if alt_mod_label is None:
        alt_mod_label = ''

    t = np.arange(0, int(null_mod['real_testset_traces'][cell_no].shape[0] * dt), dt)

    plt.plot(t,
    null_mod['real_testset_traces'][cell_no].mean(axis = 1),
    '-', color = real_color, linewidth = 0.7)
    plt.plot(t,
    null_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = null_color, linewidth = 0.7, label = null_mod_label)
    plt.plot(t,
    alt_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = alt_color, linewidth = 0.7, alpha = 0.85, label = alt_mod_label)
    plt.legend(loc = 'lower left')


def extract_friedman_p(mod, lower_V_bin = -11, upper_V_bin = -3):
    """
    Non-parametric repeated measures test for voltage-dependence of error
    """

    return stats.friedmanchisquare(*[mod['binned_e2_values'][i, :] for i in range(lower_V_bin, upper_V_bin)])[1]


#%% FRIEDMAN TESTS ON VOLTAGE BINS

extract_friedman_p(ohmic_mod_coeffs)
extract_friedman_p(gk1_mod_coeffs)
extract_friedman_p(gk2_mod_coeffs)
extract_friedman_p(full_mod_coeffs)

#%% MAKE SOME LATEX TEXT FOR MODEL DEFINITIONS

ohmic_shorthand = '$\omega = I(t)- g_l\\times(V(t) - E_l)$'
ohmic_latex = '$C\dot{{V}}(t) = \omega$'
gk1_latex = '$C\dot{{V}}(t) = \omega - \\bar{{g}}_{{Kfast}}mh\\times(V(t) - E_k)$'
gk2_latex = '$C\dot{{V}}(t) = \omega - \\bar{{g}}_{{Kslow}}n\\times(V(t) - E_k)$'
full_latex = '$C\dot{{V}}(t) = \omega - (\\bar{{g}}_{{Kfast}}mh + \\bar{{g}}_{{Kslow}}n)\\times(V(t) - E_k)$'

#%% ASSEMBLE FIGURE

linear_color = (.93, .53, .14)
Kfast_color = (0.3, 0.3, 0.9)
Kslow_color = (0.9, 0.2, 0.2)

plt.style.use('./figs/scripts/thesis/thesis_mplrc.dms')

IMG_PATH = './figs/ims/defence/'

plt.figure(figsize = (6, 5))

spec_outer = gridspec.GridSpec(
    2, 1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.1, hspace = 0.5
)
subfig_hspace = 1.3
subfig_wspace = 0.4
spec_A = gridspec.GridSpecFromSubplotSpec(
    2, 3, spec_outer[0, :], width_ratios = [1.1, 1, 0.5],
    hspace = subfig_hspace, wspace = subfig_wspace
)
spec_B = gridspec.GridSpecFromSubplotSpec(
    2, 3, spec_outer[1, :], width_ratios = [1.1, 1, 0.5],
    hspace = subfig_hspace, wspace = subfig_wspace
)

"""
plt.subplot(spec[0, :2])
plt.title('\\textbf{{A1}} Model definitions', loc = 'left')
plt.text(0, 0.9,
'\n'.join(['$\mathrm{{H}}_0$: ' + ohmic_latex, '$\mathrm{{H}}_a$: ' + gk1_latex, 'Where ' + ohmic_shorthand]),
horizontalalignment = 'left', verticalalignment = 'top')
pltools.hide_border()
pltools.hide_ticks()
"""

plt.subplot(spec_A[:, 0])
plt.title('\\textbf{{A1}} $I_A$ alone', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk1_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear + $I_A$',
null_color = linear_color,
alt_color = Kfast_color
)
plt.axhline(-70, color = 'k', linestyle = '--', dashes = (10, 10), lw = 0.5, zorder = 0)
plt.text(
5200, -70,
'$-70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)
plt.xlim(4000, 8000)
plt.ylim(plt.ylim()[0] * 1.1, plt.ylim()[1])
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', bar_space = 0)


plt.subplot(spec_A[:, 1])
plt.title('\\textbf{{A2}} Binned test set error', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk1_mod_coeffs,
'Linear model', 'Linear + $I_A$',
null_color = linear_color,
alt_color = Kfast_color
)

plt.text(-60, 80, 'A3', horizontalalignment = 'center')
plt.text(-45, 140, 'A4', horizontalalignment = 'center')

plt.ylim(0, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec_A[0, 2])
plt.title('\\textbf{{A3}}  -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -60,
                             'Linear', '$I_A$',
                             null_markerfacecolor = linear_color,
                             alt_markerfacecolor = Kfast_color)
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec_A[1, 2])
plt.title('\\textbf{{A4}}  -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -45,
                             'Linear', '$I_A$',
                             null_markerfacecolor = linear_color,
                             alt_markerfacecolor = Kfast_color)
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


# B: gk2 vs ohmic
"""
plt.subplot(spec_B[:, 0])
plt.title('\\textbf{{B1}} Model definitions', loc = 'left')
plt.text(0, 0.9,
'\n'.join(['$\mathrm{{H}}_0$: ' + ohmic_latex, '$\mathrm{{H}}_a$: ' + gk2_latex, 'Where ' + ohmic_shorthand]),
horizontalalignment = 'left', verticalalignment = 'top')
pltools.hide_border()
pltools.hide_ticks()
"""

plt.subplot(spec_B[:, 0])
plt.title('\\textbf{{B1}} $K_{{\mathrm{{slow}}}}$ alone', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk2_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear + $K_{{\mathrm{{slow}}}}$',
null_color = linear_color,
alt_color = Kslow_color
)
plt.axhline(-70, color = 'k', linestyle = '--', dashes = (10, 10), lw = 0.5, zorder = 0)
plt.text(
5200, -70,
'$-70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)
plt.xlim(4000, 8000)
plt.ylim(plt.ylim()[0] * 1.1, plt.ylim()[1])
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', bar_space = 0)

plt.subplot(spec_B[:, 1])
plt.title('\\textbf{{B2}} Binned test set error', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk2_mod_coeffs,
'Linear model', 'Linear + $K_{{\mathrm{{slow}}}}$',
null_color = linear_color,
alt_color = Kslow_color
)

plt.text(-60, 100, 'B3', horizontalalignment = 'center')
plt.text(-45, 140, 'B4', horizontalalignment = 'center')

plt.ylim(0, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec_B[0, 2])
plt.title('\\textbf{{B3}}  -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -60,
                             'Linear', '$K_{{\mathrm{{slow}}}}$',
                             null_markerfacecolor = linear_color,
                             alt_markerfacecolor = Kslow_color)
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec_B[1, 2])
plt.title('\\textbf{{B4}}  -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -45,
                             'Linear', '$K_{{\mathrm{{slow}}}}$',
                             null_markerfacecolor = linear_color,
                             alt_markerfacecolor = Kslow_color)
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')



plt.savefig(IMG_PATH + 'kcond_subthresh_5HT_comparison.png', dpi = 300)
plt.show()
