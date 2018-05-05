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
sys.path.append('./figs/scripts/')
sys.path.append('analysis/subthresh_mod_selection')

import pltools


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
                                 null_markeredgecolor = (0.1, 0.1, 0.1), null_markerfacecolor = (0.5, 0.5, 0.5),
                                 alt_markeredgecolor = (0.9, 0.1, 0.1), alt_markerfacecolor = (0.9, 0.5, 0.5),
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

    plt.plot(x, y, '-', color = 'gray', alpha = 0.5)
    plt.plot(x[0, :], y[0, :],
    'o', markerfacecolor = null_markerfacecolor, markeredgecolor = null_markeredgecolor, markersize = 10)
    plt.plot(x[1, :], y[1, :],
    'o', markerfacecolor = alt_markerfacecolor, markeredgecolor = alt_markeredgecolor, markersize = 10)
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
    '-', color = real_color, linewidth = 2, label = 'Real data')
    plt.plot(t,
    null_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = null_color, linewidth = 2, label = null_mod_label)
    plt.plot(t,
    alt_mod['simulated_testset_traces'][cell_no].mean(axis = 1),
    '-', color = alt_color, linewidth = 2, alpha = 0.85, label = alt_mod_label)
    plt.legend()


#%% MAKE SOME LATEX TEXT FOR MODEL DEFINITIONS

ohmic_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l)$'
gk1_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - \\bar{{g}}_{{Kfast}}mh\\times(V(t) - E_k)$'
gk2_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - \\bar{{g}}_{{Kslow}}n\\times(V(t) - E_k)$'
full_latex = '$C\dot{{V}}(t) = I(t) - g_l\\times(V(t) - E_l) - (\\bar{{g}}_{{Kfast}}mh + \\bar{{g}}_{{Kslow}}n)\\times(V(t) - E_k)$'

#%% ASSEMBLE FIGURE

Kfast_color = (0.2, 0.2, 0.8)
Kslow_color = (0.8, 0.2, 0.2)

mpl.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath',              # <- tricky! -- gotta actually tell tex to use!
]
mpl.rc('text', usetex = True)
mpl.rc('svg', fonttype = 'none')

SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

IMG_PATH = './figs/ims/'

plt.figure(figsize = (14.67, 18))

spec = gridspec.GridSpec(12, 5,
left = 0.05, right = 0.95, top = 0.95, bottom = 0.05, hspace = 1.5, wspace = 0.4)


plt.subplot(spec[0, :2])
plt.title('\\textbf{{A1}} Model definitions', loc = 'left')
plt.text(0, 0.5,
'\n'.join([ohmic_latex, gk1_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[1:4, :2])
plt.title('\\textbf{{A2}} Sample predictions on test set', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk1_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear model + $g_{{Kfast}}$'
)
plt.axhline(-70, color = 'k', linestyle = '--', dashes = (10, 10), zorder = 0)
plt.text(
5200, -70,
'$V_m = -70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)
plt.xlim(4000, 8000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', text_spacing = (0.02, -0.05), bar_spacing = 0)


plt.subplot(spec[0:4, 2:4])
plt.title('\\textbf{{A3}} Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk1_mod_coeffs,
'Linear model', 'Linear model + $g_{{Kfast}}$')

plt.text(-60, 80, 'A4', horizontalalignment = 'center')
plt.text(-45, 140, 'A5', horizontalalignment = 'center')

plt.ylim(0, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[0:2, 4])
plt.title('\\textbf{{A4}} Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -60,
                             'Linear', 'Linear + $g_{{Kfast}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[2:4, 4])
plt.title('\\textbf{{A5}} Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk1_mod_coeffs, -45,
                             'Linear', 'Linear + $g_{{Kfast}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


# B: gk2 vs ohmic
plt.subplot(spec[4, :2])
plt.title('\\textbf{{B1}} Model definitions', loc = 'left')
plt.text(0, 0.5,
'\n'.join([ohmic_latex, gk2_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[5:8, :2])
plt.title('\\textbf{{B2}} Sample predictions on test set', loc = 'left')
testset_traces_plot(
ohmic_mod_coeffs, gk2_mod_coeffs, 13,
null_mod_label = 'Linear model',
alt_mod_label = 'Linear model + $g_{{Kslow}}$'
)
plt.axhline(-70, color = 'k', linestyle = '--', dashes = (10, 10), zorder = 0)
plt.text(
5200, -70,
'$V_m = -70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)
plt.xlim(4000, 8000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', text_spacing = (0.02, -0.05), bar_spacing = 0)

plt.subplot(spec[4:8, 2:4])
plt.title('\\textbf{{B3}} Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
ohmic_mod_coeffs, gk2_mod_coeffs,
'Linear model', 'Linear model + $g_{{Kslow}}$')

plt.text(-60, 100, 'B4', horizontalalignment = 'center')
plt.text(-45, 140, 'B5', horizontalalignment = 'center')

plt.ylim(0, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[4:6, 4])
plt.title('\\textbf{{B4}} Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -60,
                             'Linear', 'Linear + $g_{{Kslow}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[6:8, 4])
plt.title('\\textbf{{B5}} Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(ohmic_mod_coeffs, gk2_mod_coeffs, -45,
                             'Linear', 'Linear + $g_{{Kslow}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


# C: gk1 given gk2
plt.subplot(spec[8, :2])
plt.title('\\textbf{{C1}} Model definitions', loc = 'left')
plt.text(0.0, 0.5,
'\n'.join([gk2_latex, full_latex]),
horizontalalignment = 'left', verticalalignment = 'center')
pltools.hide_border()
pltools.hide_ticks()

plt.subplot(spec[9:12, :2])
plt.title('\\textbf{{C2}} Sample predictions on test set', loc = 'left')
testset_traces_plot(
gk2_mod_coeffs, full_mod_coeffs, 13,
null_mod_label = 'Linear model + $g_{{Kslow}}$',
alt_mod_label = 'Linear model + $g_{{Kfast}}$ \& $g_{{Kslow}}$'
)
plt.axhline(-70, color = 'k', linestyle = '--', dashes = (10, 10), zorder = 0)
plt.text(
5200, -70,
'$V_m = -70$mV',
horizontalalignment = 'center',
verticalalignment = 'bottom'
)
plt.xlim(4000, 8000)
pltools.add_scalebar(x_units = 'ms', y_units = 'mV', text_spacing = (0.02, -0.05), bar_spacing = 0)

plt.subplot(spec[8:12, 2:4])
plt.title('\\textbf{{C3}} Model error according to voltage', loc = 'left')
binned_e_comparison_plot(
gk2_mod_coeffs, full_mod_coeffs,
'Linear model + $g_{{Kslow}}$', 'Linear model + $g_{{Kfast}}$ \& $g_{{Kslow}}$')

plt.text(-60, 90, 'C4', horizontalalignment = 'center')
plt.text(-45, 80, 'C5', horizontalalignment = 'center')

plt.ylim(0, 310)
plt.legend(loc = 'upper right')
plt.xlabel('$V_m$ (mV)')
plt.ylabel('Mean squared error ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[8:10, 4])
plt.title('\\textbf{{C4}} Error at -60mV', loc = 'left')
single_bin_e_comparison_plot(gk2_mod_coeffs, full_mod_coeffs, -60,
                             '$g_{{Kslow}}$', '$g_{{Kfast}} + g_{{Kslow}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')

plt.subplot(spec[10:12, 4])
plt.title('\\textbf{{C5}} Error at -45mV', loc = 'left')
single_bin_e_comparison_plot(gk2_mod_coeffs, full_mod_coeffs, -45,
                             '$g_{{Kslow}}$', '$g_{{Kfast}} + g_{{Kslow}}$')
plt.ylabel('MSE ($\mathrm{{mV}}^2$)')
pltools.hide_border('tr')


plt.savefig(IMG_PATH + 'fig4_modelComparison.png', dpi = 300)
plt.show()
