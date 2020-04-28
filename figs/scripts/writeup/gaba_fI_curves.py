# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Python 2
#     language: python
#     name: python2
# ---

# ## Import modules

# +
from __future__ import division

import os; os.chdir(os.path.join('..', '..', '..'))
print os.getcwd()

# +
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd
from scipy import stats

from grr.cell_class import Cell
from grr.Trace import detectSpikes
from grr.Tools import stripNan
from ezephys.pltools import hide_border
# -

plt.style.use(os.path.join('figs', 'scripts', 'writeup', 'publication_figure_style.dms'))
IMG_PATH = os.path.join('figs', 'ims', 'writeup', 'fI')

# ## Load data

# First we'll just load the current step data.

# +
DATA_PATH = os.path.join('data', 'raw', 'GABA', 'current_steps')

fnames = pd.read_csv(os.path.join(DATA_PATH, 'index.csv'))
fnames
# -

# **Drop JF's cells for now** since only one is usable and would require significant code modification. (2 cells with current channel not registered, one cell with current steps at a different time than my cells.)

fnames.drop(fnames.index[14:17], inplace = True)

curr_steps = Cell().read_ABF([os.path.join(DATA_PATH, fn) for fn in fnames['Steps']])
curr_steps[0].plot()

# Current step recordings have a similar structure, but differ in number of sweeps. Also possibly in spacing of current steps.
#
# Automatically detect the start/end of the test pulse and current steps based on the first recording and then show whether this works for all cells.

# +
change_threshold = 5. # pA threshold at which to detect a step.

tstpts = {}
mainpts = {}

try:
    tstpts['start'], mainpts['start'] = np.where(np.diff(curr_steps[0][1, :, 0]) < -change_threshold)[0]
    tstpts['stop'], mainpts['stop'] = np.where(np.diff(curr_steps[0][1, :, 0]) > change_threshold)[0]
except ValueError:
    print 'Too many or too few steps detected. Might need to adjust `change_threshold`.'
    raise

del change_threshold

# +
dt = 0.1 # ms. Assumed.

buffer_timesteps = 500

plt.figure()

tst_ax = plt.subplot(121)
tst_ax.set_title('Test pulse')

step_ax = plt.subplot(122)
step_ax.set_title('Current step')

for expt in curr_steps:
    tst_ax.plot(
        expt[0, (tstpts['start'] - buffer_timesteps):(tstpts['stop'] + buffer_timesteps), :].mean(axis = 1), 
        'k-', lw = 0.5, alpha = 0.5
    )
    step_ax.plot(
        expt[0, (mainpts['start'] - buffer_timesteps):(mainpts['stop'] + buffer_timesteps), 8], 
        'k-', lw = 0.5, alpha = 0.5
    )
    
tst_ax.set_xlabel('Time (timesteps)')
tst_ax.set_ylabel('V (mV)')

step_ax.set_xlabel('Time (timesteps)')
step_ax.set_ylabel('')

plt.tight_layout()

plt.show()
# -

# Quality control. Remove experiments where $I$ channel wasn't registered correctly. Cells being rejected are plotted, and number of retained cells is printed at the end.

# +
qc_mask = []
for i, rec in enumerate(curr_steps):
    if (np.abs(rec[1, :, :] - np.mean(rec[1, :, :])) < 1.).all() :
        qc_mask.append(False)
        rec.plot()
    else:
        qc_mask.append(True)
        
curr_steps = [curr_steps[i] for i in range(len(curr_steps)) if qc_mask[i]]
print '{} of {} cells passed quality control.'.format(len(curr_steps), len(qc_mask))

del qc_mask
# -

# ## Generate f/I curves
#
# f/I curves are usually rectified linear. However, in some cases non-monotonic f/I curves are observed, usually due to depolarization block.

# Detect spikes in all recordings.
spktimes = [detectSpikes(rec[0, :, :], 0., 3., 0, 0.1) for rec in curr_steps]

spktimes

# +
# Extract f/I data.

# Dict to hold output.
fi_data = {'f': [], 'I': [], 'CV': [], 'rheobase': [], 'freq_at_rheobase': [], 'coeffs': [], 'is_monotonic': []}

# Throwaway function to detect whether an f/I curve increases monotonically.
is_monotonic = lambda x_: np.all(np.nan_to_num(np.diff(x_) / x_[:-1]) > -0.25)

for rec, times in zip(curr_steps, spktimes):
    
    spks_in_window = [x[np.logical_and(x >= mainpts['start'] * dt, x < mainpts['stop'] * dt)] * dt for x in times]
    ISIs_tmp = [np.diff(x) for x in spks_in_window]
    cv_tmp = [x.std() / x.mean() if len(x) > 0 else 0 for x in ISIs_tmp]
    
    f_tmp = np.array(
            [len(x) for x in spks_in_window]
        ) / (1e-3 * dt * (mainpts['stop'] - mainpts['start'])) # Convert to a rate in Hz.
    I_tmp = rec[1, (mainpts['stop'] - 1000):(mainpts['stop'] - 10), :].mean(axis = 0) # Scrape input current.
    
    try:
        rheobase_ind = np.where(f_tmp > 1e-4)[0][0]
        freq_at_rheobase_tmp = f_tmp[rheobase_ind]
    except IndexError:
        rec.plot()
        print f_tmp
        print times
        raise
    
    if is_monotonic(f_tmp):
        coeffs_tmp = np.polyfit(I_tmp[rheobase_ind:], f_tmp[rheobase_ind:], 1)
    else:
        coeffs_tmp = [np.nan for i in range(2)]
        
    fi_data['f'].append(f_tmp)
    fi_data['I'].append(I_tmp)
    fi_data['CV'].append(cv_tmp)
    fi_data['rheobase'].append(I_tmp[rheobase_ind])
    fi_data['freq_at_rheobase'].append(freq_at_rheobase_tmp)
    fi_data['coeffs'].append(coeffs_tmp)
    fi_data['is_monotonic'].append(is_monotonic(f_tmp))
    
fi_data['coeffs'] = np.array(fi_data['coeffs'])
# -

fi_df = pd.DataFrame({
    'rheobase': fi_data['rheobase'], 
    'freq_at_rheobase': fi_data['freq_at_rheobase'],
    'gain': fi_data['coeffs'][:, 0], 
    'is_monotonic': fi_data['is_monotonic']
})
fi_df.to_csv(os.path.join('data', 'processed', 'GABA', 'current_steps_gain.csv'), index=False)

# +
plt.figure(figsize=(1.5, 1))

curves = plt.subplot(111)

legend_flag = False
for x, y, coeffs in zip(fi_data['I'], fi_data['f'], fi_data['coeffs']):
    
    if is_monotonic(y):
        curves.plot(x, y, 'k-', alpha = 0.9)
        
        if not legend_flag:
            curves.plot(
                x[y>1e-3],
                np.polyval(coeffs, x[y>1e-3]),
                'r--',
                alpha = 0.7,
                label = 'Linear fit'
            )
            legend_flag = True
        else:
            curves.plot(x[y>1e-3], np.polyval(coeffs, x[y>1e-3]), 
                             'r--', alpha = 0.7)
    else:
        curves.plot(x, y, '-', color = 'gray', alpha = 0.4)

curves.set_xlim(-50, 155)
curves.set_ylim(-2, 40)

curves.legend(loc='upper left')
curves.set_xlabel('$I$ (pA)')
curves.set_ylabel('$f$ (Hz)')
hide_border('tr', ax=curves, trim=True)

plt.subplots_adjust(top=0.97, right=0.97, bottom=0.3, left=0.25)

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'som_fi_curve_only.png'))
    plt.savefig(os.path.join(IMG_PATH, 'som_fi_curve_only.svg'))

# +
plt.figure(figsize=(2, 1.5))

curves = plt.subplot(111)

legend_flag = False
for x, y, coeffs in zip(fi_data['I'], fi_data['f'], fi_data['coeffs']):
    
    if is_monotonic(y):
        curves.plot(x, y, 'k-', alpha = 0.9)
        
        if not legend_flag:
            curves.plot(
                x[y>1e-3],
                np.polyval(coeffs, x[y>1e-3]),
                'r--',
                alpha = 0.7,
                label = 'Linear fit'
            )
            legend_flag = True
        else:
            curves.plot(x[y>1e-3], np.polyval(coeffs, x[y>1e-3]), 
                             'r--', alpha = 0.7)
    else:
        curves.plot(x, y, '-', color = 'gray', alpha = 0.4)

curves.set_xlim(-50, curves.get_xlim()[1])

curves.legend(loc='upper left')
curves.set_xlabel('$I$ (pA)')
curves.set_ylabel('$f$ (Hz)')
hide_border('tr', ax=curves, trim=True)

plt.subplots_adjust(top=0.97, right=0.97, bottom=0.3, left=0.25)

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'som_fi_curve_only_unscaled.png'))
    plt.savefig(os.path.join(IMG_PATH, 'som_fi_curve_only_unscaled.svg'))
# -

print(
    'Rheobase: {:.3f} +/- {:.3f} pA'.format(
        np.nanmean(fi_data['rheobase']), stats.sem(fi_data['rheobase'], nan_policy='omit')
    )
)
print(
    'Gain: {:.3f} +/- {:.3f} Hz/pA'.format(
        np.nanmean(fi_data['coeffs'][:, 0]), stats.sem(fi_data['coeffs'][:, 0], nan_policy='omit')
    )
)

# Half of the cells have linear monotonic curves. Current steps of cells from the non-monotonic group are below.

for y, rec in zip(fi_data['f'], curr_steps):
    if not is_monotonic(y):
        first_sweep_with_spikes = np.min(np.where(y > 0)[0])
        
        plt.figure()
        spec_tmp = gs.GridSpec(2, 1, height_ratios = [0.2, 1], hspace = 0)
        
        plt.subplot(spec_tmp[0, :])
        plt.plot(
            rec[1, (mainpts['start'] - 2000):(mainpts['stop'] + 2000), first_sweep_with_spikes],
            '-', color = 'gray', lw = 0.5
        )
        plt.plot(
            rec[1, (mainpts['start'] - 2000):(mainpts['stop'] + 2000), -1],
            '-', color = 'gray', lw = 0.5, alpha = 0.6
        )
        
        plt.ylabel('I (pA)')
        
        plt.subplot(spec_tmp[1, :])
        plt.plot(
            rec[0, (mainpts['start'] - 2000):(mainpts['stop'] + 2000), first_sweep_with_spikes],
            'k-', lw = 0.5
        )
        plt.plot(
            rec[0, (mainpts['start'] - 2000):(mainpts['stop'] + 2000), -1],
            'k-', lw = 0.5, alpha = 0.6
        )
        
        plt.ylabel('V (mV)')
        plt.xlabel('Time (timesteps)')
        
        plt.show()

# Usually the cells have non-monotonic f/I curves because they go into depolarization block. In one case, it looks to be due to a weird 'either delayed or immediate' firing phenotype. The last cell went into depol. block because I pushed it harder than the others, which isn't very interesting.
