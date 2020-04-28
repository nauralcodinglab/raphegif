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
DATA_PATH = os.path.join('data', 'raw', '5HT', 'current_steps')

fnames = pd.read_csv(os.path.join(DATA_PATH, 'index.csv'))
fnames
# -

curr_steps = Cell().read_ABF([os.path.join(DATA_PATH, fn) for fn in fnames['Steps']])
curr_steps[0].plot()

for expt in curr_steps:
    plt.plot(expt[1, :, 0])
plt.show()

# Current step recordings have a similar structure, but differ in number of sweeps. Also possibly in spacing of current steps.
#
# Automatically detect the start/end of the test pulse and current steps based on the first recording and then show whether this works for all cells.

# +
change_threshold = 6. # pA threshold at which to detect a step.

tstpts = {'start': [], 'stop': []}
mainpts = {'start': [], 'stop': []}

for expt in curr_steps:
    try:
        falling = np.where(np.diff(expt[1, :, 0]) < -change_threshold)[0]
        tstpts['start'].append(falling[0])
        mainpts['start'].append(falling[1])
        
        rising = np.where(np.diff(expt[1, :, 0]) > change_threshold)[0]
        tstpts['stop'].append(rising[0])
        mainpts['stop'].append(rising[1])
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

for i, expt in enumerate(curr_steps):
    tst_ax.plot(
        expt[0, (tstpts['start'][i] - buffer_timesteps):(tstpts['stop'][i] + buffer_timesteps), :].mean(axis = 1), 
        'k-', lw = 0.5, alpha = 0.5
    )
    step_ax.plot(
        expt[0, (mainpts['start'][i] - buffer_timesteps):(mainpts['stop'][i] + buffer_timesteps), 8], 
        'k-', lw = 0.5, alpha = 0.5
    )
    
tst_ax.set_xlabel('Time (timesteps)')
tst_ax.set_ylabel('V (mV)')

step_ax.set_xlabel('Time (timesteps)')
step_ax.set_ylabel('')

plt.tight_layout()

plt.show()

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

for i, (rec, times) in enumerate(zip(curr_steps, spktimes)):
    
    spks_in_window = [
        x[np.logical_and(x >= mainpts['start'][i] * dt, x < mainpts['stop'][i] * dt)] * dt for x in times
    ]
    ISIs_tmp = [np.diff(x) for x in spks_in_window]
    cv_tmp = [x.std() / x.mean() if len(x) > 0 else 0 for x in ISIs_tmp]
    
    f_tmp = np.array(
            [len(x) for x in spks_in_window]
        ) / (1e-3 * dt * (mainpts['stop'][i] - mainpts['start'][i])) # Convert to a rate in Hz.
    I_tmp = rec[1, (mainpts['stop'][i] - 1000):(mainpts['stop'][i] - 10), :].mean(axis = 0) # Scrape input current.
    
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

fi_df = pd.DataFrame(
    {'rheobase': fi_data['rheobase'], 'freq_at_rheobase': fi_data['freq_at_rheobase'], 'gain': fi_data['coeffs'][:, 0], 'is_monotonic': fi_data['is_monotonic']}
)
fi_df.to_csv(os.path.join('data', 'processed', '5HT', 'current_steps_gain.csv'), index=False)

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

curves.set_xlim(-50, 170)
curves.set_ylim(-2, 40)

curves.legend(loc='upper left')
curves.set_xlabel('$I$ (pA)')
curves.set_ylabel('$f$ (Hz)')
hide_border('tr', ax=curves, trim=True)

plt.subplots_adjust(top=0.97, right=0.97, bottom=0.3, left=0.25)

if IMG_PATH is not None:
    plt.savefig(os.path.join(IMG_PATH, 'ser_fi_curve_only.png'))
    plt.savefig(os.path.join(IMG_PATH, 'ser_fi_curve_only.svg'))
    

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
    plt.savefig(os.path.join(IMG_PATH, 'ser_fi_curve_only_unscaled.png'))
    plt.savefig(os.path.join(IMG_PATH, 'ser_fi_curve_only_unscaled.svg'))
    
# -

print(
    'Rheobase: {:.3f} +/- {:.3f} pA'.format(
        np.nanmean(fi_data['rheobase']), stats.sem(fi_data['rheobase'], nan_policy='omit')
    )
)
print(
    'Gain: {:.4f} +/- {:.4f} Hz/pA'.format(
        np.nanmean(fi_data['coeffs'][:, 0]), stats.sem(fi_data['coeffs'][:, 0], nan_policy='omit')
    )
)
