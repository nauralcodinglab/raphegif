"""
This script extracts the kinetics, activation, and inactivation curves of
subthreshold conductances in 5HT cells.
"""

#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import matplotlib as mpl

import sys
sys.path.append('./analysis/gating/')

import cell_class as ce


#%% READ IN FILES

"""
As of Feb. 5, 2018 all cells are positively identified 5HT neurons.
"""

path = './data/gating/'

rec_fnames = {
        'hyperpol_act':     ['c0_hyperpol_18213022.abf',
                             'c1_hyperpol_18214003.abf',
                             'c2_hyperpol_18214006.abf',
                             'c3_hyperpol_18214009.abf',
                             'c4_hyperpol_18214012.abf',
                             'c5_hyperpol_18214017.abf',
                             'c6_hyperpol_18214023.abf',
                             'c7_hyperpol_18214026.abf'],

        'depol_act':        ['c0_depol_18214000.abf',
                             'c1_depol_18214001.abf',
                             'c2_depol_18214004.abf',
                             'c3_depol_18214007.abf',
                             'c4_depol_18214010.abf',
                             'c5_depol_18214015.abf',
                             'c6_depol_18214018.abf',
                             'c7_depol_18214021.abf'],

        'depol_inact':      ['c0_inact_18201021.abf',
                             'c1_inact_18201029.abf',
                             'c2_inact_18201034.abf',
                             'c3_inact_18201039.abf',
                             'c4_inact_18213011.abf',
                             'c5_inact_18213017.abf',
                             'c6_inact_18213020.abf']

        }

# Create dict to hold loaded recordings.
ce_dict = {}

for cat in rec_fnames.keys():

    ce_dict[cat] = []

    for fname in rec_fnames[cat]:

        ce_dict[cat].append( ce.Cell(fname[:8], V_steps = path + fname) )



#%% INSPECT AND EXTRACT TEST PULSE PARAMS

"""
Note that as of Feb. 5, 2018 all recordings use whole-cell capacitance
correction and Rs compensation. Therefore, R_a measurements are not meaningful.
"""

### Set script parameters

make_plots = True
extract_R_in = True

R_in_baseline_dict = {
        'depol_inact': (750, 1000),
        'hyperpol_act': (1000, 1500),
        'depol_act': (1000, 1500)
        }

R_in_ss_dict = {
        'depol_inact': (3750, 4000),
        'hyperpol_act': (2500, 3000),
        'depol_act': (2500, 3000)
        }

# Create dict to hold R_input.
Rin_dict = {}


for cat in ce_dict.keys():

    Rin_dict[cat] = []

    R_in_baseline = R_in_baseline_dict[cat]
    R_in_ss = R_in_ss_dict[cat]

    cnt = 0
    for cell_ in ce_dict[cat]:

        if make_plots:
            cell_.V_steps[0].plot()
            plt.suptitle('{} - cell {}'.format(cat, cnt))
            plt.subplots_adjust(top = 0.85)

        if extract_R_in:
            Rin_dict[cat].append(
                    cell_.V_steps[0].fit_test_pulse(
                            R_in_baseline, R_in_ss,
                            verbsose = False
                            )['R_input'].mean())

        cnt += 1

    if make_plots:

        plt.figure()
        plt.subplot(111)
        plt.title('{} - R_input'.format(cat))
        plt.plot(Rin_dict[cat], 'k.')
        plt.ylabel('R_input (MOhm)')
        plt.xlabel('Cell no.')

del (R_in_baseline, R_in_ss, cat, cell_, make_plots, extract_R_in)


#%% REMOVE CRAPPY CELLS

crappy_cell_dict = {
        'depol_inact': None,
        'hyperpol_act': None,
        'depol_act': {3, 4}
        }

for cat in ce_dict.keys():

    if crappy_cell_dict[cat] is not None:

        inds = set([j for j in range(len(ce_dict[cat]))])
        inds -= crappy_cell_dict[cat]

        ce_dict[cat] = [i for j, i in enumerate(ce_dict[cat]) if j in inds]


#%% EXTRACT CONDUCTANCE CURVES

"""
This block subtracts leak conductance from current traces and converts current
to conductance (assuming the current is mediated by K).

Voltage-conductance relationships for I_A (peak current and inactivation) and
delayed rectifier (activation only) are extracted and placed in `cond_vals` and
`V_vals` dicts. Each measurement category is stored separately in the dict.
Measurements of the same category are stored in np.arrays with dimensionality
[V, cell].
"""

### Set script parameters.

V_channel = 1
I_channel = 0

normalize = True
truncate = 13 # Only include this number of V_values. Set to None to skip.

baseline_range = (750, 1000)
prepulse_bl_range = {
        'depol_inact': (2350, 2600),
        'depol_act': (30000, 31000)
        }
presumed_reversal = -101 # Reversal for K.

make_plots = {
        'conductance': False,
        'I': False,
        'curve': True
        }

ss_range = {
        'depol_inact': (54750, 55000),
        'depol_act': (42000, 43000)
        }
peak_range = {
        'depol_inact': (56130, 56140),
        'depol_act': (33100, 33200)
        }


### Allocate arrays for output

cond_traces = []

if truncate is None:
    ss_no_pts = ce_dict['depol_act'][0].V_steps[0].shape[2]
    peak_no_pts = ce_dict['depol_act'][0].V_steps[0].shape[2]
    inact_no_pts = ce_dict['depol_inact'][0].V_steps[0].shape[2]
else:
    ss_no_pts = truncate
    peak_no_pts = truncate
    inact_no_pts = truncate


cond_vals = {
        'steady_state': np.empty((ss_no_pts,
                                  len(ce_dict['depol_act']))),
        'peak': np.empty((peak_no_pts,
                                  len(ce_dict['depol_act']))),
        'inactivation': np.empty((inact_no_pts,
                                  len(ce_dict['depol_inact'])))
        }


V_vals = {
        'steady_state': np.empty((ss_no_pts,
                                  len(ce_dict['depol_act']))),
        'peak': np.empty((peak_no_pts,
                                  len(ce_dict['depol_act']))),
        'inactivation': np.empty((inact_no_pts,
                                  len(ce_dict['depol_inact'])))
        }


### Main

for cat in ['depol_inact', 'depol_act']:

    for i in range(len(ce_dict[cat])):

        # Make local copy of recording and params.
        cell_tmp = ce_dict[cat][i].V_steps[0].copy()
        R_in = Rin_dict[cat]

        # Remove Ohmic component.
        baseline_V = cell_tmp[V_channel, slice(*baseline_range), :].mean()
        delta_V_b = cell_tmp[V_channel, :, :] - baseline_V

        delta_I = 1000 * delta_V_b / R_in[i]

        cell_tmp[I_channel, :, :] -= delta_I

        # Subtract baselines, using pre-pulse as baseline.
        baseline_I = cell_tmp[I_channel, slice(*prepulse_bl_range[cat]), :]
        baseline_I = baseline_I.mean(axis = 0)
        cell_tmp[I_channel, :, :] -= baseline_I

        if make_plots['I']:
            plt.figure(figsize = (10, 5))
            plt.subplot(111)
            plt.title('Leak-subtracted currents - {} {}'.format(cat, i))

            plt.plot(cell_tmp[I_channel, :, :], 'k-', linewidth = 0.5, alpha = 0.3)

            plt.ylabel('I (pA)')
            plt.xlabel('Time (steps)')
            plt.tight_layout()
            plt.show()

        # Convert current measurements to conductance.
        # That is, divide out changes in voltage relative to presumed reversal.
        delta_V_con = cell_tmp[V_channel, :, :] - presumed_reversal

        conductance = ce.Recording(cell_tmp[I_channel, :, :][np.newaxis, :, :])
        conductance[0, :, :] /= delta_V_con

        # Save conductance traces.
        cond_traces.append(conductance)

        if make_plots['conductance']:
            conductance.plot()

        if cat == 'depol_act':

            # Extract V-dependence of steady-state current.
            ss_con = conductance[0, slice(*ss_range[cat]), :].mean(axis = 0)
            ss_V = cell_tmp[V_channel, slice(*ss_range[cat]), :].mean(axis = 0)

            peak_con = conductance[0, slice(*peak_range[cat]), :].mean(axis = 0)
            peak_V = cell_tmp[V_channel, slice(*peak_range[cat]), :].mean(axis = 0)


            if truncate is not None:

                ss_con = ss_con[:truncate]
                ss_V = ss_V[:truncate]

                peak_con = peak_con[:truncate]
                peak_V = peak_V[:truncate]

            if normalize:

                ss_con -= ss_con.min()
                ss_con /= ss_con.max()

                peak_con -= peak_con.min()
                peak_con /= peak_con.max()


            # Save results.
            cond_vals['steady_state'][:, i] = ss_con
            V_vals['steady_state'][:, i] = ss_V

            cond_vals['peak'][:, i] = peak_con
            V_vals['peak'][:, i] = peak_V

        if cat == 'depol_inact':

            # Extract V-dependence of steady-state current.
            ss_con = conductance[0, slice(*ss_range[cat]), :].mean(axis = 0)
            ss_V = cell_tmp[V_channel, slice(*ss_range[cat]), :].mean(axis = 0)

            inac_con = conductance[0, slice(*peak_range[cat]), :].mean(axis = 0)
            inac_con -= ss_con
            inac_V = ss_V

            if truncate is not None:

                inac_con = inac_con[:truncate]
                inac_V = inac_V[:truncate]

            if normalize:

                inac_con -= inac_con.min()
                inac_con /= inac_con.max()

            cond_vals['inactivation'][:, i] = inac_con
            V_vals['inactivation'][:, i] = inac_V


        if make_plots['curve']:

            plt.figure(figsize = (7, 7))
            plt.subplot(111)

            plt.plot(ss_V, ss_con, '.',
                     label = 'Steady-state conductance')
            plt.plot(peak_V, peak_con, '.',
                     label = 'Peak conductance (activation)')
            plt.plot(inac_V, inac_con, '.',
                     label = 'Peak conductance (inactivation)')

            plt.ylabel('Conductance')
            plt.xlabel('Vm (mV)')
            plt.legend()
            plt.tight_layout()
            plt.show()



#%% DEFINE FITTING FUNCTIONS

"""
This block defines functions to be used in fitting activation/inactivation
curves.
"""

# Sigmoid curve for fitting activation curves.
def sigmoid_curve(p, V):

    """Three parameter logit.

    p = [A, k, V0]

    y = A / ( 1 + exp(-k * (V - V0)) )
    """

    if len(p) != 3:
        raise ValueError('p must be vector-like with len 3.')

    A = p[0]
    k = p[1]
    V0 = p[2]

    return A / (1 + np.exp(-k * (V - V0)))

# Single exponential for fitting activation curves.
def exp_curve(p, V):

    """Three parameter exponential curve.

    p = [A, k, V0]

    y = A * np.exp(k * (V - V0))
    """

    if len(p) != 3:
        raise ValueError('p must be vector-like with len 3.')

    A = p[0]
    k = p[1]
    V0 = p[2]

    return A * np.exp(k * (V - V0))


# Multiexponential for fitting kinetics.
def multiexp_curve(p, X):

    """N three-parameter exponentials and offset.

    p = [B, A_1, k_1, x0_1, A_2, k_2, x0_2, ...]

    Y = sum( A_i * exp(k_i * (V - x0_i)) ) + B
    """

    if type(X) is not np.ndarray:
        raise TypeError('X must be provided as a numpy array.')

    if (len(p) - 1) % 3 != 0:
        raise ValueError('len p - 1 % 3 must be 0.')

    B = p[0]
    coeffs = p[1:]

    # Variable to hold sum of exponential term value.
    exp_terms = np.zeros(X.shape, dtype = np.float64)

    # Iterate over three-parameter exponential terms.
    for i in range(0, len(coeffs), 3):

        A_i = coeffs[i]
        k_i = coeffs[i + 1]
        x0_i = coeffs[i + 2]

        exp_terms += A_i * np.exp(k_i * (X - x0_i))

    return exp_terms + B

"""
(1 - A exp(-t/tau)) - A exp(-t/tau)
"""


# General function for computing residuals.
def compute_residuals(p, func, Y, X):

    """Compute residuals of a fitted curve.

    Inputs:
        p       -- vector of function parameters
        func    -- a callable function
        Y       -- real values
        X       -- vector of points on which to compute fitted values

    Returns:
        Array of residuals.
    """

    if len(Y) != len(X):
        raise ValueError('Y and X must be of the same length.')

    Y_hat = func(p, X)

    return Y - Y_hat


#%% FIT CURVES

"""This block fits activation/inactivation curves.
"""

### Set script parameters.
verbose = True
funcs_to_use = {
        'steady_state': sigmoid_curve,
        'peak': sigmoid_curve,
        'inactivation': sigmoid_curve
        }
p0 = {
      'steady_state': [12, 1, -25],
      'peak': [12, 1, -30],
      'inactivation': [12, -1, -60]}

### Create dicts to hold outputs.
fitted_params = {}
fitted_points = {}


### Main
for key in V_vals.keys():

    V = np.broadcast_to(V_vals[key].mean(axis = 1)[:, np.newaxis],
                        V_vals[key].shape)
    V = V.flatten()
    Y = cond_vals[key].flatten()

    p = opt.least_squares(compute_residuals, p0[key], kwargs = {
            'func': funcs_to_use[key], 'X': V, 'Y': Y
            })['x']
    fitted = funcs_to_use[key](p, V)

    fitted_params[key] = p
    fitted_points[key] = fitted

    if verbose:
        print('\n\nFitted params for {} using {}:\n'
              'A = {:0.2f} \nk = {:0.2f} \nV0 = {:0.2f}'.format(
                      key, funcs_to_use[key].__name__, p[0], p[1], p[2]))



#%% PLOT FITTED CONDUCTANCE CURVES

"""This block makes a pretty plot of the raw conductance data, its average,
and the fitted curves.

Activation/inactivation/delayed rectifier curves.
"""

### Set script parameters

plt.figure(figsize = (12, 7))

# Activation subplot.
plt.subplot(131)
plt.title('I_A activation')
x = np.broadcast_to(V_vals['peak'].mean(axis = 1)[:, np.newaxis],
                    V_vals['peak'].shape)
plt.plot(x,
         cond_vals['peak'],
         'ko', markerfacecolor = 'none', alpha = 0.5)
plt.plot(x.mean(axis = 1),
         cond_vals['peak'].mean(axis = 1),
         'ro', label = 'Mean')
x = np.arange(-65, 10, 0.1)
plt.plot(x,
         funcs_to_use['peak'](
                 fitted_params['peak'],
                 x),
         'b-',
         linewidth = 0.5,
         label = 'Fitted')
plt.ylabel('Conductance (pA/mV)')
plt.xlabel('Vm (mV)')
plt.legend()

# Inactivation subplot.
plt.subplot(132)
plt.title('I_A inactivation')
x = np.broadcast_to(V_vals['inactivation'].mean(axis = 1)[:, np.newaxis],
                    V_vals['inactivation'].shape)
plt.plot(x,
         cond_vals['inactivation'],
         'ko', markerfacecolor = 'none', alpha = 0.5)
plt.plot(x.mean(axis = 1),
         cond_vals['inactivation'].mean(axis = 1),
         'ro', label = 'Mean')
x = np.arange(-90, -17, 0.1)
plt.plot(x,
         funcs_to_use['inactivation'](
                 fitted_params['inactivation'],
                 x),
         'b-',
         linewidth = 0.5,
         label = 'Fitted')
plt.ylabel('Conductance (pA/mV)')
plt.xlabel('Vm (mV)')
plt.legend()

# Steady-state subplot.
plt.subplot(133)
plt.title('Delayed rectifier')
x = np.broadcast_to(V_vals['steady_state'].mean(axis = 1)[:, np.newaxis],
                    V_vals['steady_state'].shape)
plt.plot(x,
         cond_vals['steady_state'],
         'ko', markerfacecolor = 'none', alpha = 0.5)
plt.plot(x.mean(axis = 1),
         cond_vals['steady_state'].mean(axis = 1),
         'ro', label = 'Mean')
x = np.arange(-65, 10, 0.1)
plt.plot(x,
         funcs_to_use['steady_state'](
                 fitted_params['steady_state'],
                 x),
         'b-',
         linewidth = 0.5,
         label = 'Fitted')
plt.ylabel('Conductance (pA/mV)')
plt.xlabel('Vm (mV)')
plt.legend()

plt.tight_layout()


#%%

plt.figure()

for key in ['steady_state', 'inactivation', 'peak']:
    plt.plot(x.mean(axis = 1),
             funcs_to_use[key](
                     fitted_params[key],
                     x))

#%% KINETIC FITTING FUNCTIONS

"""Hard to get a good fit, esp. on trough. Previous work has experimentally
separated Ia and Ik before fitting.
"""

exp_fit_range = (26500, 30000)
p0 = [0, 1, -1/50, 2633, 1, 1/500, 3000]

p_fitted = []

plt.figure()

for i in range(len(cond_traces)):

    X = np.arange(exp_fit_range[0]/10, exp_fit_range[1]/10, 0.1)
    Y = cond_traces[i][0, slice(*exp_fit_range), -1]
    p_fitted.append(opt.least_squares(compute_residuals, p0, kwargs = {
            'func': multiexp_curve, 'X': X, 'Y': Y
            })['x'])

    plt.plot(X, multiexp_curve(p_fitted[i], X),
             color = mpl.cm.viridis(i / len(cond_traces)),
             linewidth = 1, alpha = 0.5,
             label = i)
    plt.plot(X, Y,
             color = mpl.cm.viridis(i / len(cond_traces)),
             linewidth = 0.5, alpha = 0.5)

plt.legend()
