#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

#%%

no_trials = 30
no_syn = 2

outcomes = {
    'no_trials': [],
    'no_syn': [],
    'no_releases': [],
    'no_failures': []
}

for no_syn in no_syn_ls:
    p_release = np.random.uniform(size = no_syn)
    no_releases = []
    for trial in range(no_trials):
        no_releases.append(np.sum(np.random.uniform(size = no_syn) < p_release))
    no_releases = np.array(no_releases)
    no_failures = np.sum(no_releases == 0)

    outcomes['no_trials'].append(no_trials)
    outcomes['no_syn'].append(no_syn)
    outcomes['no_releases'].append(no_releases)
    outcomes['no_failures'].append(no_failures)

outcomes = pd.DataFrame(outcomes)


outcomes

#%%

def poisson(x, no_neurons):
    output = []
    for x_ in x:
        output.append(
            np.exp(-no_neurons / 2.) * (no_neurons / 2)**x_ / np.math.factorial(x_)
        )
    return output

#%%

# Simulation parameters.
no_neurons = 10
no_reps = 10000

# Perform simulations.
f = [] # List to hold no. of failures per trial.
for i in range(no_reps):
    pf = np.random.uniform(size = no_neurons)                   # Pick uniform fail prob.
    f.append(np.sum(np.random.uniform(size = no_neurons) < pf)) # Fail randomly.

def binom(x, no_neurons):
    output = []
    for x_ in x:
        binom_coeff = (np.math.factorial(no_neurons)
            / (np.math.factorial(no_neurons - x_) * np.math.factorial(x_)))
        output.append(
            binom_coeff * 0.5**no_neurons
        )
    return output

plt.figure()
plt.subplot(121)
plt.title('Failure distribution')
plt.bar(
    np.arange(0, no_neurons + 1),
    binom(np.arange(0, no_neurons + 1), no_neurons),
    color = 'k', label = 'Binomial model'
)
plt.hist(
    f,
    color = 'blue', alpha = 0.7, density = True, label = 'Simulations',
    bins = np.arange(-0.5, no_neurons + 0.6, 1)
)
plt.ylabel('Density')
plt.xlabel('No. failures')
plt.legend()
plt.subplot(122)
plt.title('Sample pf distribution')
plt.hist(pf)
plt.xlim(0, 1)
plt.xlabel('P failure')
plt.ylabel('Count')

plt.tight_layout()

plt.savefig('/Users/eharkin/Desktop/synrelease/failures.png', dpi = 300)

plt.show()

#%%

def binom_coeff(N, k):
    return np.float(np.math.factorial(N)) / (np.math.factorial(N - k) * np.math.factorial(k))

def p_all_failures_onetrial(no_neurons):
    return binom_coeff(no_neurons, no_neurons) * 0.5**no_neurons

def p_all_failures(no_neurons, no_trials):
    """
    Probability of observing 50% failures post-synaptically given no_neurons over no_trials.
    """
    output = binom_coeff(no_trials, no_trials//2)
    output *= p_all_failures_onetrial(no_neurons) ** (no_trials//2)
    output *= (1 - p_all_failures_onetrial(no_neurons)) ** (no_trials//2)
    return output

pallf = []
for i in range(10):
    pallf.append(p_all_failures(i, 10))
pallf = np.array(pallf)

plt.figure()
plt.subplot(121)
plt.title('Model ')
plt.bar(np.arange(10), pallf / pallf.sum())
plt.xlim(0, 4.5)
plt.show()

rand_draws = np.digitize(np.random.uniform(size = 30), np.cumsum(pallf / pallf.sum()))
