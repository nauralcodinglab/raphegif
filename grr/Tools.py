import sys
import os

import numpy as np
from scipy.optimize import leastsq
import weave


###########################################################
# Remove axis
###########################################################

def removeAxis(ax, which_ax=['top', 'right']):

    for loc, spine in ax.spines.iteritems():
        if loc in which_ax:
            spine.set_color('none')  # don't draw spine

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


###########################################################
# Reprint
###########################################################
def reprint(str):
    sys.stdout.write('%s\r' % (str))
    sys.stdout.flush()


class gagProcess(object):
    """Class to forcibly gag verbose methods.

    Temporarily redirects stdout to block print commands.

    Usage:

    with gagProcess:
        print 'Things that will not be printed.'
    """

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


###########################################################
# Generate Ornstein-Uhlenbeck process
###########################################################

def generateOUprocess(T=10000.0, tau=3.0, mu=0.0, sigma=1.0, dt=0.1, random_seed=42):
    """
    Generate an Ornstein-Uhlenbeck (stationnary) process with:
    - mean mu
    - standard deviation sigma
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    T_ind = int(T/dt)

    if random_seed is not None:
        np.random.seed(random_seed)
    white_noise = np.random.randn(T_ind)
    white_noise = white_noise.astype("double")

    OU_process = np.zeros(T_ind)
    OU_process[0] = mu
    OU_process = OU_process.astype("double")

    code = """

            #include <math.h>

            int cT_ind    = int(T_ind);
            float cdt     = float(dt);
            float ctau    = float(tau);
            float cmu     = float(mu);
            float csigma  = float(sigma);

            float OU_k1 = cdt / ctau ;
            float OU_k2 = sqrt(2.0*cdt/ctau) ;

            for (int t=0; t < cT_ind-1; t++) {
                OU_process[t+1] = OU_process[t] + (cmu - OU_process[t])*OU_k1 +  csigma*OU_k2*white_noise[t] ;
            }

            """

    vars = ['T_ind', 'dt', 'tau', 'sigma', 'mu', 'OU_process', 'white_noise']
    v = weave.inline(code, vars)

    return OU_process


def generateOUprocess_sinSigma(f=1.0, T=10000.0, tau=3.0, mu=0.0, sigma=1.0, delta_sigma=0.5, dt=0.1):
    """
    Generate an Ornstein-Uhlenbeck process with time dependent standard deviation:
    - mean mu
    - sigma(t) = sigma*(1+delta_sigma*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=1.0, dt=dt)
    t = np.arange(len(OU_process))*dt

    sin_sigma = sigma*(1+delta_sigma*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process*sin_sigma + mu

    return I


def generateOUprocess_sinMean(f=1.0, T=10000.0, tau=3.0, mu=0.2, delta_mu=0.5, sigma=1.0, dt=0.1):
    """
    Generate an Ornstein-Uhlenbeck process with time dependent mean:
    - sigma
    - mu(t) = mu*(1+delta_mu*sin(2pift)), f in Hz
    - temporal correlation tau (ms)
    The duration of the signal is specified by the input parameter T (in ms).
    The process is generated in discrete time with temporal resolution dt (in ms)
    """

    OU_process = generateOUprocess(T=T, tau=tau, mu=0.0, sigma=sigma, dt=dt)
    t = np.arange(len(OU_process))*dt

    sin_mu = mu*(1+delta_mu*np.sin(2*np.pi*f*t*10**-3))

    I = OU_process + sin_mu

    return I


###########################################################
# Functin to convert spike times in spike indices
###########################################################
def timeToIndex(x_t, dt):

    x_t = np.array(x_t)
    x_i = np.array([int(np.round(s/dt)) for s in x_t])
    x_i = x_i.astype('int')

    return x_i


def timeToIntVec(x_t, T, dt):
    """Convert vector of timestamps to a vector of zeros and ones.

    Inputs:
        x_t -- Vector of timestamps
        T   -- Total length of integer vector output (time units)
        dt  -- Timestep width
    """

    x_i = timeToIndex(x_t, dt)
    intvec = np.zeros(int(T / dt), dtype=np.int8)
    intvec[x_i] = 1

    return intvec


###########################################################
# Functions to perform exponential fit
###########################################################

def multiExpEval(x, bs, taus):

    result = np.zeros(len(x))
    L = len(bs)

    for i in range(L):
        result = result + bs[i] * np.exp(-x/taus[i])

    return result


def multiExpResiduals(p, x, y, d):
    bs = p[0:d]
    taus = p[d:2*d]

    return (y - multiExpEval(x, bs, taus))


def fitMultiExpResiduals(bs, taus, x, y):
    x = np.array(x)
    y = np.array(y)
    d = len(bs)
    p0 = np.concatenate((bs, taus))
    plsq = leastsq(multiExpResiduals, p0, args=(x, y, d), maxfev=100000, ftol=0.00000001)
    p_opt = plsq[0]
    bs_opt = p_opt[0:d]
    taus_opt = p_opt[d:2*d]

    fitted_data = multiExpEval(x, bs_opt, taus_opt)

    ind = np.argsort(taus_opt)

    taus_opt = taus_opt[ind]
    bs_opt = bs_opt[ind]

    return (bs_opt, taus_opt, fitted_data)


###########################################################
# Get indices far from spikes
###########################################################

def getIndicesFarFromSpikes(T, spikes_i, dt_before, dt_after, initial_cutoff, dt):

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_before_i = int(dt_before/dt)
    dt_after_i = int(dt_after/dt)

    for s in spikes_i:
        flag[max(s-dt_before_i, 0): min(s+dt_after_i, T_i)] = 1

    selection = np.where(flag == 0)[0]

    return selection


def getIndicesDuringSpikes(T, spikes_i, dt_after, initial_cutoff, dt):

    T_i = int(T/dt)
    flag = np.zeros(T_i)
    flag[:int(initial_cutoff/dt)] = 1
    flag[-1] = 1

    dt_after_i = int(dt_after/dt)

    for s in spikes_i:
        flag[max(s, 0): min(s+dt_after_i, T_i)] = 1

    selection = np.where(flag > 0.1)[0]

    return selection


def PSTH(spktrain, window_width, no_neurons, dt=0.1):
    """
    Obtain the population firing rate with a resolution of `window_width`.
    """

    window_width *= 1e-3
    dt *= 1e-3

    kernel = np.ones(int(window_width / dt)) / (window_width * no_neurons)
    psth = np.convolve(spktrain, kernel, 'same')
    return psth