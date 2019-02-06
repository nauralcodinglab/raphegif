#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

from Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from AugmentedGIF import AugmentedGIF

#%%

class SplineGIF(AugmentedGIF):

    def __init__(self, dt=0.1):

        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)

        # Define model parameters

        self.gl      = 0.001        # nS, leak conductance
        self.C       = 0.1     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential

        self.Vr      = -50.0            # mV, voltage reset
        self.Tref    = 4.0              # ms, absolute refractory period

        self.Vt_star = -48.0            # mV, steady state voltage threshold VT*
        self.DV      = 0.5              # mV, threshold sharpness
        self.lambda0 = 1.0              # by default this parameter is always set to 1.0 Hz


        self.eta     = Filter_Rect_LogSpaced()    # nA, spike-triggered current (must be instance of class Filter)
        self.gamma   = Filter_Rect_LogSpaced()    # mV, spike-triggered movement of the firing threshold (must be instance of class Filter)


        # Initialize the spike-triggered current eta with an exponential function

        def expfunction_eta(x):
            return 0.2*np.exp(-x/100.0)

        self.eta.setFilter_Function(expfunction_eta)


        # Initialize the spike-triggered current gamma with an exponential function

        def expfunction_gamma(x):
            return 0.01*np.exp(-x/100.0)

        self.gamma.setFilter_Function(expfunction_gamma)


        # Variables related to fitting procedure

        self.avg_spike_shape = 0
        self.avg_spike_shape_support = 0
        self.var_explained_dV = 0
        self.var_explained_V = 0

        # Parameters related to potassium conductances
        self.m_coeffs = (4.469e-2, 8.973e2, -2.349e-4, 1.468e-5, 3.469e-4, -1.171e-4)
        self.m_nodes = (-60, -40)
        self.h_coeffs = (-6.495e-2, 9.448e-3, -1.267e-3, 7.340e-6, 2.715e-4, 1.316e-5)
        self.h_nodes = (-60, -40)
        self.h_tau = 70

        self.n_coeffs = (-1.043e-1, 9.824e2, -3.427e-4, 1.274e-5, -3.828e-4, -8.303e-6)
        self.n_nodes = (-60, -40)
        self.n_tau = 1

        self.E_K = -101

        self.gbar_K1 = 0.010
        self.gbar_K2 = 0.001

    @staticmethod
    def _expand_basis(X, nodes, order = [2, 3], flip_X = False):

        if not flip_X:
            softplus_col = np.log(1 + 1.1**(X - 50))
        else:
            softplus_col = np.log(1 + 1.1**(-(X - 50)))

        X_expanded = [np.ones_like(X), softplus_col]

        for i, node in enumerate(nodes):
            for j in order:
                if not flip_X:
                    X_expanded.append(np.clip(X - node, 0, None) ** j)
                else:
                    X_expanded.append(np.clip(-(X-node), 0, None) ** j)

        return np.array(X_expanded).T

    def mInf(self, V):

        """Compute the equilibrium activation gate state of the potassium conductance.
        """

        minf_raw = np.dot(self._expand_basis(V, self.m_nodes, flip_X = False), self.m_coeffs)
        minf_bounded = np.clip(minf_raw, None, 1)

        return minf_bounded


    def hInf(self, V):

        """Compute the equilibrium state of the inactivation gate of the potassium conductance.
        """

        hinf_raw = np.dot(self._expand_basis(V, self.h_nodes, flip_X = True), self.h_coeffs)
        hinf_bounded = np.clip(hinf_raw, None, 1)

        return hinf_bounded


    def nInf(self, V):

        """Compute the equilibrium state of the non-inactivating conductance.
        """

        ninf_raw = np.dot(self._expand_basis(V, self.n_nodes, flip_X = False), self.n_coeffs)
        ninf_bounded = np.clip(ninf_raw, None, 1)

        return ninf_bounded

    ### Overwrite unimplemented fitting/simulation methods

    def simulate(self):
        raise NotImplementedError

    def simulateDeterministic_forceSpikes(self):
        raise NotImplementedError

    def fitSubthresholdDynamics(self):
        raise NotImplementedError


#%% MAKE TEST PLOT

if __name__ == '__main__':

    test = SplineGIF()
    V_tmp = np.arange(-100, -10, 0.1)
    plt.figure(figsize = (6, 4))
    plt.subplot(111)
    plt.title('Gating functions implemented in SplineGIF')
    plt.axvspan(-80, -20, facecolor = 'gray', alpha = 0.25, edgecolor = 'None')
    plt.axhline(0, color = 'k', lw = 0.5)
    plt.plot(V_tmp, test.mInf(V_tmp), 'b-', label = 'IA activation')
    plt.plot(V_tmp, test.hInf(V_tmp), 'g-', label = 'IA inactivation')
    plt.plot(V_tmp, test.nInf(V_tmp), 'r-', label = 'Kslow activation')
    plt.ylabel(r'$\frac{g}{g_{ref}}$')
    plt.xlabel('$V$ (mV)')
    plt.legend()
    plt.tight_layout()

    plt.savefig('./figs/ims/gating/' + 'SplineGIF_gating.png', dpi = 300)

    plt.show()
