#%% IMPORT MODULES

from __future__ import division

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

from src.GIF import GIF
from src.Tools import reprint


#%% DEFINE resGIF CLASS

class resGIF(GIF):

    def __init__(self, dt = 0.1):

        super(resGIF, self).__init__(dt = dt)

        self.gw = 0.001     # Magnitude of resonating current.
        self.Ew = -70.      # Reversal of resonating current (mV).
        self.tau_w = 50.    # Time constant of resonating current (ms).

        self.fit_all_tau_w = None   # List with all tau_w tried during fitting.
        self.fit_all_gw = None      # List of estimated gw for each tau_w.
        self.fit_all_r2 = None      # Coefficient of determination for all tau_w tried.

    def simulate(self, I, V0):
        raise NotImplementedError

    def simulateDeterministic_forceSpikes(self, I, V0, spks):
        raise NotImplementedError

    def compute_w(self, V, tau_w = 'default'):
        """Compute filtered resonating current w over voltage vector V.

        Inputs:
            V -- numeric vector
                Voltage timeseries over which to compute w.
            tau_w -- float or 'default'
                Time constant to use to compute w. Uses self.tau_w if set to 'default'.
        """

        if tau_w == 'default':
            tau_w = self.tau_w

        w = np.empty_like(V, dtype = np.float64)
        w[0] = V[0] - self.Ew

        for t in range(1, len(V)):
            dw = ((V[t-1] - self.Ew) - w[t-1]) / tau_w
            w[t] = w[t-1] + dw * self.dt

        return w

    def fitSubthresholdDynamics(self, experiment, DT_beforeSpike=5.0, tau_w_all = 'default', Vmin = None, plot = False):

        """
        Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        experiment: Experiment object on which the model is fitted.
        DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
        """

        if tau_w_all == 'default':
            tau_w_all = np.logspace(np.log2(1.), np.log2(300.), 15, base = 2)

        self.tau_w_all = tau_w_all

        print "\nresGIF MODEL - Fit subthreshold dynamics..."

        # Expand eta in basis functions
        self.dt = experiment.dt

        # Instantiate objects to hold output.
        b_ls = [] # List to hold regression coeffs. Retain set with highest R2 at the end.
        self.fit_all_tau_w = tau_w_all
        self.fit_all_gw = []
        self.fit_all_r2 = []

        # Try each tau_w in tau_w_all.
        for tau_w in tau_w_all:

            print "\nTrying tau_w = {:.2f}ms".format(tau_w)

            # Build X matrix and Y vector to perform linear regression (use all traces in training set)
            # For each training set an X matrix and a Y vector is built.
            ####################################################################################################
            X = []
            Y = []

            cnt = 0

            for tr in experiment.trainingset_traces :

                if tr.useTrace :

                    cnt += 1
                    reprint( "Compute X matrix for repetition %d" % (cnt) )

                    # Compute the the X matrix and Y=\dot_V_data vector used to perform the multilinear linear regression (see Eq. 17.18 in Pozzorini et al. PLOS Comp. Biol. 2015)
                    (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(
                        tr, DT_beforeSpike=DT_beforeSpike, tau_w = tau_w
                    )

                    X.append(X_tmp)
                    Y.append(Y_tmp)


            # Concatenate matrixes associated with different traces to perform a single multilinear regression
            ####################################################################################################
            if cnt == 1:
                X = X[0]
                Y = Y[0]

            elif cnt > 1:
                X = np.concatenate(X, axis=0)
                Y = np.concatenate(Y, axis=0)

            else :
                print "\nError, at least one training set trace should be selected to perform fit."


            # Perform linear Regression defined in Eq. 17 of Pozzorini et al. PLOS Comp. Biol. 2015
            ####################################################################################################

            print "\nPerform linear regression..."
            XTX     = np.dot(np.transpose(X), X)
            XTX_inv = inv(XTX)
            XTY     = np.dot(np.transpose(X), Y)
            b_tmp       = np.dot(XTX_inv, XTY)
            b_tmp       = b_tmp.flatten()

            b_ls.append(b_tmp)
            self.fit_all_gw.append(b_tmp[3]/b_tmp[1])
            self.fit_all_r2.append(1.0 - np.mean((Y - np.dot(X,b_tmp))**2)/np.var(Y))

            print "Percentage of variance explained (on dV/dt): %0.2f" % (self.fit_all_r2[-1]*100.0)

            del b_tmp

        # Select optimal regression coeffs 'b'.
        b = b_ls[np.argmax(self.fit_all_r2)]

        # Extract explicit model parameters from regression result b
        ####################################################################################################

        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
        self.gw = b[3] * self.C
        self.eta.setFilter_Coefficients(-b[4:]*self.C)

        self.tau_w = self.fit_all_tau_w[np.argmax(self.fit_all_r2)]

        self.printParameters()


        # Print optimal r2
        ####################################################################################################
        self.var_explained_dV = np.max(self.fit_all_r2)
        print "Percentage of variance explained (on dV/dt): %0.2f" % (self.var_explained_dV*100.0)

        if plot:
            self.plot_tauw()


        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        """SSE = 0     # sum of squared errors
        VAR = 0     # variance of data

        for tr in experiment.trainingset_traces :

            if tr.useTrace :

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                indices_tmp = tr.getROI_FarFromSpikes(0.0, self.Tref)

                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])

        var_explained_V = 1.0 - SSE / VAR

        self.var_explained_V = var_explained_V
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)"""


    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace, DT_beforeSpike=5.0, tau_w = 'default', Vmin = None):

        """
        Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression
        defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
        The input parameter trace is an ojbect of class Trace.

        Inputs:
            trace -- Trace object
            DT_beforeSpike -- float
                Time (ms) before each spike to omit from fitting.
            tau_w -- float or 'default'
                Value to use for tau_w when computing w. Uses self.tau_w if set to 'default'.
            Vmin -- float or None
                Data below this voltage is not used for fitting.
        """

        if tau_w == 'default':
            tau_w = self.tau_w

        # Length of the voltage trace
        Tref_ind = int(self.Tref/trace.dt)


        # Select region where to perform linear regression (specified in the ROI of individual taces)
        ####################################################################################################
        selection = trace.getROI_FarFromSpikes(DT_beforeSpike, self.Tref)
        selection_l = len(selection)

        # Handle voltage cutoff.
        ####################################################################################################
        if Vmin is not None:
            selection = np.logical_and(selection, trace.V > Vmin)


        # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
        ####################################################################################################
        X = np.zeros( (selection_l, 4) )

        # Fill first four columns of X matrix
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l)
        X[:,3] = self.compute_w(trace.V[selection], tau_w = tau_w) # Resonating current.


        # Compute and fill the remaining columns associated with the spike-triggered current eta
        X_eta = self.eta.convolution_Spiketrain_basisfunctions(trace.getSpikeTimes() + self.Tref, trace.T, trace.dt)
        X = np.concatenate( (X, X_eta[selection,:]), axis=1 )


        # Build Y vector (voltage derivative \dot_V_data)
        ####################################################################################################
        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]

        return (X, Y)

    ### Visualization-related methods.
    def plot_tauw(self):
        """Visualize model fits as a function of tau_w.
        """

        plt.figure()

        plt.subplot(211)
        plt.title('Variance explained')
        plt.plot(self.fit_all_tau_w, self.fit_all_r2, 'k.-')
        plt.xlabel(r'$\tau_w$')
        plt.ylabel(r'$R^2$')

        plt.subplot(212)
        plt.title('Estimated $g_w$')
        plt.plot(self.fit_all_tau_w, self.fit_all_gw, 'k.-')
        plt.xlabel(r'$\tau_w$')
        plt.ylabel(r'$\hat{g}_w$')

        plt.tight_layout()

        plt.show()


#%% SIMPLE TESTS FOR resGIF

if __name__ == '__main__':

    testgif = resGIF(0.1)

    # Load a test experiment
    import os
    from src.Experiment import Experiment

    tstexpt = Experiment('Test', 0.1)
    tstexpt.addTrainingSetTrace(
        FILETYPE = 'Axon',
        fname = os.path.join('data', 'gif_test', 'DRN656_train.abf'),
        V_channel = 0, I_channel = 1
    )

    for tr in tstexpt.trainingset_traces:
        tr.detectSpikes()

    testgif.fitSubthresholdDynamics(tstexpt, plot = True)
