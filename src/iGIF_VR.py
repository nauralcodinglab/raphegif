#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import weave

from src.iGIF_NP import iGIF_NP
import src.Tools as Tools


#%% DEFINE iGIF WITH VARIABLE RESET

class iGIF_VR(iGIF_NP):

    """iGIF_VR subclass of iGIF_NP with voltage-dependent reset rule.
    """

    def __init__(self, dt = 0.1):

        super(iGIF_VR, self).__init__(dt = dt)

        # Initialize attributes for variable reset rule.
        self.Vr_intercept = self.Vr
        del self.Vr
        self.Vr_slope = 0.


    ### Fitting methods.

    def fit(self, experiment, DT_beforeSpike = 5.0, theta_inf_nbbins=5, theta_tau_all=np.linspace(1.0, 10.0, 5), do_plot=False):

        """
        Fit the iGIF_NP model on experimental data (details of the mehtod can be found in Mensi et al. 2016).
        The experimental data are stored in the object experiment (the fit is performed on the training set traces).

        Input parameters:

        - experiment       : object Experiment containing the experimental data to be fitted.

        - DT_beforeSpike   : ms, amount of data removed before each spike to perform the linear regression on the voltage derivative.

        - theta_inf_nbbins : integer, number of rectangular basis functions used to define the nonlinear coupling f(V).
                             The actual rectangular basis functions will be computed automatically based on the data (as explained in Mensi et al. 2016).

        - theta_tau_all    : list of float, timescales of the threshold-voltage coupling tau_theta tested during the fit (the one of those giving the max likelihood solution is reteined).

        - do_plot          : if True, a plot is made which shows the max likelihood as a function of the timescale tau_theta.

        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit iGIF_NP"
        print "################################\n"

        self.fitVoltageReset(experiment, Tref = self.Tref, DT_beforeSpike = DT_beforeSpike, do_plot=False)
        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)
        self.defineBinningForThetaInf(experiment, theta_inf_nbbins)
        self.fitStaticThreshold(experiment)
        self.fitThresholdDynamics(experiment, theta_tau_all, do_plot=do_plot)

        self.fit_flag = True


    def fitVoltageReset(self, experiment, Tref, DT_beforeSpike, do_plot=False):

        """
        Implement modified version of Step 1 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015

        experiment: Experiment object on which the model is fitted.
        Tref: ms, absolute refractory period.

        The voltage reset is estimated using a linear fit of the spike-triggered
        voltage immediately following each spike.
        """

        print "Estimate voltage reset (Tref = %0.1f ms)..." % (Tref)

        # Fix absolute refractory period
        self.dt = experiment.dt
        self.Tref = Tref

        # Input processing.
        Tref_ind = int(self.Tref / self.dt)
        DT_beforespike_ind = int(DT_beforeSpike / self.dt)

        # Extract voltage just before and just after each spike.
        V_before = []
        V_after = []
        for tr in experiment.trainingset_traces :
            if tr.useTrace and len(tr.spks) > 0:
                for spkind in tr.spks:
                    if spkind - DT_beforespike_ind > 0 and spkind + Tref_ind < len(tr.V):
                        V_before.append(tr.V[spkind - DT_beforespike_ind])
                        V_after.append(tr.V[spkind + Tref_ind])
                    else:
                        continue
            else:
                continue

        V_before = np.array(V_before)
        V_after = np.array(V_after)

        assert len(V_before) == len(V_after), 'Number of voltage points before and after spikes do not match.'
        all_spike_nb = len(V_before)

        # Estimate voltage reset
        p_vec = np.polyfit(V_before, V_after, 1)
        self.Vr_intercept = p_vec[1]
        self.Vr_slope = p_vec[0]

        # Avg spike shape not saved as in parent class.
        # Set to None in case someone tries to use.
        self.avg_spike_shape = None
        self.avg_spike_shape_support = None

        # Save data used to compute reset rule.
        self._Vreset_data = {'V_before_spk': V_before, 'V_after_spk': V_after}

        if do_plot :
            self.plotVoltageReset()

        print "Done! Vr_intercept = %0.2f mV, Vr_slope = %0.2f (computed on %d spikes)" % (self.Vr_intercept, self.Vr_slope, all_spike_nb)


    def defineBinningForThetaInf(self, experiment, theta_inf_nbbins) :

        """
        Simulate by forcing spikes, and based on voltage distribution, define binning to extract nonlinear coupling.

        Note: argument last_bin_constrained removed because of dynamic reset rule.
        """

        # Precompute all the matrices used in the gradient ascent

        all_V_spikes = []

        for tr in experiment.trainingset_traces:

            if tr.useTrace :

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                all_V_spikes.append(V_est[tr.getSpikeIndices()])


        all_V_spikes = np.concatenate(all_V_spikes)

        V_min = np.min(all_V_spikes)
        V_max = np.max(all_V_spikes)


        print "\nDefine binning to extract theta_inf (V)..."
        print "Interval: %0.1f - %0.1f " % (V_min, V_max)

        self.theta_bins = np.linspace(V_min, V_max, theta_inf_nbbins+1)
        self.theta_bins[-1] += 100.0
        self.theta_i = np.zeros(theta_inf_nbbins)

        print "Bins (mV): ", self.theta_bins


    ### Methods related to simulations.

    def simulate(self, I, V0):

        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 (mV) indicate the initial condition V(0)=V0.

        The function returns:

        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times

        """

        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt

        # Model parameters
        p_gl                = self.gl
        p_C                 = self.C
        p_El                = self.El
        p_Vr_slope          = self.Vr_slope
        p_Vr_intercept      = self.Vr_intercept
        p_Tref              = self.Tref
        p_Vt_star           = self.Vt_star
        p_DV                = self.DV
        p_lambda0           = self.lambda0


        # Model parameters  definin threshold coupling
        p_theta_tau = self.theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")
        p_theta_i    = self.theta_i
        p_theta_i    = p_theta_i.astype("double")


        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma     = p_gamma.astype('double')
        p_gamma_l   = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")

        theta_trace = np.array(np.zeros(p_T), dtype="double")
        R     = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta = np.zeros((p_T,R))
        theta = theta.astype("double")


        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

        # Set initial condition
        V[0] = V0

        code =  """
                #include <math.h>

                int   T_ind     = int(p_T);
                float dt        = float(p_dt);

                float gl                = float(p_gl);
                float C                 = float(p_C);
                float El                = float(p_El);
                float Vr_slope          = float(p_Vr_slope);
                float Vr_intercept      = float(p_Vr_intercept);
                int   Tref_ind          = int(float(p_Tref)/dt);
                float Vt_star           = float(p_Vt_star);
                float DeltaV            = float(p_DV);
                float lambda0           = float(p_lambda0);
                float theta_tau         = float(p_theta_tau);

                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);

                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float rr = 0.0;

                float theta_tot;
                float theta_taufactor = (1.0-dt/theta_tau);

                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );


                    // INTEGRATION THRESHOLD DYNAMICS
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    for (int r=0; r<R; r++) {

                        theta[t+1,r] = theta_taufactor*theta[t,r];                           // everybody decay

                        if ( V[t] >= p_theta_bins[r] && V[t] < p_theta_bins[r+1] ) {         // identify who integrates
                            theta[t+1,r] += dt/theta_tau;
                        }
                    }

                    theta_tot = 0.0;
                    for (int r=0; r<R; r++) {
                        theta_tot += p_theta_i[r]*theta[t+1,r];
                    }

                    theta_trace[t+1] = theta_tot;
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t+1]-theta_trace[t+1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    rr = rand()/rand_max;
                    if (rr > p_dontspike) {

                        if (t+1 < T_ind-1)
                            spks[t+1] = 1.0;

                        t = t + Tref_ind;

                        if (t+1 < T_ind-1){
                            V[t+1] = V[t-Tref_ind] * Vr_slope + Vr_intercept;

                            for (int r=0; r<R; r++)
                                theta[t+1,r] = theta[t-Tref_ind,r];
                        }

                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+1+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+1+j] += p_gamma[j] ;

                    }

                }

                """

        vars = ['theta_trace', 'theta', 'R', 'p_theta_tau', 'p_theta_bins',
                'p_theta_i', 'p_T','p_dt','p_gl','p_C','p_El','p_Vr_slope',
                'p_Vr_intercept', 'p_Tref','p_Vt_star','p_DV','p_lambda0',
                'V','I','p_eta','p_eta_l','eta_sum','p_gamma','gamma_sum',
                'p_gamma_l','spks' ]

        v = weave.inline(code, vars)

        time      = np.arange(p_T)*self.dt
        eta_sum   = eta_sum[:p_T]
        V_T       = gamma_sum[:p_T] + p_Vt_star + theta_trace[:p_T]
        spks      = (np.where(spks==1)[0])*self.dt

        return (time, V, eta_sum, V_T, spks)


    def simulateDeterministic_forceSpikes(self, I, V0, spks):

        """
        Simulate the subthresohld response of the GIF model to an input current I (nA) with time step dt.
        Adaptation currents are forces to accur at times specified in the list spks (in ms) given as an argument
        to the function.
        V0 indicate the initial condition V(t=0)=V0.

        The function returns:

        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        """

        # Input parameters
        p_T          = len(I)
        p_dt         = self.dt


        # Model parameters
        p_gl            = self.gl
        p_C             = self.C
        p_El            = self.El
        p_Vr_slope      = self.Vr_slope
        p_Vr_intercept  = self.Vr_intercept
        p_Tref          = self.Tref
        p_Tref_i        = int(self.Tref/self.dt)


        # Model kernel
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta       = p_eta.astype('double')
        p_eta_l     = len(p_eta)


        # Define arrays
        V        = np.array(np.zeros(p_T), dtype="double")
        I        = np.array(I, dtype="double")
        spks     = np.array(spks, dtype="double")
        spks_i   = Tools.timeToIndex(spks, self.dt)


        # Compute adaptation current (sum of eta triggered at spike times in spks)
        eta_sum  = np.array(
                np.zeros(p_T + int(1.1*p_eta_l) + p_Tref_i),
                dtype="double")

        for s in spks_i :
            eta_sum[s + 1 + p_Tref_i  : s + 1 + p_Tref_i + p_eta_l] += p_eta

        eta_sum  = eta_sum[:p_T]


        # Set initial condition
        V[0] = V0


        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl                = float(p_gl);
                float C                 = float(p_C);
                float El                = float(p_El);
                float Vr_slope          = float(p_Vr_slope);
                float Vr_intercept      = float(p_Vr_intercept);
                int   Tref_ind          = int(float(p_Tref)/dt);


                int next_spike = spks_i[0] + Tref_ind;
                int spks_cnt = 0;


                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    V[t+1] = V[t] + dt/C*( -gl*(V[t] - El) + I[t] - eta_sum[t] );


                    if ( t == next_spike ) {
                        spks_cnt = spks_cnt + 1;
                        next_spike = spks_i[spks_cnt] + Tref_ind;
                        V[t] = V[t-1] * Vr_slope + Vr_intercept;
                        V[t-1] = 0 ;
                        t=t-1;
                    }

                }

                """

        vars = ['p_T','p_dt','p_gl','p_C','p_El','p_Vr_slope','p_Vr_intercept',
                'p_Tref','V','I','eta_sum','spks_i']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]

        return (time, V, eta_sum)


    ### Methods related to data presentation

    def printParameters(self):

        print "\n-------------------------"
        print "iGIF_NP model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t\t%0.3f"          % (self.C/self.gl)
        print "R (MOhm):\t\t%0.6f"            % (1.0/self.gl)
        print "C (nF):\t\t\t\t%0.3f"            % (self.C)
        print "gl (nS):\t\t%0.3f"             % (self.gl)
        print "El (mV):\t\t%0.3f"             % (self.El)
        print "Tref (ms):\t\t%0.3f"           % (self.Tref)
        print "Vr_slope :\t\t%0.3f"           % (self.Vr_slope)
        print "Vr_intercpt (mV) :\t\t%0.3f"  % (self.Vr_intercept)
        print "Vt* (mV):\t\t%0.3f"            % (self.Vt_star)
        print "DV (mV):\t\t%0.3f"             % (self.DV)
        print "tau_theta (ms):\t\t%0.3f"      % (self.theta_tau)
        print "-------------------------\n"


    def plotVoltageReset(self):
        """Make a simple plot of the voltage-dependent reset rule.
        """

        plt.figure()
        plt.plot(
            self._Vreset_data['V_before_spk'],
            self._Vreset_data['V_after_spk'],
            'ko', alpha = 0.7
        )
        x_tmp = np.array(
            [np.min(self._Vreset_data['V_before_spk']),
            np.max(self._Vreset_data['V_before_spk'])]
        )
        plt.plot(
            x_tmp, x_tmp * self.Vr_slope + self.Vr_intercept,
            'r-', label = 'Fitted reset rule'
        )
        plt.xlabel('V before spike (mV)')
        plt.ylabel('V after spike (mV)')
        plt.legend()
        plt.show()

#%% SIMPLE TESTS

if __name__ == '__main__':

    # Try instantiating.
    tstmod = iGIF_VR(0.1)

    # Try fitting a reset rule.
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

    tstmod.fitVoltageReset(tstexpt, 6.5, 1.5, do_plot = True)

    # Try fitting the model to an experiment.
    tstmod.fit(tstexpt, 1.5)

    # Try running a simulation using the model.
    I = np.concatenate([np.zeros(5000), 0.2 * np.ones(10000)])
    t, V, eta_sum, V_T, spks = tstmod.simulate(I, tstmod.El)

    plt.figure()

    plt.subplot(211)
    plt.plot(t, V, 'k-')
    plt.ylabel('V (mV)')

    plt.subplot(212)
    plt.plot(t, I, '-', color = 'gray')
    plt.ylabel('I (nA)')
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()
