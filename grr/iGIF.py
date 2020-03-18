import abc
from copy import deepcopy

import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from numpy.linalg import inv
import weave
from weave import converters

from .GIF import GIF
from .Filter_Rect import Filter_Rect_LogSpaced
from . import Tools
from .Tools import reprint


class iGIF(GIF):

    """
    Abstract class to define the:

    inactivating Generalized Integrate and Fire models

    Spike are produced stochastically with firing intensity:

    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),

    where the membrane potential dynamics is given (as in Pozzorini et al. PLOS Comp. Biol. 2015) by:

    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)

    This equation differs from the one used in Mensi et al. PLOS Comp. Biol. 2016 only because spike-triggerend adaptation is
    current based and not conductance based.

    The firing threshold V_T is given by:

    V_T = Vt_star + sum_j gamma(t-\hat t_j) + theta(t)

    and \hat t_j denote the spike times and theta(t) is given by:

    tau_theta dtheta/dt = -theta + f(V)

    Classes that inherit form iGIF must specify the nature of the coupling f(V) (this function can eg be defined as
    a liner sum of rectangular basis functions to perform a nonparametric fit).
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, dt=0.1):

        GIF.__init__(self, dt=dt)

    @abc.abstractmethod
    def getNonlinearCoupling(self):
        """
        This method should compute and return:
        - f(V): function defining the steady state value of theta as a funciton of voltage
        - support : the voltage over which f(V) is defined
        """

    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################

    def plotParameters(self):
        """
        Plot parameters of the iGIF model.
        """

        fig = plt.figure(facecolor='white', figsize=(16, 4))

        # Plot spike triggered current
        ####################################################################################################

        plt.subplot(1, 4, 1)

        (eta_support, eta) = self.eta.getInterpolatedFilter(self.dt)

        plt.plot(eta_support, eta, color='red', lw=2)
        plt.plot([eta_support[0], eta_support[-1]], [0, 0], ls=':', color='black', lw=2)

        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel("Time (ms)")
        plt.ylabel("Eta (nA)")

        # Plot spike triggered movement of the firing threshold
        ####################################################################################################

        plt.subplot(1, 4, 2)

        (gamma_support, gamma) = self.gamma.getInterpolatedFilter(self.dt)

        plt.plot(gamma_support, gamma, color='red', lw=2)
        plt.plot([gamma_support[0], gamma_support[-1]], [0, 0], ls=':', color='black', lw=2)

        plt.xlim([gamma_support[0], gamma_support[-1]])
        plt.xlabel("Time (ms)")
        plt.ylabel("Gamma (mV)")

        # Plot nonlinear coupling between firing threshold and membrane potential
        ####################################################################################################

        plt.subplot(1, 4, 3)

        (support, theta_inf) = self.getNonlinearCoupling()

        plt.plot(support, support, '--', color='black')
        plt.plot(support, theta_inf, 'red', lw=2)
        plt.plot([self.Vr], [self.Vt_star], 'o', mew=2, mec='black',  mfc='white', ms=8)

        plt.ylim([self.Vt_star-10.0, -20.0])

        plt.xlabel("Membrane potential (mV)")
        plt.ylabel("Theta (mV)")

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)

    @classmethod
    def compareModels(cls, iGIFs, labels=None):
        """
        Given a list of iGIF models, iGIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS

        print "\n#####################################"
        print "iGIF model comparison"
        print "#####################################\n"

        cnt = 0
        for iGIF in iGIFs:

            print "Model: " + labels[cnt]
            iGIF.printParameters()
            cnt += 1

        print "#####################################\n"

        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################

        plt.figure(facecolor='white', figsize=(14, 8))
        colors = plt.cm.jet(np.linspace(0.7, 1.0, len(iGIFs)))

        # MEMBRANE FILTER
        #######################################################################################################

        plt.subplot(2, 3, 1)

        cnt = 0
        for iGIF in iGIFs:

            if labels is None:
                label_tmp = ""
            else:
                label_tmp = labels[cnt]

            K_support = np.linspace(0, 150.0, 1500)
            K = 1./iGIF.C*np.exp(-K_support/(iGIF.C/iGIF.gl))
            plt.plot(K_support, K, color=colors[cnt], lw=2, label=label_tmp)
            cnt += 1

        plt.plot([K_support[0], K_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        if labels is not None:
            plt.legend()

        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')

        # SPIKE TRIGGERED CURRENT
        #######################################################################################################

        plt.subplot(2, 3, 2)

        cnt = 0
        for iGIF in iGIFs:

            (eta_support, eta) = iGIF.eta.getInterpolatedFilter(0.1)
            plt.plot(eta_support, eta, color=colors[cnt], lw=2)
            cnt += 1

        plt.plot([eta_support[0], eta_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        #plt.xscale('log', nonposx='clip')
        #plt.yscale('log', nonposy='clip')
        plt.xlim([eta_support[0], eta_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Eta (ms)')

        # ESCAPE RATE
        #######################################################################################################

        plt.subplot(2, 3, 3)

        cnt = 0
        for iGIF in iGIFs:

            V_support = np.linspace(iGIF.Vt_star-5*iGIF.DV, iGIF.Vt_star+10*iGIF.DV, 1000)
            escape_rate = iGIF.lambda0*np.exp((V_support-iGIF.Vt_star)/iGIF.DV)
            plt.plot(V_support, escape_rate, color=colors[cnt], lw=2)
            cnt += 1

        plt.plot([V_support[0], V_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        plt.ylim([0, 100])
        plt.xlim([V_support[0], V_support[-1]])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Escape rate (Hz)')

        # SPIKE TRIGGERED MOVEMENT OF THE FIRING THRESHOLD
        #######################################################################################################

        plt.subplot(2, 3, 4)

        cnt = 0
        for myiGIF in iGIFs:

            (gamma_support, gamma) = myiGIF.gamma.getInterpolatedFilter(0.1)
            plt.plot(gamma_support, gamma, color=colors[cnt], lw=2)

            cnt += 1

        plt.plot([gamma_support[0], gamma_support[-1]], [0, 0], ls=':', color='black', lw=2, zorder=-1)

        #plt.xscale('log', nonposx='clip')
        #plt.yscale('log', nonposy='clip')
        plt.xlim([gamma_support[0]+0.1, gamma_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Gamma (mV)')

        # NONLINEAR COUPLING OF THE FIRING THRESHOLD
        #######################################################################################################

        plt.subplot(2, 3, 5)

        cnt = 0
        for iGIF in iGIFs:
            (V, theta) = iGIF.getNonlinearCoupling()
            plt.plot(V, theta, color=colors[cnt], lw=2)
            plt.plot([iGIF.Vr], [iGIF.Vt_star], 'o', mew=2, mec=colors[cnt],  mfc=colors[cnt], ms=8)
            cnt += 1

        plt.plot(V, V, color='black', lw=2, ls='--')

        plt.xlim([-80, -20])
        plt.ylim([-60, -20])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Threshold theta (mV)')

        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)

        plt.show()

    @classmethod
    def plotAverageModel(cls, iGIFs):
        """
        Average model parameters and plot summary data.
        """

        GIF.plotAverageModel(iGIFs)

        # NONLINEAR THRESHOLD COUPLING
        #######################################################################################################
        plt.subplot(2, 4, 4)

        K_all = []

        plt.plot([-80, -20], [-80, -20], ls='--', color='black', lw=2, zorder=100)

        for iGIF in iGIFs:

            (K_support, K) = iGIF.getNonlinearCoupling()

            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)

            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std = np.std(K_all, axis=0)

        plt.fill_between(K_support, K_mean+K_std, y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)

        plt.xlim([-80, -20])
        plt.ylim([-65, -20])
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlabel('Membrane potential (mV)')
        plt.ylabel('Threshold coupling (mV)')

        # tau_theta
        #######################################################################################################
        plt.subplot(4, 6, 12+4)

        p_all = []
        for iGIF in iGIFs:

            p = iGIF.theta_tau
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau theta (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

        plt.show()


class iGIF_Na(iGIF):

    """
    inactivating Generalized Integrate and Fire model iGIF_Na
    in which the nonlinear coupling between membrane potential and firing threshold
    is defined as in Platkiewicz J and Brette R, PLOS CB 2011.

    Mathematically the nonlinear function f(V) defining the coupling is given by:

    f(V) = ka * log( 1 + exp( (V-Vi)/ki) )

    where:

    k_a: defines the slope of sodium channel activation in the Hodgkin-Huxley model (i.e. the slope of m_\infty(V)).
    k_i: defines the slope of sodium channel inactivation in the Hodgkin-Huxley model (i.e. the absolute value of the slope of h_\infty(V)).
    V_i: half-voltage inactivation of sodium channel

    This equation can be derived analytically from the HH model assuming that fast sodium channel inactivation is
    given by an inverted sigmoidal function of the membrane potential.

    For more details see Platkiewicz J and Brette R, PLOS CB 2011 or Mensi et al. PLOS CB 2016.
    """

    def __init__(self, dt=0.1):

        GIF.__init__(self, dt=dt)

        # Initialize parametres for nonlinear coupling

        self.theta_tau = 5.0                   # ms, timescale of threshold-voltage coupling

        self.theta_ka = 2.5                   # mV, slope of Na channel activation

        self.theta_ki = 3.0                   # mV, absolute value of the slope of Na channel inactivation

        self.theta_Vi = -55.0                 # mV, half-inactivation voltage for Na channels

        # Store parameters related to parameters extraction

        self.fit_all_ki = 0                     # mV, list containing all the values tested during the fit for ki

        self.fit_all_Vi = 0                     # mV, list containing all the values tested during the fit for Vi

        self.fit_all_likelihood = 0             # 2D matrix containing all the log-likelihood obtained with different (ki, Vi)

    def getNonlinearCoupling(self):
        """
        Compute and return the nonlinear function f(V) defining the threshold-voltage coupling.
        The function is computed as:

        f(V) = ka * log( 1 + exp( (V-Vi)/ki) )
        """

        support = np.linspace(-100, -20.0, 200)

        theta_inf = self.Vt_star + self.theta_ka*np.log(1 + np.exp((support - self.theta_Vi)/self.theta_ki))

        return (support, theta_inf)

    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, dt):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El.
        """

        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return spks_times

    def simulateVoltageResponse(self, I, dt):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt (ms).
        Return a list of spike times (in ms) as well as the dynamics of the subthreshold membrane potential V (mV) and the voltage threshold V_T (mV).
        The initial conditions for the simulation is V(0)=El and VT = VT^* (i.e. the membrane is at rest and threshold is at baseline).
        """

        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return (spks_times, V, V_T)

    def simulate(self, I, V0):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        - eta_sum  : nA, adaptation current
        - V_T      : mV, firing threshold
        - spks     : ms, list of spike times
        """
        # Input variables.
        modStim = deepcopy(self._coerceInputToModelStimulus(I))
        netInputCurrent = modStim.getNetCurrentVector(dtype='double')
        p_numberOfInputCurrents = modStim.numberOfCurrents
        inputConductanceVector = modStim.getConductanceMajorFlattening(dtype='double')
        inputConductanceReversals = modStim.getConductanceReversals(dtype='double')
        p_numberOfInputConductances = np.int32(modStim.numberOfConductances)

        # Input parameters
        p_T = modStim.timesteps
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Vt_star = self.Vt_star
        p_DV = self.DV
        p_lambda0 = self.lambda0

        # Model parameters  definin threshold coupling
        p_theta_ka = self.theta_ka
        p_theta_ki = self.theta_ki
        p_theta_Vi = self.theta_Vi
        p_theta_tau = self.theta_tau

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma = p_gamma.astype('double')
        p_gamma_l = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        theta = np.array(np.zeros(p_T), dtype="double")
        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

        # Set initial condition
        V[0] = V0

        code = """
                #include <math.h>

                int numberOfInputCurrents = int(p_numberOfInputCurrents);
                int numberOfInputConductances = int(p_numberOfInputConductances);

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float Vt_star    = float(p_Vt_star);
                float DeltaV     = float(p_DV);
                float lambda0    = float(p_lambda0);

                float theta_ka         = float(p_theta_ka);
                float theta_ki         = float(p_theta_ki);
                float theta_Vi         = float(p_theta_Vi);
                float theta_tau        = float(p_theta_tau);

                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);

                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float r = 0.0;


                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    float dV = dt / C * (-gl * (V[t] - El) - eta_sum[t]);
                    if (numberOfInputCurrents > 0)
                        dV += dt / C * netInputCurrent[t];
                    for (int i=0; i<numberOfInputConductances; i++)
                        dV +=
                            dt / C
                            * inputConductanceVector[t * numberOfInputConductances + i]
                            * (V[t] - inputConductanceReversals[i]);
                    V[t+1] = V[t] + dV;

                    // INTEGRATE THETA
                    theta[t+1] = theta[t] + dt/theta_tau*(-theta[t] + theta_ka*log(1+exp((V[t]-theta_Vi)/theta_ki)));


                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t]-theta[t+1])/DeltaV );
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    r = rand()/rand_max;
                    if (r > p_dontspike) {

                        if (t+1 < T_ind-1)
                            spks[t+1] = 1.0;

                        t = t + Tref_ind;

                        if (t+1 < T_ind-1)
                            V[t+1] = Vr;


                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+1+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+1+j] += p_gamma[j] ;

                    }

                }

                """

        vars = ['netInputCurrent', 'p_numberOfInputCurrents',
                'inputConductanceVector', 'inputConductanceReversals',
                'p_numberOfInputConductances',
                'theta', 'p_theta_ka', 'p_theta_ki', 'p_theta_Vi',
                'p_theta_tau', 'p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr',
                'p_Tref', 'p_Vt_star', 'p_DV', 'p_lambda0', 'V',
                'p_eta', 'p_eta_l', 'eta_sum',
                'p_gamma', 'gamma_sum', 'p_gamma_l', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt

        eta_sum = eta_sum[:p_T]
        gamma_sum = gamma_sum[:p_T]
        V_T = gamma_sum + p_Vt_star + theta[:p_T]

        spks = (np.where(spks == 1)[0])*self.dt

        return (time, V, eta_sum, V_T, spks)

    ######################################################################################################################
    # FUNCTIONS TO FIT DYNAMIC THRESHOLD BY BRUTE FORCE
    ######################################################################################################################

    def fit(self, experiment, theta_tau, ki_all, Vi_all, DT_beforeSpike=5.0, do_plot=False):
        """
        Fit the model to the training set data in experiment.

        Input parameters:

        - experiment     : an instance of the class Experiment containing the experimental data that will be used for the fit (only training set data will be used).
        - theta_tau      : ms, timescale of the threshold-voltage coupling (this parameter is not fitted but has to be known). To fit this parameter, fit first a GIF_NP model to the data.
        - ki_all         : mV, array of values containing the parameters k_i (ie, Na channel inactivation slope) tested during the fit
        - Vi_all         : mV, array of values containing the parameters V_i (ie, Na channel half inactivation voltage) tested during the fit
        - DT_beforeSpike : ms, amount of time removed before each action potential (these data will not be considered when fitting the subthreshold membrane potential dynamics)
        - doPlot         : if True plot the max-likelihood as a function of ki and Vi.
        """

        print "\n################################"
        print "# Fit iGIF_Na"
        print "################################\n"

        # Three step procedure used for parameters extraction

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)

        self.theta_tau = theta_tau

        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics_bruteforce(experiment, ki_all, Vi_all, do_plot=do_plot)

        #self.fit_bruteforce_flag = True
        #self.fit_binary_flag     = False

    def fitThresholdDynamics_bruteforce(self, experiment, ki_all, Vi_all, do_plot=False):

        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold

        print "Fit dynamic threshold..."

        #beta0_dynamicThreshold = np.concatenate( ( [1/self.DV], [-self.Vt_star/self.DV], [0], self.gamma.getCoefficients()/self.DV))
        beta0_dynamicThreshold = np.concatenate(([1/self.DV], [-self.Vt_star/self.DV], [0], np.zeros(self.gamma.getNbOfBasisFunctions())))

        all_L = np.zeros((len(ki_all), len(Vi_all)))
        L_opt = -10**20
        beta_opt = 0
        ki_opt = 0
        Vi_opt = 0

        for ki_i in np.arange(len(ki_all)):

            for Vi_i in np.arange(len(Vi_all)):

                ki = ki_all[ki_i]
                Vi = Vi_all[Vi_i]

                print "\nTest parameters: ki = %0.2f mV, Vi = %0.2f mV" % (ki, Vi)

                # Perform fit
                (beta_tmp, L_tmp) = self.maximizeLikelihood_dynamicThreshold(experiment, ki, Vi, beta0_dynamicThreshold)

                all_L[ki_i, Vi_i] = L_tmp

                if L_tmp > L_opt:

                    print "NEW OPTIMAL SOLUTION: LL = %0.5f (bit/spike)" % (L_tmp)

                    L_opt = L_tmp
                    beta_opt = beta_tmp
                    Vi_opt = Vi
                    ki_opt = ki

        # Store result

        self.DV = 1.0/beta_opt[0]
        self.Vt_star = -beta_opt[1]*self.DV
        self.theta_ka = -beta_opt[2]*self.DV
        self.gamma.setFilter_Coefficients(-beta_opt[3:]*self.DV)
        self.theta_Vi = Vi_opt
        self.theta_ki = ki_opt

        self.fit_all_ki = ki_all
        self.fit_all_Vi = Vi_all
        self.fit_all_likelihood = all_L

        # Plot landscape

        if do_plot:

            (ki_plot, Vi_plot) = np.meshgrid(Vi_all, ki_all)

            print np.shape(ki_plot)
            print np.shape(Vi_plot)
            print np.shape(all_L)

            plt.figure(facecolor='white', figsize=(6, 6))

            plt.pcolor(Vi_plot, ki_plot, all_L)
            plt.plot(ki_opt, Vi_opt, 'o', mfc='white', mec='black', ms=10)

            plt.xlabel('ki (mV)')
            plt.ylabel('Vi (mV)')

            plt.xlim([ki_all[0], ki_all[-1]])
            plt.ylim([Vi_all[0], Vi_all[-1]])
            plt.show()

    def maximizeLikelihood_dynamicThreshold(self, experiment, ki, Vi, beta0, maxIter=10**3, stopCond=10**-6):

        all_X = []
        all_X_spikes = []
        all_sum_X_spikes = []

        T_tot = 0.0
        N_spikes_tot = 0.0

        traces_nb = 0

        for tr in experiment.trainingset_traces:

            if tr.useTrace:

                traces_nb += 1

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                # Precomputes matrices to perform gradient ascent on log-likelihood
                (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = self.buildXmatrix_dynamicThreshold(tr, V_est, ki, Vi)

                T_tot += T
                N_spikes_tot += N_spikes

                all_X.append(X_tmp)
                all_X_spikes.append(X_spikes_tmp)
                all_sum_X_spikes.append(sum_X_spikes_tmp)

        logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)

        # Perform gradient ascent

        print "Maximize log-likelihood (bit/spks)..."

        beta = beta0
        old_L = 1

        for i in range(maxIter):

            learning_rate = 1.0

            if i <= 10:                      # be careful in the first iterations (using a small learning rate in the first step makes the fit more stable)
                learning_rate = 0.1

            L = 0; G = 0; H = 0;

            for trace_i in np.arange(traces_nb):
                (L_tmp, G_tmp, H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
                L += L_tmp; G += G_tmp; H += H_tmp;

            beta = beta - learning_rate*np.dot(inv(H), G)

            if (i > 0 and abs((L-old_L)/old_L) < stopCond):              # If converged

                print "\nConverged after %d iterations!\n" % (i+1)
                break

            old_L = L

            # Compute normalized likelihood (for print)
            # The likelihood is normalized with respect to a poisson process and units are in bit/spks
            L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
            reprint(L_norm)

        if (i == maxIter - 1):                                           # If too many iterations
            print "\nNot converged after %d iterations.\n" % (maxIter)

        return (beta, L_norm)

    def buildXmatrix_dynamicThreshold(self, tr, V_est, ki, Vi):
        """
        Use this function to fit a model in which the firing threshold dynamics is defined as:
        V_T(t) = Vt_star + sum_i gamma(t-\hat t_i) (i.e., model with spike-triggered movement of the threshold)
        """

        # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)
        selection = tr.getROI_FarFromSpikes(-tr.dt, self.Tref)
        T_l_selection = len(selection)

        # Get spike indices in coordinates of selection
        spk_train = tr.getSpikeTrain()
        spks_i_afterselection = np.where(spk_train[selection] == 1)[0]

        # Compute average firing rate used in the fit
        T_l = T_l_selection*tr.dt/1000.0                # Total duration of trace used for fit (in s)
        N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit

        # Define X matrix
        X = np.zeros((T_l_selection, 3))
        X[:, 0] = V_est[selection]
        X[:, 1] = np.ones(T_l_selection)

        X_theta = self.exponentialFiltering_Brette_ref(V_est, tr.getSpikeIndices(), ki, Vi)
        X[:, 2] = X_theta[selection]

        # Compute and fill the remaining columns associated with the spike-triggered current gamma
        X_gamma = self.gamma.convolution_Spiketrain_basisfunctions(tr.getSpikeTimes() + self.Tref, tr.T, tr.dt)
        X = np.concatenate((X, X_gamma[selection, :]), axis=1)

        # Precompute other quantities
        X_spikes = X[spks_i_afterselection, :]
        sum_X_spikes = np.sum(X_spikes, axis=0)

        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)

    def exponentialFiltering_Brette_ref(self, V, spks_ind, ki, Vi):
        """
        Auxiliary function used to compute the matrix Y used in maximum likelihood.
        This function compute a set of integrals:

        theta_i(t) = \int_0^T 1\tau_theta exp(-s/tau_theta) f{ V(t-s) }ds

        wheref(V) = log( 1 + exp( (V-Vi)/ki) )

        After each spike in spks_ind theta_i(t) is reset to 0 mV and the integration restarts.

        The function returns a matrix where each line is given by theta_i(t).

        Input parameters:

        - V : numpy array containing the voltage trace (in mV)
        - spks_ind   : list of spike times in ms (used to reset)
        - theta_tau  : ms, timescale used in the intergration.

        """

        # Input parameters
        p_T = len(V)
        p_dt = self.dt

        # Model parameters  definin threshold coupling
        p_theta_tau = self.theta_tau
        p_Tref = self.Tref
        p_theta_ki = ki
        p_theta_Vi = Vi

        # Define arrays
        V = np.array(V, dtype="double")
        theta = np.array(np.zeros(p_T), dtype="double")

        spks = np.array(spks_ind, dtype='double')
        p_spks_L = len(spks)

        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                int   Tref_ind   = int(float(p_Tref)/dt);
                float theta_ki         = float(p_theta_ki);
                float theta_Vi         = float(p_theta_Vi);
                float theta_tau        = float(p_theta_tau);

                float theta_taufactor = (1.0-dt/theta_tau);

                int spks_L     = int(p_spks_L);
                int spks_cnt   = 0;
                int next_spike = int(spks[0]);


                for (int t=0; t<T_ind-1; t++) {

                    // INTEGRATE THETA

                    theta[t+1] = theta[t] + dt/theta_tau*(-theta[t] + log(1+exp((V[t]-theta_Vi)/theta_ki)));


                    // MANAGE RESET

                    if ( t+1 >= next_spike ) {

                        if(spks_cnt < spks_L) {
                            spks_cnt  += 1;
                            next_spike = int(spks[spks_cnt]);
                        }
                        else {
                            next_spike = T_ind+1;
                        }


                        if ( t + Tref_ind < T_ind-1 ) {
                            theta[t + Tref_ind] = 0.0;
                        }

                        t = t + Tref_ind;

                    }

                }

                """

        vars = ['spks', 'p_spks_L', 'theta', 'p_theta_ki', 'p_theta_Vi', 'p_theta_tau', 'p_T', 'p_dt', 'p_Tref', 'V']

        v = weave.inline(code, vars)

        return theta

    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################

    def printParameters(self):

        print "\n-------------------------"
        print "iGIF_Na model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f" % (self.C/self.gl)
        print "R (MOhm):\t%0.3f" % (1.0/self.gl)
        print "C (nF):\t\t%0.3f" % (self.C)
        print "gl (nS):\t%0.3f" % (self.gl)
        print "El (mV):\t%0.3f" % (self.El)
        print "Tref (ms):\t%0.3f" % (self.Tref)
        print "Vr (mV):\t%0.3f" % (self.Vr)
        print "Vt* (mV):\t%0.3f" % (self.Vt_star)
        print "DV (mV):\t%0.3f" % (self.DV)
        print "tau_theta (ms):\t%0.3f" % (self.theta_tau)
        print "ka (mV):\t%0.3f" % (self.theta_ka)
        print "ki (mV):\t%0.3f" % (self.theta_ki)
        print "Vi (mV):\t%0.3f" % (self.theta_Vi)
        print "ka/ki (mV):\t%0.3f" % (self.theta_ka/self.theta_ki)
        print "-------------------------\n"

    def plotParameters(self):

        super(iGIF_Na, self).plotParameters()

        plt.subplot(1, 4, 4)

        (ki_plot, Vi_plot) = np.meshgrid(self.fit_all_Vi, self.fit_all_ki)

        plt.pcolor(Vi_plot, ki_plot, self.fit_all_likelihood)
        plt.plot(self.theta_ki, self.theta_Vi, 'o', mfc='white', mec='black', ms=10)

        plt.xlim([self.fit_all_ki[0], self.fit_all_ki[-1]])
        plt.ylim([self.fit_all_Vi[0], self.fit_all_Vi[-1]])
        plt.xlabel('ki (mV)')
        plt.ylabel('Vi (mV)')

    @classmethod
    def plotAverageModel(cls, iGIFs):
        """
        Averae and plot the parameters of a list of iGIF_Na models.
        Input paramters:
        - iGIFs : list of iGFI objects
        """

        GIF.plotAverageModel(iGIFs)

        iGIF.plotAverageModel(iGIFs)

        # ki
        #######################################################################################################
        plt.subplot(4, 6, 12+5)

        p_all = []
        for myiGIF in iGIFs:

            p = myiGIF.theta_ka
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('ka (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

        # ki
        #######################################################################################################
        plt.subplot(4, 6, 18+5)

        p_all = []
        for myiGIF in iGIFs:

            p = myiGIF.theta_ki
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('ki (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])

        # Vi
        #######################################################################################################
        plt.subplot(4, 6, 12+6)

        p_all = []
        for myiGIF in iGIFs:

            p = myiGIF.theta_Vi
            p_all.append(p)

        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('Vi (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])


class iGIF_NP(iGIF):

    """
    inactivating Generalized Integrate and Fire model
    in which the nonlinear coupling between membrane potential and firing threshold
    is defined as a linear combination of rectangular basis functions.

    Mathematically the nonlinear function f(V) defining the coupling is given by:

    f(V) = sum_j b_j * g_j(V)

    where:

    b_j: are parameters
    g_j(V) : are rectangular functions of V
    """

    def __init__(self, dt=0.1):

        GIF.__init__(self, dt=dt)

        # Initialize threshold-voltage coupling

        self.theta_tau = 5.0                          # ms, timescale of threshold-voltage coupling

        self.theta_bins = np.linspace(-50, -10, 11)    # mV, nodes of rectangular basis functions g_j(V) used to define f(V)

        self.theta_i = np.linspace(0, 30.0, 10)   # mV, coefficients b_j associated with the rectangular basis functions above (these parameters define the functional shape of the threshodl-voltage coupling )

        self.fit_flag = False

        self.fit_all_tau_theta = 0                     # list containing all the tau_theta (i.e. the timescale of the threshold-voltage coupling) tested during the fit

        self.fit_all_likelihood = 0                    # list containing all the log-likelihoods obtained with different tau_theta
                                                       # (the optimal timescale tau_theta is the one that maximize the model likelihood)

    def getNonlinearCoupling(self):
        """
        Compute and return the nonlinear coupling f(V), as well as its support, according to the rectangular basis functions and its coefficients.
        """

        support = np.linspace(-100, 0.0, 1000)
        dV = support[1]-support[0]

        theta_inf = np.ones(len(support))*self.Vt_star

        for i in np.arange(len(self.theta_i)-1):

            lb_i = np.where(support >= self.theta_bins[i])[0][0]
            ub_i = np.where(support >= self.theta_bins[i+1])[0][0]

            theta_inf[lb_i:ub_i] = self.theta_i[i] + self.Vt_star

        theta_inf[ub_i:] = self.theta_i[-1] + self.Vt_star

        return (support, theta_inf)

    ########################################################################################################
    # FUNCTIONS FOR SIMULATIONS
    ########################################################################################################
    def simulateSpikingResponse(self, I, dt):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt (ms).
        Return a list of spike times (in ms).
        The initial conditions for the simulation is V(0)=El and VT = VT^* (i.e. the membrane is at rest and threshold is at baseline).
        """
        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return spks_times

    def simulateVoltageResponse(self, I, dt):
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt (ms).
        Return a list of spike times (in ms) as well as the dynamics of the subthreshold membrane potential V (mV) and the voltage threshold V_T (mV).
        The initial conditions for the simulation is V(0)=El and VT = VT^* (i.e. the membrane is at rest and threshold is at baseline).
        """

        self.setDt(dt)

        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)

        return (spks_times, V, V_T)

    def simulate(self, I, V0, return_dict=False):
        """Simulate the spiking response of the iGIF_NP.

        Arguments
        ---------
        I : 1D float array
            Input current in nA.
        V0 : float
            Initial voltage (mV).
        return_dict : bool, default False
            Whether to return a tuple (for backwards compatibility) or a dict.

        Returns
        -------
        If return_dict is False, a tuple of
        (time, V, eta_sum, V_T, spike_times).
        Otherwise, a dict containing the following keys:
            - `time`
            - `V`
            - `eta_sum` (adaptation current in nA)
            - `gamma_sum` (threshold movement in mV)
            - `theta` (voltage-threshold coupling in mV)
            - `V_T` (voltage threshold in mV)
            - `firing_intensity` (intensity of spike-generating process in Hz)
            - `spike_times`

        """
        # Input variables.
        modStim = deepcopy(self._coerceInputToModelStimulus(I))
        netInputCurrent = modStim.getNetCurrentVector(dtype='double')
        p_numberOfInputCurrents = modStim.numberOfCurrents
        inputConductanceVector = modStim.getConductanceMajorFlattening(dtype='double')
        inputConductanceReversals = modStim.getConductanceReversals(dtype='double')
        p_numberOfInputConductances = np.int32(modStim.numberOfConductances)

        # Input parameters
        p_T = modStim.timesteps
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr = self.Vr
        p_Tref = self.Tref
        p_Vt_star = self.Vt_star
        p_DV = self.DV
        p_lambda0 = self.lambda0

        # Model parameters  definin threshold coupling
        p_theta_tau = self.theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")
        p_theta_i = self.theta_i
        p_theta_i = p_theta_i.astype("double")

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma = p_gamma.astype('double')
        p_gamma_l = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")

        theta_trace = np.array(np.zeros(p_T), dtype="double")
        R = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta = np.zeros((p_T, R))
        theta = theta.astype("double")

        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

        lambda_storage = np.zeros_like(V, dtype="double")

        # Set initial condition
        V[0] = V0

        code = """
                #include <math.h>

                int numberOfInputCurrents = int(p_numberOfInputCurrents);
                int numberOfInputConductances = int(p_numberOfInputConductances);

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);

                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                float Vr         = float(p_Vr);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float Vt_star    = float(p_Vt_star);
                float DeltaV     = float(p_DV);
                float lambda0    = float(p_lambda0);
                float theta_tau  = float(p_theta_tau);

                int eta_l        = int(p_eta_l);
                int gamma_l      = int(p_gamma_l);

                float rand_max  = float(RAND_MAX);
                float p_dontspike = 0.0 ;
                float lambda = 0.0 ;
                float rr = 0.0;

                float theta_taufactor = (1.0-dt/theta_tau);

                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATE VOLTAGE
                    float dV = dt / C * (-gl * (V[t] - El) - eta_sum[t]);
                    if (numberOfInputCurrents > 0)
                        dV += dt / C * netInputCurrent[t];
                    for (int i=0; i<numberOfInputConductances; i++)
                        dV +=
                            dt / C
                            * inputConductanceVector[t * numberOfInputConductances + i]
                            * (V[t] - inputConductanceReversals[i]);
                    V[t+1] = V[t] + dV;

                    // INTEGRATION THRESHOLD DYNAMICS
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
                    for (int r=0; r<R; r++) {

                        theta[t+1,r] = theta_taufactor*theta[t,r];                           // everybody decay

                        if ( V[t] >= p_theta_bins[r] && V[t] < p_theta_bins[r+1] ) {         // identify who integrates
                            theta[t+1,r] += dt/theta_tau;
                        }
                    }

                    float theta_tot = 0.0;
                    for (int r=0; r<R; r++) {
                        theta_tot += p_theta_i[r]*theta[t+1,r];
                    }

                    theta_trace[t+1] = theta_tot;
                    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



                    // COMPUTE PROBABILITY OF EMITTING ACTION POTENTIAL
                    lambda = lambda0*exp( (V[t+1]-Vt_star-gamma_sum[t+1]-theta_trace[t+1])/DeltaV );
                    lambda_storage[t+1] = lambda;
                    p_dontspike = exp(-lambda*(dt/1000.0));                                  // since lambda0 is in Hz, dt must also be in Hz (this is why dt/1000.0)


                    // PRODUCE SPIKE STOCHASTICALLY
                    rr = rand()/rand_max;
                    if (rr > p_dontspike) {

                        if (t+1 < T_ind-1)
                            spks[t+1] = 1.0;

                        t = t + Tref_ind;

                        if (t+1 < T_ind-1){
                            V[t+1] = Vr;

                            for (int r=0; r<R; r++)
                                theta[t+1,r] = 0.0;
                        }

                        // UPDATE ADAPTATION PROCESSES
                        for(int j=0; j<eta_l; j++)
                            eta_sum[t+1+j] += p_eta[j];

                        for(int j=0; j<gamma_l; j++)
                            gamma_sum[t+1+j] += p_gamma[j] ;

                    }

                }

                """

        vars = ['netInputCurrent', 'p_numberOfInputCurrents',
                'inputConductanceVector', 'inputConductanceReversals',
                'p_numberOfInputConductances',
                'theta_trace', 'theta', 'R', 'p_theta_tau', 'p_theta_bins',
                'p_theta_i', 'p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr',
                'p_Tref', 'p_Vt_star', 'p_DV', 'p_lambda0', 'lambda_storage', 'V',
                'p_eta', 'p_eta_l', 'eta_sum',
                'p_gamma', 'gamma_sum', 'p_gamma_l', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]
        gamma_sum = gamma_sum[:p_T]
        V_T = gamma_sum + p_Vt_star + theta_trace[:p_T]
        spks = (np.where(spks == 1)[0])*self.dt

        if return_dict:
            return {
                'time': time,
                'V': V,
                'eta_sum': eta_sum,
                'gamma_sum': gamma_sum,
                'theta': theta_trace,
                'V_T': V_T,
                'spike_times': spks,
                'firing_intensity': lambda_storage,
            }
        else:
            # Return tuple (backwards compatible)
            return (time, V, eta_sum, V_T, spks)

    def fit(self, experiment, DT_beforeSpike=5.0, theta_inf_nbbins=5, theta_tau_all=np.linspace(1.0, 10.0, 5), last_bin_constrained=False, do_plot=False):
        """
        Fit the iGIF_NP model on experimental data (details of the mehtod can be found in Mensi et al. 2016).
        The experimental data are stored in the object experiment (the fit is performed on the training set traces).

        Input parameters:

        - experiment       : object Experiment containing the experimental data to be fitted.

        - DT_beforeSpike   : ms, amount of data removed before each spike to perform the linear regression on the voltage derivative.

        - theta_inf_nbbins : integer, number of rectangular basis functions used to define the nonlinear coupling f(V).
                             The actual rectangular basis functions will be computed automatically based on the data (as explained in Mensi et al. 2016).

        - theta_tau_all    : list of float, timescales of the threshold-voltage coupling tau_theta tested during the fit (the one of those giving the max likelihood solution is reteined).

        - last_bin_constrained : {True, False}, set this to True in order to guarantee that the rectangular basis functions defining f(V) only starts above the voltage reset.

        - do_plot          : if True, a plot is made which shows the max likelihood as a function of the timescale tau_theta.

        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """

        # Three step procedure used for parameters extraction

        print "\n################################"
        print "# Fit iGIF_NP"
        print "################################\n"

        self.fitVoltageReset(experiment, self.Tref, do_plot=False)

        self.fitSubthresholdDynamics(experiment, DT_beforeSpike=DT_beforeSpike)

        self.defineBinningForThetaInf(experiment, theta_inf_nbbins, last_bin_constrained=last_bin_constrained)

        self.fitStaticThreshold(experiment)

        self.fitThresholdDynamics(experiment, theta_tau_all, do_plot=do_plot)

        self.fit_flag = True

    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################
    def defineBinningForThetaInf(self, experiment, theta_inf_nbbins, last_bin_constrained=True):
        """
        Simulate by forcing spikes, and based on voltage distribution, define binning to extract nonlinear coupling.
        """

        # Precompute all the matrices used in the gradient ascent

        all_V_spikes = []

        for tr in experiment.trainingset_traces:

            if tr.useTrace:

                # Simulate subthreshold dynamics
                (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                all_V_spikes.append(V_est[tr.getSpikeIndices()])

        all_V_spikes = np.concatenate(all_V_spikes)

        V_min = np.min(all_V_spikes)
        V_max = np.max(all_V_spikes)

        # Do not allow to have a free bin at voltage reset (this should avoid a bad interaction between gamma and theta_inf)

        if last_bin_constrained:
            if V_min < self.Vr + 0.5:
                V_min = self.Vr + 0.5

        print "\nDefine binning to extract theta_inf (V)..."
        print "Interval: %0.1f - %0.1f " % (V_min, V_max)

        self.theta_bins = np.linspace(V_min, V_max, theta_inf_nbbins+1)
        self.theta_bins[-1] += 100.0
        self.theta_i = np.zeros(theta_inf_nbbins)

        print "Bins (mV): ", self.theta_bins

    ########################################################################################################
    # FUNCTIONS TO FIT DYNAMIC THRESHLD
    ########################################################################################################

    def fitThresholdDynamics(self, experiment, theta_tau_all, do_plot=False):

        self.setDt(experiment.dt)

        # Fit a dynamic threshold using a initial condition the result obtained by fitting a static threshold

        print "Fit dynamic threshold..."

        # Perform fit
        beta0_dynamicThreshold = np.concatenate(([1/self.DV], [-self.Vt_star/self.DV], self.gamma.getCoefficients()/self.DV, self.theta_i))
        (beta_opt, theta_tau_opt) = self.maximizeLikelihood_dynamicThreshold(experiment, beta0_dynamicThreshold, theta_tau_all, do_plot=do_plot)

        # Store result
        self.DV = 1.0/beta_opt[0]
        self.Vt_star = -beta_opt[1]*self.DV
        self.gamma.setFilter_Coefficients(-beta_opt[2:2+self.gamma.getNbOfBasisFunctions()]*self.DV)
        self.theta_i = -beta_opt[2+self.gamma.getNbOfBasisFunctions():]*self.DV
        self.theta_tau = theta_tau_opt

        self.printParameters()

    def maximizeLikelihood_dynamicThreshold(self, experiment, beta0, theta_tau_all, maxIter=10**3, stopCond=10**-6, do_plot=False):

        beta_all = []
        L_all = []

        for theta_tau in theta_tau_all:

            print "\nTest tau_theta = %0.1f ms... \n" % (theta_tau)

            # Precompute all the matrices used in the gradient ascent

            all_X = []
            all_X_spikes = []
            all_sum_X_spikes = []

            T_tot = 0.0
            N_spikes_tot = 0.0

            traces_nb = 0

            for tr in experiment.trainingset_traces:

                if tr.useTrace:

                    traces_nb += 1

                    # Simulate subthreshold dynamics
                    (time, V_est, eta_sum_est) = self.simulateDeterministic_forceSpikes(tr.I, tr.V[0], tr.getSpikeTimes())

                    # Precomputes matrices to perform gradient ascent on log-likelihood
                    (X_tmp, X_spikes_tmp, sum_X_spikes_tmp, N_spikes, T) = self.buildXmatrix_dynamicThreshold(tr, V_est, theta_tau)

                    T_tot += T
                    N_spikes_tot += N_spikes

                    all_X.append(X_tmp)
                    all_X_spikes.append(X_spikes_tmp)
                    all_sum_X_spikes.append(sum_X_spikes_tmp)

            logL_poisson = N_spikes_tot*(np.log(N_spikes_tot/T_tot)-1)

            # Perform gradient ascent

            print "Maximize log-likelihood (bit/spks)..."

            beta = beta0
            old_L = 1

            for i in range(maxIter):

                learning_rate = 1.0

                if i <= 10:                      # be careful in the first iterations (using a small learning rate in the first step makes the fit more stable)
                    learning_rate = 0.1

                L = 0; G = 0; H = 0;

                for trace_i in np.arange(traces_nb):
                    (L_tmp, G_tmp, H_tmp) = self.computeLikelihoodGradientHessian(beta, all_X[trace_i], all_X_spikes[trace_i], all_sum_X_spikes[trace_i])
                    L += L_tmp; G += G_tmp; H += H_tmp;

                beta = beta - learning_rate*np.dot(inv(H), G)

                if (i > 0 and abs((L-old_L)/old_L) < stopCond):              # If converged
                    print "\nConverged after %d iterations!\n" % (i+1)
                    break

                old_L = L

                # Compute normalized likelihood (for print)
                # The likelihood is normalized with respect to a poisson process and units are in bit/spks
                L_norm = (L-logL_poisson)/np.log(2)/N_spikes_tot
                reprint(L_norm)

            if (i == maxIter - 1):                                           # If too many iterations
                print "\nNot converged after %d iterations.\n" % (maxIter)

            L_all.append(L_norm)
            beta_all.append(beta)

        ind_opt = np.argmax(L_all)

        theta_tau_opt = theta_tau_all[ind_opt]
        beta_opt = beta_all[ind_opt]
        L_norm_opt = L_all[ind_opt]

        print "\n Optimal timescale: %0.2f ms" % (theta_tau_opt)
        print "Log-likelihood: %0.2f bit/spike" % (L_norm_opt)

        self.fit_all_tau_theta = theta_tau_all
        self.fit_all_likelihood = L_all

        if do_plot:

            plt.figure(figsize=(6, 6), facecolor='white')
            plt.plot(theta_tau_all, L_all, '.-', color='black')
            plt.plot([theta_tau_opt], [L_norm_opt], '.', color='red')
            plt.xlabel('Threshold coupling timescale (ms)')
            plt.ylabel('Log-likelihood (bit/spike)')
            plt.show()

        return (beta_opt, theta_tau_opt)

    def buildXmatrix_dynamicThreshold(self, tr, V_est, theta_tau):
        """
        Use this function to fit a model in which the firing threshold dynamics is defined as:
        V_T(t) = Vt_star + sum_i gamma(t-\hat t_i) (i.e., model with spike-triggered movement of the threshold)
        """

        # Get indices be removing absolute refractory periods (-self.dt is to not include the time of spike)
        selection = tr.getROI_FarFromSpikes(-tr.dt, self.Tref)
        T_l_selection = len(selection)

        # Get spike indices in coordinates of selection
        spk_train = tr.getSpikeTrain()
        spks_i_afterselection = np.where(spk_train[selection] == 1)[0]

        # Compute average firing rate used in the fit
        T_l = T_l_selection*tr.dt/1000.0                # Total duration of trace used for fit (in s)
        N_spikes = len(spks_i_afterselection)           # Nb of spikes in the trace used for fit

        # Define X matrix
        X = np.zeros((T_l_selection, 2))
        X[:, 0] = V_est[selection]
        X[:, 1] = np.ones(T_l_selection)

        # Compute and fill the remaining columns associated with the spike-triggered current gamma
        X_gamma = self.gamma.convolution_Spiketrain_basisfunctions(tr.getSpikeTimes() + self.Tref, tr.T, tr.dt)
        X = np.concatenate((X, X_gamma[selection, :]), axis=1)

        # Fill columns related with nonlinera coupling
        X_theta = self.exponentialFiltering_ref(V_est, tr.getSpikeIndices(), theta_tau)
        X = np.concatenate((X, X_theta[selection, :]), axis=1)

        # Precompute other quantities
        X_spikes = X[spks_i_afterselection, :]
        sum_X_spikes = np.sum(X_spikes, axis=0)

        return (X, X_spikes, sum_X_spikes,  N_spikes, T_l)

    def exponentialFiltering_ref(self, V, spks_ind, theta_tau):
        """
        Auxiliary function used to compute the matrix Y used in maximum likelihood.
        This function compute a set of integrals:

        theta_i(t) = \int_0^T 1\tau_theta exp(-s/tau_theta) g_j{ V(t-s) }ds

        After each spike in spks_ind theta_i(t) is reset to 0 mV and the integration restarts.

        The function returns a matrix where each line is given by theta_i(t).

        Input parameters:

        - V : numpy array containing the voltage trace (in mV)
        - spks_ind   : list of spike times in ms (used to reset)
        - theta_tau  : ms, timescale used in the intergration.

        """

        # Input parameters
        p_T = len(V)
        p_dt = self.dt
        p_Tref = self.Tref

        # Model parameters  definin threshold coupling
        p_theta_tau = theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")

        # Define arrays
        V = np.array(V, dtype="double")

        R = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta = np.zeros((p_T, R))
        theta = theta.astype("double")

        spks = np.array(spks_ind, dtype='double')
        p_spks_L = len(spks)

        code = """
                #include <math.h>

                int   T_ind      = int(p_T);
                float dt         = float(p_dt);
                int   Tref_ind   = int(float(p_Tref)/dt);
                float theta_tau  = float(p_theta_tau);

                float theta_taufactor = (1.0-dt/theta_tau);

                int spks_L     = int(p_spks_L);
                int spks_cnt   = 0;
                int next_spike = int(spks(0));

                for (int t=0; t<T_ind-1; t++) {


                    // INTEGRATION THRESHOLD DYNAMICS

                    for (int r=0; r<R; r++) {

                        theta(t+1,r) = theta_taufactor*theta(t,r);                           // everybody decay

                        if ( V(t) >= p_theta_bins(r) && V(t) < p_theta_bins(r+1) ) {         // identify who integrates
                            theta(t+1,r) += dt/theta_tau;
                        }
                    }


                    // MANAGE RESET

                    if ( t+1 >= next_spike ) {

                        if(spks_cnt < spks_L) {
                            spks_cnt  += 1;
                            next_spike = int(spks(spks_cnt));
                        }
                        else {
                            next_spike = T_ind+1;
                        }


                        if ( t + Tref_ind < T_ind-1 ) {
                            for (int r=0; r<R; r++)
                                theta(t + Tref_ind ,r) = 0.0;                                // reset
                        }

                        t = t + Tref_ind;

                    }

                }

                """

        vars = ['spks', 'p_spks_L', 'theta', 'R', 'p_theta_tau', 'p_theta_bins', 'p_T', 'p_dt', 'p_Tref', 'V']

        v = weave.inline(code, vars, type_converters=converters.blitz)

        return theta

    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################

    def printParameters(self):

        print "\n-------------------------"
        print "iGIF_NP model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f" % (self.C/self.gl)
        print "R (MOhm):\t%0.6f" % (1.0/self.gl)
        print "C (nF):\t\t%0.3f" % (self.C)
        print "gl (nS):\t%0.3f" % (self.gl)
        print "El (mV):\t%0.3f" % (self.El)
        print "Tref (ms):\t%0.3f" % (self.Tref)
        print "Vr (mV):\t%0.3f" % (self.Vr)
        print "Vt* (mV):\t%0.3f" % (self.Vt_star)
        print "DV (mV):\t%0.3f" % (self.DV)
        print "tau_theta (ms):\t%0.3f" % (self.theta_tau)
        print "-------------------------\n"

    def plotParameters(self):

        super(iGIF_NP, self).plotParameters()

        if self.fit_flag:

            plt.subplot(1, 4, 4)
            plt.plot(self.fit_all_tau_theta, self.fit_all_likelihood, '.-', color='black')
            plt.plot([self.theta_tau], [np.max(self.fit_all_likelihood)], '.', color='red')
            plt.xlabel('Threshold coupling timescale (ms)')
            plt.ylabel('Max log-likelihood (bit/spike)')

        plt.subplots_adjust(left=0.07, bottom=0.2, right=0.98, top=0.90, wspace=0.35, hspace=0.10)

        plt.show()


class iGIF_VR(iGIF_NP):

    """iGIF_VR subclass of iGIF_NP with voltage-dependent reset rule.
    """

    def __init__(self, dt=0.1):

        super(iGIF_VR, self).__init__(dt=dt)

        # Initialize attributes for variable reset rule.
        self.Vr_intercept = self.Vr
        del self.Vr
        self.Vr_slope = 0.

    ### Fitting methods.

    def fit(self, experiment, DT_beforeSpike=5.0, theta_inf_nbbins=5, theta_tau_all=np.linspace(1.0, 10.0, 5), do_plot=False):
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

        self.fitVoltageReset(experiment, Tref=self.Tref, DT_beforeSpike=DT_beforeSpike, do_plot=False)
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
        for tr in experiment.trainingset_traces:
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

        if do_plot:
            self.plotVoltageReset()

        print "Done! Vr_intercept = %0.2f mV, Vr_slope = %0.2f (computed on %d spikes)" % (self.Vr_intercept, self.Vr_slope, all_spike_nb)

    def defineBinningForThetaInf(self, experiment, theta_inf_nbbins):
        """
        Simulate by forcing spikes, and based on voltage distribution, define binning to extract nonlinear coupling.

        Note: argument last_bin_constrained removed because of dynamic reset rule.
        """

        # Precompute all the matrices used in the gradient ascent

        all_V_spikes = []

        for tr in experiment.trainingset_traces:

            if tr.useTrace:

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
        p_T = len(I)
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr_slope = self.Vr_slope
        p_Vr_intercept = self.Vr_intercept
        p_Tref = self.Tref
        p_Vt_star = self.Vt_star
        p_DV = self.DV
        p_lambda0 = self.lambda0

        # Model parameters  definin threshold coupling
        p_theta_tau = self.theta_tau
        p_theta_bins = self.theta_bins
        p_theta_bins = p_theta_bins.astype("double")
        p_theta_i = self.theta_i
        p_theta_i = p_theta_i.astype("double")

        # Model kernels
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        (p_gamma_support, p_gamma) = self.gamma.getInterpolatedFilter(self.dt)
        p_gamma = p_gamma.astype('double')
        p_gamma_l = len(p_gamma)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")

        theta_trace = np.array(np.zeros(p_T), dtype="double")
        R = len(self.theta_bins)-1                 # subthreshold coupling theta
        theta = np.zeros((p_T, R))
        theta = theta.astype("double")

        spks = np.array(np.zeros(p_T), dtype="double")
        eta_sum = np.array(np.zeros(p_T + 2*p_eta_l), dtype="double")
        gamma_sum = np.array(np.zeros(p_T + 2*p_gamma_l), dtype="double")

        # Set initial condition
        V[0] = V0

        code = """
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
                'p_theta_i', 'p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr_slope',
                'p_Vr_intercept', 'p_Tref', 'p_Vt_star', 'p_DV', 'p_lambda0',
                'V', 'I', 'p_eta', 'p_eta_l', 'eta_sum', 'p_gamma', 'gamma_sum',
                'p_gamma_l', 'spks']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]
        V_T = gamma_sum[:p_T] + p_Vt_star + theta_trace[:p_T]
        spks = (np.where(spks == 1)[0])*self.dt

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
        p_T = len(I)
        p_dt = self.dt

        # Model parameters
        p_gl = self.gl
        p_C = self.C
        p_El = self.El
        p_Vr_slope = self.Vr_slope
        p_Vr_intercept = self.Vr_intercept
        p_Tref = self.Tref
        p_Tref_i = int(self.Tref/self.dt)

        # Model kernel
        (p_eta_support, p_eta) = self.eta.getInterpolatedFilter(self.dt)
        p_eta = p_eta.astype('double')
        p_eta_l = len(p_eta)

        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        spks = np.array(spks, dtype="double")
        spks_i = Tools.timeToIndex(spks, self.dt)

        # Compute adaptation current (sum of eta triggered at spike times in spks)
        eta_sum = np.array(
                np.zeros(p_T + int(1.1*p_eta_l) + p_Tref_i),
                dtype="double")

        for s in spks_i:
            eta_sum[s + 1 + p_Tref_i: s + 1 + p_Tref_i + p_eta_l] += p_eta

        eta_sum = eta_sum[:p_T]

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

        vars = ['p_T', 'p_dt', 'p_gl', 'p_C', 'p_El', 'p_Vr_slope', 'p_Vr_intercept',
                'p_Tref', 'V', 'I', 'eta_sum', 'spks_i']

        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        eta_sum = eta_sum[:p_T]

        return (time, V, eta_sum)

    ### Methods related to data presentation

    def printParameters(self):

        print "\n-------------------------"
        print "iGIF_NP model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t\t%0.3f" % (self.C/self.gl)
        print "R (MOhm):\t\t%0.6f" % (1.0/self.gl)
        print "C (nF):\t\t\t\t%0.3f" % (self.C)
        print "gl (nS):\t\t%0.3f" % (self.gl)
        print "El (mV):\t\t%0.3f" % (self.El)
        print "Tref (ms):\t\t%0.3f" % (self.Tref)
        print "Vr_slope :\t\t%0.3f" % (self.Vr_slope)
        print "Vr_intercpt (mV) :\t\t%0.3f" % (self.Vr_intercept)
        print "Vt* (mV):\t\t%0.3f" % (self.Vt_star)
        print "DV (mV):\t\t%0.3f" % (self.DV)
        print "tau_theta (ms):\t\t%0.3f" % (self.theta_tau)
        print "-------------------------\n"

    def plotVoltageReset(self):
        """Make a simple plot of the voltage-dependent reset rule.
        """

        plt.figure()
        plt.plot(
            self._Vreset_data['V_before_spk'],
            self._Vreset_data['V_after_spk'],
            'ko', alpha=0.7
        )
        x_tmp = np.array(
            [np.min(self._Vreset_data['V_before_spk']),
            np.max(self._Vreset_data['V_before_spk'])]
        )
        plt.plot(
            x_tmp, x_tmp * self.Vr_slope + self.Vr_intercept,
            'r-', label='Fitted reset rule'
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
    from .Experiment import Experiment

    tstexpt = Experiment('Test', 0.1)
    tstexpt.addTrainingSetTrace(
        FILETYPE='Axon',
        fname=os.path.join('data', 'gif_test', 'DRN656_train.abf'),
        V_channel=0, I_channel=1
    )

    for tr in tstexpt.trainingset_traces:
        tr.detectSpikes()

    tstmod.fitVoltageReset(tstexpt, 6.5, 1.5, do_plot=True)

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
    plt.plot(t, I, '-', color='gray')
    plt.ylabel('I (nA)')
    plt.xlabel('Time')

    plt.tight_layout()
    plt.show()
