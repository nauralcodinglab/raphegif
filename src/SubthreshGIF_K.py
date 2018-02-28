import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import numba as nb

import weave
from numpy.linalg import inv

from GIF import *
from Filter_Rect_LogSpaced import *
from Trace import *

from Tools import reprint
from numpy import nan, NaN

import math


class SubthreshGIF_K(GIF) :

    """
    Generalized Integrate and Fire model defined in Pozzorini et al. PLOS Comp. Biol. 2015
    
    Spike are produced stochastically with firing intensity:
    
    lambda(t) = lambda0 * exp( (V(t)-V_T(t))/DV ),
    
    where the membrane potential dynamics is given by:
    
    C dV/dt = -gl(V-El) + I - sum_j eta(t-\hat t_j)
    
    and the firing threshold V_T is given by:
    
    V_T = Vt_star + sum_j gamma(t-\hat t_j)
    
    and \hat t_j denote the spike times.
    """

    def __init__(self, dt=0.1):
                   
        self.dt = dt                    # dt used in simulations (eta and gamma are interpolated according to this value)
  
        # Define model parameters
        self.gl      = 1.0/100.0        # nS, leak conductance
        self.C       = 20.0*self.gl     # nF, capacitance
        self.El      = -65.0            # mV, reversal potential
        
        # Define attributes to store goodness-of-fit
        self.var_explained_dV = 0
        self.var_explained_V = 0
        
        # Define attributes to store data used during fitting
        self.I_data = 0
        
        self.V_data = 0
        self.V_sim = 0
        self.m_sim = 0
        self.h_sim = 0
        self.n_sim = 0
        
        # Define attributes related to K_conductance gating parameters
        self.m_Vhalf = None
        self.m_k = None
        self.m_tau = None
        
        self.h_Vhalf = None
        self.h_k = None
        self.h_tau = None
        
        self.n_Vhalf = None
        self.n_k = None
        self.n_tau = None
        
        self.E_K = None
        
        self.gbar_K1 = 0
        self.gbar_K2 = 0
    

    
    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Spiking model
    ########################################################################################################
    
    
    def simulateSpikingResponse(self, I, dt):
        
        """
        Subthreshold model does not spike.
        """
        
        raise RuntimeError('Subthreshold model does not spike.')


    ########################################################################################################
    # IMPLEMENT ABSTRACT METHODS OF Threshold Model
    ########################################################################################################
    
    
    def simulateVoltageResponse(self, I, dt) :

        self.setDt(dt)
    
        (time, V, eta_sum, V_T, spks_times) = self.simulate(I, self.El)
        
        return (spks_times, V, V_T)


    ########################################################################################################
    # METHODS FOR K_CONDUCTANCE GATES
    ########################################################################################################
    
    def mInf(self, V):
    
        """Compute the equilibrium activation gate state of the potassium conductance.
        """
        
        return 1/(1 + np.exp(-self.m_k * (V - self.m_Vhalf)))
    

    def hInf(self, V):
        
        """Compute the equilibrium state of the inactivation gate of the potassium conductance.
        """
        
        return 1/(1 + np.exp(-self.h_k * (V - self.h_Vhalf)))
    
    
    def nInf(self, V):
        
        """Compute the equilibrium state of the non-inactivating conductance.
        """
        
        return 1/(1 + np.exp(-self.n_k * (V - self.n_Vhalf)))
    
    
    def computeGating(self, V, inf_vec, tau):
        
        """
        Compute the state of a gate over time.
        
        Wrapper for _computeGatingInternal, which is a nb.jit-accelerated static method.
        """
        
        return self._computeGatingInternal(V, inf_vec, tau, self.dt)
    
    
    @staticmethod
    @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64, nb.float64))
    def _computeGatingInternal(V, inf_vec, tau, dt):
        
        """
        Internal method called by computeGating.
        """
        
        output = np.empty_like(V, dtype = np.float64)
        output[0] = inf_vec[0]
        
        for i in range(1, len(V)):
            
            output[i] = output[i - 1] + (inf_vec[i - 1] - output[i - 1])/tau * dt
            
        return output
    
    
    def getDF_K(self, V):
        
        """
        Compute driving force on K based on SubthreshGIF_K.E_K and V.
        """
        
        return V - self.E_K
    
    
    ########################################################################################################
    # METHODS FOR NUMERICAL SIMULATIONS
    ########################################################################################################  
      
    def simulate(self, I, V0):
 
        """
        Simulate the spiking response of the GIF model to an input current I (nA) with time step dt.
        V0 indicate the initial condition V(0)=V0.
        The function returns:
        - time     : ms, support for V, eta_sum, V_T, spks
        - V        : mV, membrane potential
        """
 
        # Input parameters
        p_T         = len(I)
        p_dt        = self.dt
        
        # Model parameters
        p_gl        = self.gl
        p_C         = self.C 
        p_El        = self.El
        
        p_m_Vhalf   = self.m_Vhalf
        p_m_k       = self.m_k
        p_m_tau     = self.m_tau
        
        p_h_Vhalf   = self.h_Vhalf
        p_h_k       = self.h_k
        p_h_tau     = self.h_tau
        
        p_n_Vhalf   = self.n_Vhalf
        p_n_k       = self.n_k
        p_n_tau     = self.n_tau
        
        p_E_K       = self.E_K
        
        p_gbar_K1   = self.gbar_K1
        p_gbar_K2   = self.gbar_K2
        
      
        # Define arrays
        V = np.array(np.zeros(p_T), dtype="double")
        I = np.array(I, dtype="double")
        
        m = np.zeros_like(V, dtype = "double")
        h = np.zeros_like(V, dtype = "double")
        n = np.zeros_like(V, dtype = "double")
 
        # Set initial condition
        V[0] = V0
        
        m[0] = self.mInf(V0)
        h[0] = self.hInf(V0)
        n[0] = self.nInf(V0)
         
        code =  """
                #include <math.h>
                
                
                // DECLARE IMPORTED PARAMETERS
                
                int   T_ind      = int(p_T);                
                float dt         = float(p_dt); 
                
                float gl         = float(p_gl);
                float C          = float(p_C);
                float El         = float(p_El);
                
                float m_Vhalf    = float(p_m_Vhalf);
                float m_k        = float(p_m_k);
                float m_tau      = float(p_m_tau);
                
                float h_Vhalf    = float(p_h_Vhalf);
                float h_k        = float(p_h_k);
                float h_tau      = float(p_h_tau);
                
                float n_Vhalf    = float(p_n_Vhalf);
                float n_k        = float(p_n_k);
                float n_tau      = float(p_n_tau);
                
                float E_K        = float(p_E_K);
                
                float gbar_K1    = float(p_gbar_K1);
                float gbar_K2    = float(p_gbar_K2);
                
                
                // DECLARE ADDITIONAL VARIABLES
                
                float m_inf_t;
                float h_inf_t;
                float n_inf_t;
                
                float DF_K_t;
                float gk_1_term;
                float gk_2_term;
                                                
                for (int t=1; t<T_ind; t++) {
    
                    // INTEGRATE m GATE
                    m_inf_t = 1/(1 + exp(-m_k * (V[t] - m_Vhalf)));
                    m[t] = m[t-1] + dt/m_tau*(m_inf_t - m[t-1]);
                    
                    // INTEGRATE h GATE
                    h_inf_t = 1/(1 + exp(-h_k * (V[t] - h_Vhalf)));
                    h[t] = h[t-1] + dt/h_tau*(h_inf_t - h[t-1]);
                    
                    // INTEGRATE n GATE
                    n_inf_t = 1/(1 + exp(-n_k * (V[t] - n_Vhalf)));
                    m[t] = n[t-1] + dt/n_tau*(n_inf_t - n[t-1]);
                    
                    // COMPUTE K CONDUCTANCES
                    DF_K_t = V[t-1] - E_K;
                    gk_1_term = -DF_K_t * m[t] * h[t] * gbar_K1;
                    gk_2_term = -DF_K_t * n[t] * gbar_K2;
                    
                    // INTEGRATE VOLTAGE
                    V[t] = V[t-1] + dt/C*( -gl*(V[t-1] - El) + I[t-1] + gk_1_term + gk_2_term);
               
               
                }
                
                """
 
        vars = [ 'p_T','p_dt','p_gl','p_C','p_El',
                'p_m_Vhalf', 'p_m_k', 'p_m_tau',
                'p_h_Vhalf', 'p_h_k', 'p_h_tau',
                'p_n_Vhalf', 'p_n_k', 'p_n_tau',
                'p_E_K', 'p_gbar_K1', 'p_gbar_K2',
                'V','I','m','h','n' ]
        
        v = weave.inline(code, vars)

        time = np.arange(p_T)*self.dt
        
        return (time, V, m, h, n)

        
    def simulateDeterministic_forceSpikes(self, *args):
        
        """
        Subthreshold model does not spike.
        """
 
        raise RuntimeError('Subthreshold model does not spike.')
        

           
    ########################################################################################################
    # METHODS FOR MODEL FITTING
    ########################################################################################################  
      
         
    def fit(self, experiment):
        
        """
        Fit the GIF model on experimental data.
        The experimental data are stored in the object experiment provided as an input.
        The parameter DT_beforeSpike (in ms) defines the region that is cut before each spike when fitting the subthreshold dynamics of the membrane potential.
        Only training set traces in experiment are used to perform the fit.
        """
        
        # Three step procedure used for parameters extraction 
        
        print "\n################################"
        print "# Fit GIF"
        print "################################\n"
        
        
        self.fitSubthresholdDynamics(experiment)



    ########################################################################################################
    # FIT VOLTAGE RESET GIVEN ABSOLUTE REFRACOTORY PERIOD (step 1)
    ########################################################################################################


    def fitVoltageReset(self, experiment, Tref, do_plot=False):
        
        """
        Subthreshold model does not spike.
        """
        
        raise RuntimeError('Subthreshold model does not spike.')
        
        


    ########################################################################################################
    # FUNCTIONS RELATED TO FIT OF SUBTHRESHOLD DYNAMICS (step 2)
    ########################################################################################################


    def fitSubthresholdDynamics(self, experiment):
          
        """
        Implement Step 2 of the fitting procedure introduced in Pozzorini et al. PLOS Comb. Biol. 2015
        The voltage reset is estimated by computing the spike-triggered average of the voltage.
        experiment: Experiment object on which the model is fitted.
        DT_beforeSpike: in ms, data right before spikes are excluded from the fit. This parameter can be used to define that time interval.
        """  
                  
        print "\nGIF MODEL - Fit subthreshold dynamics..." 
            
        # Expand eta in basis functions
        self.dt = experiment.dt
        
        
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
                (X_tmp, Y_tmp) = self.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)
     
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
        b       = np.dot(XTX_inv, XTY)
        b       = b.flatten()
   
   
        # Extract explicit model parameters from regression result b
        ####################################################################################################

        self.C  = 1./b[1]
        self.gl = -b[0]*self.C
        self.El = b[2]*self.C/self.gl
        
        self.gbar_K1 = b[3] * self.C
        self.gbar_K2 = b[4] * self.C
    
        self.printParameters()   
        
        
        # Compute percentage of variance explained on dV/dt
        ####################################################################################################

        var_explained_dV = 1.0 - np.mean((Y - np.dot(X,b))**2)/np.var(Y)
        
        self.var_explained_dV = var_explained_dV
        print "Percentage of variance explained (on dV/dt): %0.2f" % (var_explained_dV*100.0)

        
        # Compute percentage of variance explained on V (see Eq. 26 in Pozzorini et al. PLOS Comp. Biol. 2105)
        ####################################################################################################

        SSE = 0     # sum of squared errors
        VAR = 0     # variance of data
        
        self.I_data = []
        self.V_data = []
        self.V_sim = []
        self.m_sim = []
        self.h_sim = []
        self.n_sim = []
        
        for tr in experiment.trainingset_traces :
        
            if tr.useTrace :

                # Simulate subthreshold dynamics 
                (time, V_est, m_est, h_est, n_est) = self.simulate(tr.I, tr.V[0])
                
                # Store data used for simulation along with simulated points
                self.I_data.append(tr.I)
                self.V_data.append(tr.V)
                self.V_sim.append(V_est)
                self.m_sim.append(m_est)
                self.h_sim.append(h_est)
                self.n_sim.append(n_est)
                
                # Compute SSE on points in ROI
                indices_tmp = tr.getROI()
                
                SSE += sum((V_est[indices_tmp] - tr.V[indices_tmp])**2)
                VAR += len(indices_tmp)*np.var(tr.V[indices_tmp])
                
        var_explained_V = 1.0 - SSE / VAR
        
        self.var_explained_V = var_explained_V
        print "Percentage of variance explained (on V): %0.2f" % (var_explained_V*100.0)
                
                    
    def fitSubthresholdDynamics_Build_Xmatrix_Yvector(self, trace):
           
        """
        Compute the X matrix and the Y vector (i.e. \dot_V_data) used to perfomr the linear regression 
        defined in Eq. 17-18 of Pozzorini et al. 2015 for an individual experimental trace provided as parameter.
        The input parameter trace is an ojbect of class Trace.
        """
        
        # Select region where to perform linear regression (specified in the ROI of individual taces)
        ####################################################################################################
        selection = trace.getROI()
        selection_l = len(selection)
        
        
        # Build X matrix for linear regression (see Eq. 18 in Pozzorini et al. PLOS Comp. Biol. 2015)
        ####################################################################################################
        X = np.zeros( (selection_l, 5) )
        
        # Compute equilibrium state of each gate
        m_inf_vec = self.mInf(trace.V)
        h_inf_vec = self.hInf(trace.V)
        n_inf_vec = self.nInf(trace.V)
        
        # Compute time-dependent state of each gate over whole trace
        m_vec = self.computeGating(trace.V, m_inf_vec, self.m_tau)
        h_vec = self.computeGating(trace.V, h_inf_vec, self.h_tau)
        n_vec = self.computeGating(trace.V, n_inf_vec, self.n_tau)
        
        # Compute gating state of each conductance over whole trace
        gating_vec_1 = m_vec * h_vec
        gating_vec_2 = n_vec
        
        # Compute K driving force over whole trace
        DF_K = self.getDF_K(trace.V)
        
        # Fill first three columns of X matrix        
        X[:,0] = trace.V[selection]
        X[:,1] = trace.I[selection]
        X[:,2] = np.ones(selection_l)
        
        # Fill K-conductance columns
        X[:,3] = -(gating_vec_1 * DF_K)[selection]
        X[:,4] = -(gating_vec_2 * DF_K)[selection]
        

        # Build Y vector (voltage derivative \dot_V_data)    
        ####################################################################################################
        Y = np.array( np.concatenate( (np.diff(trace.V)/trace.dt, [0]) ) )[selection]      

        return (X, Y)
        
        
        
    ########################################################################################################
    # FUNCTIONS RELATED TO FIT FIRING THRESHOLD PARAMETERS (step 3)
    ########################################################################################################        
 
         
    def fitStaticThreshold(self, *args):
        
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')

   
    def fitThresholdDynamics(self, *args):
                  
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
          
      
    def maximizeLikelihood(self, *args) :
    
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
     
        
    def computeLikelihoodGradientHessian(self, *args) : 
        
        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')


    def buildXmatrix_staticThreshold(self, *args) :

        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
        
            
    def buildXmatrix_dynamicThreshold(self, *args) :

        """
        Subthreshold models do not spike.
        """
        
        raise RuntimeError('Subthreshold models do not spike.')
 
 
    
    ########################################################################################################
    # EXTRACT POWER SPECTRUM DENSITY
    ########################################################################################################     
        
    def extractPowerSpectrumDensity(self, I, V0, dt, do_plot = False) :
        
        # Check that timestep of current and GIF are not different
        if dt != self.dt:
            raise ValueError('Timestep of I ({}ms) and GIF ({}ms) must be '
                             'the same or power spectrum may not make '
                             'sense!'.format(dt, self.dt))
        
        t, V_sim = self.simulate(I, V0)
        
        GIF_PSD = Trace(V_sim, 
                        I, 
                        len(I) * self.dt,
                        self.dt).extractPowerSpectrumDensity(do_plot)
        
        return GIF_PSD
    
    
    
    ########################################################################################################
    # PLOT AND PRINT FUNCTIONS
    ########################################################################################################     
        
        
    def plotParameters(self) :
        
        """
        Generate figure with model filters.
        """
        
        plt.figure(facecolor='white', figsize=(5,4))
            
        # Plot kappa
        plt.subplot(1,1,1)
        
        K_support = np.linspace(0,150.0, 300)             
        K = 1./self.C*np.exp(-K_support/(self.C/self.gl)) 
            
        plt.plot(K_support, K, color='red', lw=2)
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2)
            
        plt.xlim([K_support[0], K_support[-1]])    
        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane filter (MOhm/ms)")        
        
        plt.tight_layout()

        plt.show()
      
      
    def printParameters(self):

        """
        Print model parameters on terminal.
        """

        print "\n-------------------------"        
        print "GIF model parameters:"
        print "-------------------------"
        print "tau_m (ms):\t%0.3f"  % (self.C/self.gl)
        print "R (MOhm):\t%0.3f"    % (1.0/self.gl)
        print "C (nF):\t\t%0.3f"    % (self.C)
        print "gl (nS):\t%0.6f"     % (self.gl)
        print "El (mV):\t%0.3f"     % (self.El) 
        print "gbar_K1:\t%0.6f"     % (self.gbar_K1)
        print "gbar_K2:\t%0.6f"     % (self.gbar_K2)        
        print "-------------------------\n"
                  

    @classmethod
    def compareModels(cls, GIFs, labels=None):

        """
        Given a list of GIF models, GIFs, the function produce a plot in which the model parameters are compared.
        """

        # PRINT PARAMETERS        

        print "\n#####################################"
        print "GIF model comparison"
        print "#####################################\n"
        
        cnt = 0
        for GIF in GIFs :
            
            #print "Model: " + labels[cnt]          
            GIF.printParameters()
            cnt+=1

        print "#####################################\n"                
                
        # PLOT PARAMETERS
        plt.figure(facecolor='white', figsize=(9,8)) 
               
        colors = plt.cm.jet( np.linspace(0.7, 1.0, len(GIFs) ) )   
        
        # Membrane filter
        plt.subplot(111)
            
        cnt = 0
        for GIF in GIFs :
            
            K_support = np.linspace(0,150.0, 1500)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color=colors[cnt], lw=2)
            cnt += 1
            
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  


        plt.subplots_adjust(left=0.08, bottom=0.10, right=0.95, top=0.93, wspace=0.25, hspace=0.25)
        
        plt.show()
    
 
    @classmethod
    def plotAverageModel(cls, GIFs):

        """
        Average model parameters and plot summary data.
        """
                   
        #######################################################################################################
        # PLOT PARAMETERS
        #######################################################################################################        
        
        fig = plt.figure(facecolor='white', figsize=(16,7))  
        fig.subplots_adjust(left=0.07, bottom=0.08, right=0.95, top=0.90, wspace=0.35, hspace=0.5)   
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
       
       
        # MEMBRANE FILTER
        #######################################################################################################
        
        plt.subplot(2,4,1)
                    
        K_all = []
        
        for GIF in GIFs :
                      
            K_support = np.linspace(0,150.0, 300)             
            K = 1./GIF.C*np.exp(-K_support/(GIF.C/GIF.gl))     
            plt.plot(K_support, K, color='0.3', lw=1, zorder=5)
            
            K_all.append(K)

        K_mean = np.mean(K_all, axis=0)
        K_std  = np.std(K_all, axis=0)
        
        plt.fill_between(K_support, K_mean+K_std,y2=K_mean-K_std, color='gray', zorder=0)
        plt.plot(K_support, np.mean(K_all, axis=0), color='red', lw=2, zorder=10)  
        plt.plot([K_support[0], K_support[-1]], [0,0], ls=':', color='black', lw=2, zorder=-1)   
                
        Tools.removeAxis(plt.gca(), ['top', 'right'])
        plt.xlim([K_support[0], K_support[-1]])
        plt.xlabel('Time (ms)')
        plt.ylabel('Membrane filter (MOhm/ms)')  

 
      
        # R
        #######################################################################################################
    
        plt.subplot(4,6,12+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = 1./GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('R (MOhm)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])        
        
        
        # tau_m
        #######################################################################################################
    
        plt.subplot(4,6,18+1)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.C/GIF.gl
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('tau_m (ms)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
   
        # El
        #######################################################################################################
    
        plt.subplot(4,6,12+2)
 
        p_all = []
        for GIF in GIFs :
                
            p = GIF.El
            p_all.append(p)
            
        plt.hist(p_all, histtype='bar', color='red', ec='white', lw=2)
        plt.xlabel('El (mV)')
        Tools.removeAxis(plt.gca(), ['top', 'left', 'right'])
        plt.yticks([])     
       
        