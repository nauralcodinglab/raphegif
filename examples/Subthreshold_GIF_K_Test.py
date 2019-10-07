import numpy as np
import matplotlib.pyplot as plt

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.SubthreshGIF import SubthreshGIF_K
from grr.Filter_Rect import Filter_Rect_LogSpaced
from grr.Filter_Rect import Filter_Rect_LinSpaced


"""
This file tests whether the K_conductances added to SubthreshGIF work.
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Fully subthreshold K test', 0.1)

PATH = '../data/subthreshold_gif_test/'

# Load AEC data
myExp.setAECTrace(FILETYPE = 'Axon',
                  fname = PATH + 'subthresh_AEC.abf', V_channel = 0, I_channel = 1)

# Load training set data
myExp.addTrainingSetTrace(FILETYPE = 'Axon',
                          fname = PATH + 'subthresh_train.abf', V_channel = 0, I_channel = 1)

# Load test set data
myExp.addTestSetTrace(FILETYPE = 'Axon',
                      fname = PATH + 'subthresh_test.abf', V_channel = 0, I_channel = 1)

# Plot data
#myExp.plotTrainingSet()
#myExp.plotTestSet()


############################################################################################################
# STEP 2: ACTIVE ELECTRODE COMPENSATION
############################################################################################################

# Create new object to perform AEC
myAEC = AEC_Badel(myExp.dt)

# Define metaparametres
myAEC.K_opt.setMetaParameters(length=150.0, binsize_lb=myExp.dt, binsize_ub=2.0, slope=30.0, clamp_period=1.0)
myAEC.p_expFitRange = [3.0,150.0]
myAEC.p_nbRep = 15

# Assign myAEC to myExp and compensate the voltage recordings
myExp.setAEC(myAEC)
myExp.performAEC()

# Plot AEC filters (Kopt and Ke)
myAEC.plotKopt()
myAEC.plotKe()

# Plot training and test set
myExp.plotTrainingSet()
myExp.plotTestSet()


############################################################################################################
# STEP 3: TEST SPIKE DETECTION AND ROI SELECTION IN Trace
############################################################################################################

# Test spike detection
print '\nTesting spike detection...'
try:
    myExp.trainingset_traces[0].detectSpikes()
except:
    print('Spike detection test failed.\n')
    raise
print 'Success!\n'

# Test ROI selection
print '\nTesting ROI selection...'
try:
    myExp.trainingset_traces[0].getROI_FarFromSpikes(5., 5.)
except:
    print('getROI_FarFromSpikes test failed.\n')
    raise
print('Success!\n')


############################################################################################################
# STEP 4: TEST K_CONDUCTANCES
############################################################################################################

# Create a new object GIF
myGIF = SubthreshGIF_K(0.1)

# Define parameters
myGIF.m_A = 1.61
myGIF.m_Vhalf = -27
myGIF.m_k = 0.113
myGIF.m_tau = 1.

myGIF.h_A = 1.03
myGIF.h_Vhalf = -59.2
myGIF.h_k = -0.165
myGIF.h_tau = 50.

myGIF.n_A = 1.55
myGIF.n_Vhalf = -16.9
myGIF.n_k = 0.114
myGIF.n_tau = 100.

myGIF.E_K = -101.

# Test activation curves.
V_vec = np.arange(-100, 0, 0.1)

plt.figure()

plt.subplot(111)
plt.title('Equilibrium gating states for K_conductances')
plt.plot(V_vec, myGIF.mInf(V_vec), label = 'm')
plt.plot(V_vec, myGIF.hInf(V_vec), label = 'h')
plt.plot(V_vec, myGIF.nInf(V_vec), label = 'n')
plt.legend()

plt.ylabel('g')
plt.xlabel('V (mV)')

# Test time-dependence.
t = np.arange(-150, 400, myGIF.dt)
V_vec = np.heaviside(t, 1.) * 100 - 100

plt.figure(figsize = (10, 5))

ax0 = plt.subplot(211)
plt.title('Response of K_conductance gates to voltage step')
plt.plot(t, myGIF.computeGating(V_vec, myGIF.mInf(V_vec), myGIF.m_tau),
         label = 'm')
plt.plot(t, myGIF.computeGating(V_vec, myGIF.hInf(V_vec), myGIF.h_tau),
         label = 'h')
plt.plot(t, myGIF.computeGating(V_vec, myGIF.nInf(V_vec), myGIF.n_tau),
         label = 'n')

plt.ylabel('g')
plt.legend()


plt.subplot(212, sharex = ax0)
plt.plot(t, V_vec, 'k-')
plt.ylabel('V (mV)')
plt.xlabel('Time (ms)')

plt.show()

############################################################################################################
# STEP 4: FIT GIF MODEL TO DATA
############################################################################################################

# Define the ROI of the training set to be used for the fit (in this example we will use only the first 100 s)
myExp.trainingset_traces[0].setROI([[2000,58000]])

# To visualize the training set and the ROI call again
myExp.plotTrainingSet()

# Perform the fit
print '\nTesting model fitting...'
try:
    myGIF.fit(myExp)
except:
    print 'Model fit test failed.\n'
    raise
print '\nSuccessful model fit!\n'

# Plot the model parameters
myGIF.printParameters()
myGIF.plotParameters()

# Plot the model fit on the training data.
myGIF.plotFit()

# Plot simulated gating parameters.
myGIF.plotGating()

## Save the model
#myGIF.save('./myGIF.pck')


############################################################################################################
# STEP 5: EXTRACT POWER SPECTRUM DENSITY AND COMPARE BETWEEN MODEL AND DATA
############################################################################################################

# Test PSD extraction from data Trace
print '\nTesting PSD extraction...'
try:
    myExp.trainingset_traces[0].extractPowerSpectrumDensity(True)
except:
    print 'PSD extraction test failed.\n'
    raise
print '\nSuccessful PSD extraction!\n'

# Compare PSD of data and GIF model
myGIF.plotPowerSpectrumDensity()


############################################################################################################
# STEP 6: TEST SIMULATED VOLTAGE CLAMP
############################################################################################################

#%%
print '\nTesting simulated voltage clamp...'
try:
    myGIF.simulateVClamp(1000, -40, -90, False)
except:
    print 'Voltage clamp test failed.\n'
    raise
print '\nSuccessful simulated voltage clamp!'

plt.figure(figsize = (6, 4))
plt.subplot(111)
plt.title('Simulated voltage clamp test')
plt.ylabel('Holding current (nA)')
plt.xlabel('Time (ms)')

for V in np.arange(-60, -20, 10):

    I_vec = myGIF.simulateVClamp(500, V, -90, True)[1]
    t_vec = np.arange(0, int(np.round(len(I_vec) * myGIF.dt)), myGIF.dt)

    plt.plot(t_vec, I_vec, label = str(V) + 'mV')

plt.legend()


############################################################################################################
# STEP 4A (OPTIONAL): PLAY A BIT WITH THE FITTED MODEL
############################################################################################################

## Reload the model
#myGIF = GIF.load('./myGIF.pck')
#
## Generate OU process with temporal correlation 3 ms and mean modulated by a sinusoildal function of 1 Hz
#I_OU = Tools.generateOUprocess_sinMean(f=1.0, T=5000.0, tau=3.0, mu=0.3, delta_mu=0.5, sigma=0.1, dt=0.1)
#
## Simulate the model with the I_OU current. Use the reversal potential El as initial condition (i.e., V(t=0)=El)
#(time, V, I_a, V_t, S) = myGIF.simulate(I_OU, myGIF.El)
#
## Plot the results of the simulation
#plt.figure(figsize=(14,5), facecolor='white')
#plt.subplot(2,1,1)
#plt.plot(time, I_OU, 'gray')
#plt.ylabel('I (nA)')
#plt.subplot(2,1,2)
#plt.plot(time, V,'black', label='V')
#plt.plot(time, V_t,'red', label='V threshold')
#plt.ylabel('V (mV)')
#plt.xlabel('Time (ms)')
#plt.legend()
#plt.show()