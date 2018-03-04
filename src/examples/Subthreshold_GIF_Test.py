import sys
sys.path.append('../')

from Experiment import *
from AEC_Badel import *
from SubthreshGIF import *
from Filter_Rect_LogSpaced import *
from Filter_Rect_LinSpaced import *

import matplotlib.pyplot as plt


"""
This file tests whether the current implementation of the GIF model can be
fitted to recordings without any spikes.
"""

############################################################################################################
# STEP 1: LOAD EXPERIMENTAL DATA
############################################################################################################
myExp = Experiment('Fully subthreshold test', 0.1)

PATH = '../../data/subthreshold_gif_test/'

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
# STEP 2: LOWPASS FILTER VOLTAGE TRACES
############################################################################################################

for tr in myExp.trainingset_traces:
    
    tr.butterLowpassFilter(cutoff = 3000.)
    

"""    
for tr in myExp.testset_traces:
    
    tr.butterLowpassFilter(cutoff = 3000.)
"""

############################################################################################################
# STEP 3: ACTIVE ELECTRODE COMPENSATION
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
# STEP 4: TEST SPIKE DETECTION AND ROI SELECTION IN Trace
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


# Test boolean ROI selection.
myExp.trainingset_traces[0].setROI([[1000, 50000]])

above_m70 = myExp.trainingset_traces[0].V > -70.

#myExp.trainingset_traces[0].setROI_Bool(above_m70)
myExp.trainingset_traces[0].plot()

############################################################################################################
# STEP 5: FIT GIF MODEL TO DATA
############################################################################################################

# Create a new object GIF 
myGIF = SubthreshGIF(0.1) 


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

# Plot fit.
myGIF.plotFit()

## Save the model
#myGIF.save('./myGIF.pck')


############################################################################################################
# STEP 6: EXTRACT POWER SPECTRUM DENSITY AND COMPARE BETWEEN MODEL AND DATA
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