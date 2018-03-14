"""
Look for correlations among model coefficients and/or variables.

Run SubthresholdGIFAnalysis.py first to import data and place in appropriate
dicts.
"""

#%% CHECK THAT SubthresholdGIFAnalysis.py has been run first.

try:
    Base_GIFs
    KCond_GIFs
except NameError:
    raise NameError('Names of required objects are not defined. '
                    'Did you run SubthresholdGIFAnalysis.py first?')


#%% IMPORT PACKAGES

from matplotlib.mlab import PCA
import pandas as pd

#%% EXAMINE CORRELATIONS BETWEEN COEFFICIENTS

# Gather coefficients

gk_leak_pdata = []
gbase_leak_pdata = []
gk1_pdata = []
gk2_pdata = []

try:
    for KGIF in KCond_GIFs:
        
        gk_leak_pdata.append(KGIF.gl)
        gk1_pdata.append(KGIF.gbar_K1)
        gk2_pdata.append(KGIF.gbar_K2)
    
    for GIF in Base_GIFs:
        
        gbase_leak_pdata.append(GIF.gl)

except NameError:
    raise NameError('Names of required objects are not defined. '
                    'Did you run SubthresholdGIFAnalysis.py first?')
    
# Make a simple correlation matrix.
    
data_mat = np.array([gbase_leak_pdata, gk_leak_pdata, gk1_pdata, gk2_pdata]).T
data_DF = pd.DataFrame(data_mat)

plt.matshow(data_DF.corr(), cmap = 'bwr', vmin = -1, vmax = 1)
plt.xticks([0, 1, 2, 3], ['gl base', 'gl K', 'gk1', 'gk2'])
plt.yticks([0, 1, 2, 3], ['gl base', 'gl K', 'gk1', 'gk2'])
plt.colorbar()


# Perform PCA.

PCA_results = PCA(data_mat)

print PCA_results.fracs
print PCA_results.Wt

plt.figure(figsize = (10, 5))

plt.subplot(121)
plt.title('Variance explained by component')
plt.plot(100. * PCA_results.fracs, 'ko-')
plt.ylabel('Variance explained (%)')
plt.xticks([0, 1, 2, 3], ['1st', '2nd', '3rd', '4th'])
plt.xlabel('Component')
plt.ylim(-5, 105)

plt.subplot(122)
plt.title('Component coefficients')
ind = np.arange(PCA_results.Wt.shape[1])
wid = 0.35
plt.bar(ind - wid/2., PCA_results.Wt[0, :], wid, label = '1st component')
plt.bar(ind + wid/2., PCA_results.Wt[1, :], wid, label = '2nd component')
plt.ylabel('Weight')
plt.xticks(ind, ['gl base', 'gl K', 'gk1', 'gk2'])
plt.xlabel('Coefficient')
plt.legend()

plt.tight_layout()