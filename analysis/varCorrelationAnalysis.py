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
import seaborn as sns


#%% EXPORT DATA TO CSV
"""
Running this block writes X_matrices and Y_vectors for each cell to csv files.
This is so R can be used to experiment with different models.
"""

for i in range(len(KCond_GIFs)):
    
    KGIF = KCond_GIFs[i]
    expt = experiments[i]
    
    tr = expt.trainingset_traces[0]
    X_matrix, Y_vector = KGIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)
    
    data_matrix = np.concatenate((Y_vector[:, np.newaxis], X_matrix), axis = 1)
    data_df = pd.DataFrame(data_matrix)
    data_df.to_csv('../data/matrices/c{}mat.csv'.format(i), index = False)


#%% DEFINE CLASS FOR FITTING LINEAR MODELS
    
"""
Simple class that performs multiple linear regression on its inputs.
Analogous to R's `lm` function.
"""

class linear_model(object):
    
    def __init__(self, y, X):
        
        """
        Fit a linear model.
        
        Inputs:
            y -- dependent variable vector
            X -- independent variable matrix (cols as vars)
        """
        
        # Set core attributes.
        self.y = np.array(y)
        self.X = np.array(X)
        
        self.b = None
        self.yhat = None
        self.residuals = None
        self.var_explained = None
        
        
        # Perform regression.
        XTX = np.dot(self.X.T, self.X)
        XTX_inv = np.linalg.inv(XTX)
        XTY = np.dot(self.X.T, self.y)
        self.b = np.dot(XTX_inv, XTY)
        
        # Get residuals.
        self.yhat = np.dot(self.X, self.b)
        self.residuals = self.yhat - self.y
        
        # Get fraction of var explained.
        SS_tot = np.sum((self.y - self.y.mean()) ** 2)
        SS_res = np.sum(self.residuals ** 2)
        self.var_explained = 1. - SS_res/SS_tot
        
    
    def getVIF(self):
        
        """
        Get variance inflation factor for y given X
        
        VIF = 1 / (1 - var_explained)
        where var_explained is the multiple R^2 of y on X.
        
        VIF > 5-10 is sometimes taken as excessive correlation between independent variables in a multiple linear regression.
        """
        
        return 1. / (1. - self.var_explained)
    

#%% EXAMINE CORRELATIONS BETWEEN MODEL VARIABLES

"""
This block takes the X_matrix built by GIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector
and looks for correlations across columns. Substantial covariance between
model variables may lead to nonsensical coefficient estimates.
"""

# Set script varibles.
show_VIFs = True
show_correlation_matrices = False
show_PCA = False

# Initialize lists to hold variance inflation factors
if show_VIFs:
    VIF_gk1_alone = []
    VIF_gk2_alone = []
    VIF_gk1_all = []
    VIF_gk2_all = []
# Correlation matrices.
for i in range(len(KCond_GIFs)):
    
    # Set local variables.
    KGIF = KCond_GIFs[i]
    expt = experiments[i]
    
    # Build X-matrix for first trace of training set.
    # (Usually training set is only one trace.)
    tr = expt.trainingset_traces[0]
    X_matrix, Y_vector = KGIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr)
    
    # Collect VIFs for each model.
    if show_VIFs:
        X = X_matrix.copy()
        VIF_gk1_alone.append(linear_model(X[:, 3], X[:, [0, 1, 2]]).getVIF())
        VIF_gk2_alone.append(linear_model(X[:, 4], X[:, [0, 1, 2]]).getVIF())
        VIF_gk1_all.append(linear_model(X[:, 3], X[:, [0, 1, 2, 4]]).getVIF())
        VIF_gk2_all.append(linear_model(X[:, 4], X[:, [0, 1, 2, 3]]).getVIF())
    
    # Correlation matrices.
    if show_correlation_matrices:
        X_DF = pd.DataFrame(X_matrix)
        
        X_DF_norm = X_DF.copy()
        for j in range(X_DF_norm.shape[1]):
            X_DF_norm.iloc[:, j] -= X_DF_norm.iloc[:, j].mean()
            X_DF_norm.iloc[:, j] /= X_DF_norm.iloc[:, j].std()
        
        
        plt.matshow(X_DF.corr(), cmap = 'bwr', vmin = -1, vmax = 1)
        plt.xticks([0, 1, 2, 3, 4], ['V', 'I', 'const', 'gating_k1', 'gating_k2'])
        plt.yticks([0, 1, 2, 3, 4], ['V', 'I', 'const', 'gating_k1', 'gating_k2'])
        plt.colorbar()
    

    # PCA on model variables.
    if show_PCA:
        
        X_matrix_PCA = X_matrix[:, [0, 1, 3, 4]]
        
        PCA_results = PCA(X_matrix_PCA)

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
        plt.xticks(ind, ['V', 'I', 'gating_k1', 'gating_k2'])
        plt.xlabel('Coefficient')
        plt.legend()
        
        plt.tight_layout()

# Make plot of VIFs
if show_VIFs:
    
    plt.figure()
    ax = plt.subplot(111)
    
    plt_df = pd.DataFrame(np.array([VIF_gk1_alone,
                                    VIF_gk2_alone,
                                    VIF_gk1_all,
                                    VIF_gk2_all]).T)
    sns.swarmplot(data = plt_df, color = (0.1, 0.1, 0.1), size = 10, alpha = 0.7, ax = ax)
    
    plt.axhline(5, color = 'k', linestyle = 'dashed', linewidth = 0.5)
    
    plt.xticks(np.arange(0, 4), 
               ['gk1 vs. base', 'gk2 vs. base',
                'gk1 vs. base + gk2', 'gk2 vs. base + gk1'],
               rotation = 45)
    plt.xlim(-0.5, 3.5)
    plt.ylim(1, plt.ylim()[1])
    
    plt.ylabel('Variance inflation factor')
    plt.xlabel('Parameter and reference model')
    
    plt.tight_layout()

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