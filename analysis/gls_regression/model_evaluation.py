#%% IMPORT MODULES

from __future__ import division

import sys

import numpy as np
import scipy.stats as stats

sys.path.append('./src')

from AugmentedGIF import AugmentedGIF


#%%

def cross_validate(X, y, fitting_func, k = 10, random_seed = 42, verbose = False, include_proportion = True):
    """Binned cross-validated train and test error.
    """

    np.random.seed(random_seed)
    inds = np.random.permutation(len(y))
    groups = np.array_split(inds, k)

    var_explained_dict = {
        'train': [],
        'test': []
    }

    for i in range(k):

        if verbose:
            if i == 0:
                print '\n'
            print '\rCross-validating {:.1f}%'.format(100 * i / k),

        test_inds_tmp = groups[i]
        train_inds_tmp = np.concatenate([gr for j, gr in enumerate(groups) if j != i], axis = None)

        train_y = y[train_inds_tmp]
        train_X = X[train_inds_tmp, :]

        test_y = y[test_inds_tmp]
        test_X = X[test_inds_tmp, :]

        betas = fitting_func(train_X, train_y)
        var_explained_dict['train'].append(var_explained_binned(train_X, betas, train_y, 'default'))
        var_explained_dict['test'].append(var_explained_binned(test_X, betas, test_y, 'default'))

    var_explained_dict['train'] = np.mean(var_explained_dict['train'], axis = 0)
    var_explained_dict['test'] = np.mean(var_explained_dict['test'], axis = 0)

    if verbose:
        print '\rDone!                 '

    return var_explained_dict['train'], var_explained_dict['test']

def build_Xy(expt_, excl_cols = None, GIFmod = AugmentedGIF):

    X = []
    y = []

    tmpGIF = GIFmod(0.1)
    tmpGIF.Tref = 6.5

    for tr in expt_.trainingset_traces:
        X_tmp, y_tmp = tmpGIF.fitSubthresholdDynamics_Build_Xmatrix_Yvector(tr, 1.5)

        X.append(X_tmp)
        y.append(y_tmp)

    X = np.concatenate(X)
    if excl_cols is not None:
        X = X[:, [x for x in range(X.shape[1]) if x not in excl_cols]]
    y = np.concatenate(y)

    return X, y

def convert_betas(coeffs_df):
    """Transform a DataFrame of coefficient estimates into sensible model parameters.

    Expects first five columns to correspond to gl, C, El, gk1, gk2.
    Expects all further columns to be either 'group' or AHP coefficients.
    """

    AHP_cols = coeffs_df.loc[:, [x != 'group' and int(x) >= 5 for x in coeffs.columns]].rename(
        lambda s: 'AHP{}'.format(int(s) - 5),
        axis = 'columns'
    )

    param_cols = pd.DataFrame()
    param_cols['C'] = 1./coeffs.loc[:, 1]
    param_cols['gl'] = -coeffs.loc[:, 0] * param_cols['C']
    param_cols['El'] = coeffs.loc[:, 2]*param_cols['C']/param_cols['gl']
    param_cols['gk1'] = coeffs.loc[:, 3] * param_cols['C']
    param_cols['gk2'] = coeffs.loc[:, 4] * param_cols['C']

    AHP_cols = AHP_cols.multiply(param_cols['C'], axis = 0)

    try:
        results = pd.concat([coeffs_df['group'], param_cols, AHP_cols], axis = 1)
    except KeyError:
        results = pd.concat([param_cols, AHP_cols], axis = 1)

    return results

def WLS_fit(X, y):
    """Least squares weighted by voltage.
    """
    voltage = X[:, 0]
    #wts = np.exp((voltage - voltage.mean())/ voltage.std())
    wts = np.log(1 + 1.1**(voltage + 50))

    wtsy = wts * y
    wtsX = wts[:, np.newaxis] * X

    XTwtsX = np.dot(X.T, wtsX)
    XTwtsy = np.dot(X.T, wtsy)
    betas = np.linalg.solve(XTwtsX, XTwtsy)

    return betas


def OLS_fit(X, y):
    """Matrix formulation of ordinary least squares fit.
    """
    XTX = np.dot(X.T, X)
    XTY = np.dot(X.T, y)
    betas = np.linalg.solve(XTX, XTY)
    return betas

def var_explained(X, betas, y):
    yhat = np.dot(X, betas)
    MSE = np.mean((y - yhat)**2)
    var = np.var(y)
    return 1. - MSE/var

def var_explained_binned(X, betas, y, bins = 'default', return_proportion = False):
    """
    Returns a tuple of bin centres and binned means if not `return_proportion`.
    Otherwise a tuple (bin_centres, binned_means, proportions) is returned.
    """

    if bins == 'default':
        bins = np.arange(-90, -20, 5)

    V = X[:, 0]

    yhat = np.dot(X, betas)
    squared_errors = (y - yhat)**2

    mean_, edges, bin_no = stats.binned_statistic(
        V, squared_errors, 'mean', bins, [-80, -20]
    )

    if return_proportion:
        cnt_, _, _ = stats.binned_statistic(
            V, None, 'count', bins, [-80, -20]
        )

        proportion = cnt_ / np.sum(cnt_)

    bin_centres = (edges[1:] + edges[:-1]) / 2.

    if return_proportion:
        return (bin_centres, mean_, proportion)
    else:
        return (bin_centres, mean_)
