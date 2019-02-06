#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb

#%%

if __name__ == '__main__':
    no_vars = 2
    no_obs = int(1e4)

    X = np.random.uniform(-1, 1, size = (no_obs, no_vars))
    beta = np.random.normal(size = no_vars)
    y = np.dot(X, beta) + np.random.normal(size = no_obs)

    def OLS_fit(X, y):
        XTX = np.dot(X.T, X)
        XTY = np.dot(X.T, y)
        betas = np.linalg.solve(XTX, XTY)
        return betas

    beta_est = OLS_fit(X, y)

    print 'True beta: {:6.3f}{:8.3f}'.format(beta[0], beta[1])
    print 'Est. beta: {:6.3f}{:8.3f}'.format(beta_est[0], beta_est[1])

#%%

def cosine_similarity(X, B):

    if X.shape[1] == B.shape[1]:
        XB = np.dot(X, B.T)
        X_norm = np.broadcast_to(np.linalg.norm(X, axis = 1)[:, np.newaxis], XB.shape)
        B_norm = np.broadcast_to(np.linalg.norm(B, axis = 1), XB.shape)
    elif X.shape[1] == B.shape[0]:
        XB = np.dot(X, B)
        X_norm = np.broadcast_to(np.linalg.norm(X, axis = 1)[:, np.newaxis], XB.shape)
        B_norm = np.broadcast_to(np.linalg.norm(B, axis = 0), XB.shape)
    else:
        raise ValueError('Dimensionality of X and B do not match.')

    return XB / (X_norm * B_norm)

def subsample_cs(X, y, B):

    cs = cosine_similarity(X, B)
    inds = np.argmax(cs, axis = 0)
    X_points = X[inds, :]
    y_points = y[inds]

    return X_points, y_points, inds

if __name__ == '__main__':
    OLS_est = OLS_fit(X, y)
    beta_est = []

    no_reps = 100
    np.random.seed(24)
    for i in range(no_reps):
        print '\r{:.1f}'.format(100*i/no_reps),

        no_bs_pts = int(no_obs * 0.25)
        bs_pts = np.random.uniform(-1, 1, size = (no_bs_pts, no_vars))
        bs_X, bs_y, inds = subsample_cs(X, y, bs_pts)

        beta_est.append(OLS_fit(bs_X, bs_y))

    beta_est = np.array(beta_est)

    print('\rDone!')


    plt.figure()
    plt.subplot(121)
    plt.hist(beta_est[:, 0])
    plt.axvline(beta[0], color = 'k')
    plt.axvline(OLS_est[0], color = 'gray')
    plt.axvline(beta_est.mean(axis = 0)[0], color = 'r')

    plt.subplot(122)
    plt.hist(beta_est[:, 1])
    plt.axvline(beta[1], color = 'k')
    plt.axvline(OLS_est[1], color = 'gray')
    plt.axvline(beta_est.mean(axis = 0)[1], color = 'r')
    plt.show()

#%%
def convergence_plot(bs_estimates, OLS_estimates, ground_truth = None, random_seed = 42):

    np.random.seed(random_seed)

    N = bs_estimates.shape[0]

    # Get means and stds of subsample of estimates
    k_vec = []
    means_vec = []
    stds_vec = []

    for k in range(1, N):
        estimate_subset = []
        for i in range(N):
            estimate_subset.append(
                np.random.permutation(bs_estimates)[:k, :].mean(axis = 0)
            )
        k_vec.append(k)
        means_vec.append(np.mean(estimate_subset, axis = 0))
        stds_vec.append(np.std(estimate_subset, axis = 0))

    # Convert to numpy arrays
    k_vec = np.array(k_vec)
    means_vec = np.array(means_vec)
    stds_vec = np.array(stds_vec)


    # Create figure
    plt.figure(figsize = (6, 4))

    for i in range(len(OLS_estimates)):
        plt.subplot(1, len(OLS_estimates) + 1, i + 1)
        plt.fill_between(
            k_vec,
            means_vec[:, i] - stds_vec[:, i],
            means_vec[:, i] + stds_vec[:, i]
        )
        plt.plot(k_vec, means_vec[:, i], 'r-')
        plt.axhline(OLS_estimates[i], color = 'r', ls = '--', label = 'OLS estimate')
        if ground_truth is not None:
            plt.axhline(ground_truth[i], color = 'k', label = 'Ground truth')
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.show()

    return k_vec, means_vec, stds_vec

if __name__ == '__main__':
    _ = convergence_plot(beta_est, OLS_est, beta)
