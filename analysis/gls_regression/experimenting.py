#%% IMPORT MODULES

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt


#%% GENERATE SOME DATA

x = np.random.normal(-5, 5, size = no_pts)

def generate_data(x, no_pts = 1000):

    sigmoid = lambda x: 1/ (1 + np.exp(-(x - 10)))

    X = np.array([np.ones(no_pts), x, sigmoid(x)]).T
    beta = np.array([5, 10, 100])

    y = np.dot(X, beta)
    y_noisy = y + np.random.normal(0, 10, size = no_pts)

    return X, y_noisy

X, y_noisy = generate_data(x)

plt.figure()
plt.plot(x, y_noisy, 'k.')
plt.show()


#%% ATTEMPT TO FIT TERMS

def ols_fit(X, y):
    XTX_inv = np.linalg.inv(np.dot(X.T, X))
    XTY = np.dot(X.T, y)
    beta_hat = np.dot(XTX_inv, XTY)

    return beta_hat

betas = []
for i in range(5000):

    x = np.random.normal(-5, 5, size = no_pts)
    X, y_noisy = generate_data(x)

    beta_hat = ols_fit(X, y_noisy)
    betas.append(beta_hat)

betas = np.array(betas)
print(betas.mean(axis = 0))
print(betas.std(axis = 0))
