#%% IMPORT MODUELS

import pickle

import numpy as np

#%% CREATE CUTSTOM CLASS TO HOLD DATA
class ModMats(object):

    """
    Class to hold X and Y data that can be used to easily fit a linear model to an individual experiment.
    Automagically creates numpy arrays to hold training and test data (rows as timesteps and cols as sweeps) along with KCond gating vectors.
    """

    def __init__(self, dt):

        self.dt = dt

        self.VCutoff = None
        self.__VCutoffVec = None


    def scrapeTraces(self, experiment, use_ROI = True):

        """
        Scrape training data from experiment.
        """

        if experiment.dt != self.dt:
            raise ValueError('dt mismatch')

        if use_ROI:
            self.V_train = np.array([tr.V[tr.getROI()] for tr in experiment.trainingset_traces]).T
            self.I_train = np.array([tr.I[tr.getROI()] for tr in experiment.trainingset_traces]).T
            self.V_test = np.array([tr.V[tr.getROI()] for tr in experiment.testset_traces]).T
            self.I_test = np.array([tr.I[tr.getROI()] for tr in experiment.testset_traces]).T
        else:
            self.V_train = np.array([tr.V for tr in experiment.trainingset_traces]).T
            self.I_train = np.array([tr.I for tr in experiment.trainingset_traces]).T
            self.V_test = np.array([tr.V for tr in experiment.testset_traces]).T
            self.I_test = np.array([tr.I for tr in experiment.testset_traces]).T

        self.dV_train = np.gradient(self.V_train, axis = 0) / self.dt


    def computeTrainingGating(self, model):

        """
        Compute KCond gating vectors based on training data and a SubthreshGIF_K model.
        """

        if model.dt != self.dt:
            raise ValueError('dt mismatch')

        self.E_K = model.E_K

        V_tr_ls = [np.array(V) for V in self.V_train.T.tolist()] # Turn data into a list of vectors.
        self.m_train = np.array([model.computeGating(V, model.mInf(V), model.m_tau) for V in V_tr_ls]).T
        self.h_train = np.array([model.computeGating(V, model.hInf(V), model.h_tau) for V in V_tr_ls]).T
        self.n_train = np.array([model.computeGating(V, model.nInf(V), model.n_tau) for V in V_tr_ls]).T


    def setVCutoff(self, cutoff):

        """
        Set a cutoff voltage below which points are flagged for optional exclusion from regression matrices.
        """

        self.__VCutoffVec = self.V_train.flatten() > cutoff
        self.VCutoff = cutoff


    def _getYVector_XMatrix(self, subset = True):

        """
        Build matrices for linear regression.

        Returns tupple of Y-vector and X-matrix for multiple linear regression.
        X is returned as a T x 5 matrix with columns [I, -V, ones, -mh(V - Ek), -n(V - Ek)], where mh is the gating state of IA and n is the gating state of KSlow.
        If subset is true, elements of Y/rows of X where V is below a set cutoff (set using ModMats.setVCutoff).
        """

        if self.__VCutoffVec is None:
            raise RuntimeError('Cannot subset if no V cutoff has been set.')

        K_driving_force = self.V_train - self.E_K

        Y = self.dV_train.flatten()
        X = np.concatenate((
            self.I_train.flatten()[:, np.newaxis],
            -self.V_train.flatten()[:, np.newaxis],
            np.ones(len(self.V_train.flatten()), dtype = np.float64)[:, np.newaxis],
            -(self.m_train * self.h_train).flatten()[:, np.newaxis] * K_driving_force,
            -self.n_train.flatten()[:, np.newaxis] * K_driving_force
        ), axis = 1)

        if subset:
            return (Y[self.__VCutoffVec], X[self.__VCutoffVec, :])
        else:
            return (Y, X)


    @staticmethod
    def _fitLinMod(Y, X):

        """
        Perform linear regression of Y on X.

        Takes a vector of length m for Y and an m x n matrix for X.
        Returns an n-length vector of coefficients.
        """

        XTX_inv = np.linalg.inv( np.dot(X.T, X) )
        XTY = np.dot(X.T, Y)
        b = np.dot(XTX_inv, XTY).flatten()

        return b

    @staticmethod
    def _getR2_Train(Y, X, b):

        Y_var = np.var(Y)
        Y_est = np.dot(X, b)

        return (Y_var - np.mean((Y - Y_est)**2)) / Y_var


    def fitOhmicMod(self, subset = True):

        Y, X = self._getYVector_XMatrix(subset = subset)

        # Use only relevant parts of X.
        X = X[:, :3]

        b = self._fitLinMod(Y, X)

        C = 1/b[0]
        gl = b[1] * C
        El = b[2] * C / gl

        output_dict = {
        'El': El,
        'R': 1/gl,
        'C': C,
        'var_explained_dV': self._getR2_Train(Y, X, b)
        }

        return output_dict


    def fitGK1Mod(self, subset = True):

        Y, X = self._getYVector_XMatrix(subset = subset)

        # Use only relevant columns of X
        X = X[:, :4]

        b = self._fitLinMod(Y, X)

        C = 1/b[0]
        gl = b[1] * C
        El = b[2] * C / gl
        gbar_K1 = b[3] * C

        output_dict = {
        'El': El,
        'R': 1/gl,
        'C': C,
        'gbar_K1': gbar_K1,
        'var_explained_dV': self._getR2_Train(Y, X, b)
        }

        return output_dict


    def fitGK2Mod(self, subset = True):

        Y, X = self._getYVector_XMatrix(subset = subset)

        # Use only relevant columns of X
        X = X[:, [0, 1, 2, 4]]

        b = self._fitLinMod(Y, X)

        C = 1/b[0]
        gl = b[1] * C
        El = b[2] * C / gl
        gbar_K2 = b[3] * C

        output_dict = {
        'El': El,
        'R': 1/gl,
        'C': C,
        'gbar_K2': gbar_K2,
        'var_explained_dV': self._getR2_Train(Y, X, b)
        }

        return output_dict


    def fitFullMod(self, subset = True):

        Y, X = self._getYVector_XMatrix(subset = subset)

        # Use only relevant columns of X
        X = X[:, :]

        b = self._fitLinMod(Y, X)

        C = 1/b[0]
        gl = b[1] * C
        El = b[2] * C / gl
        gbar_K1 = b[3] * C
        gbar_K2 = b[4] * C

        output_dict = {
        'El': El,
        'R': 1/gl,
        'C': C,
        'gbar_K1': gbar_K1,
        'gbar_K2': gbar_K2,
        'var_explained_dV': self._getR2_Train(Y, X, b)
        }

        return output_dict


    def pickle(self, fname):

        """
        Pickle the current ModMat instance.
        """

        with open(fname, 'wb') as f:

            pickle.dump(self, f)
