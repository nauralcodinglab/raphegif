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


    def scrapeTrainingData(self, experiment):

        """
        Scrape training data from experiment.
        """

        if experiment.dt != self.dt:
            raise ValueError('dt mismatch')

        self.V_train = np.array([tr.V for tr in experiment.trainingset_traces]).T
        self.I_train = np.array([tr.I for tr in experiment.trainingset_traces]).T
        self.V_test = np.array([tr.V for tr in experiment.testset_traces]).T
        self.I_test = np.array([tr.I for tr in experiment.testset_traces]).T

        self.dV_train = np.gradient(self.V_train, axis = 0)


    def computeTrainingGating(self, model):

        """
        Compute KCond gating vectors based on training data and a SubthreshGIF_K model.
        """

        if model.dt != self.dt:
            raise ValueError('dt mismatch')

        V_tr_ls = [np.array(V) for V in self.V_train.T.tolist()] # Turn data into a list of vectors.
        self.m_train = np.array([model.computeGating(V, model.mInf(V), model.m_tau) for V in V_tr_ls]).T
        self.h_train = np.array([model.computeGating(V, model.hInf(V), model.h_tau) for V in V_tr_ls]).T
        self.n_train = np.array([model.computeGating(V, model.nInf(V), model.n_tau) for V in V_tr_ls]).T


    def pickle(self, fname):

        """
        Pickle the current ModMat instance.
        """

        with open(fname, 'wb') as f:

            pickle.dump(self, f)
