import abc

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import fftconvolve

from Filter import Filter
from . import Tools


class Filter_Rect(Filter):

    """
    Abstract class for filters defined as linear combinations of rectangular basis functions.
    A filter f(t) is defined in the form

    f(t) = sum_j b_j*rect_j(t),

    where b_j is a set of coefficient and rect_j is a set of non-overlapping rectangular basis functions.

    This class is abstract because it does not specify the kind of rectangular basis functions used in practice.
    Possible implementations could be e.g. linear spacing, log spacing, arbitrary spacing.
    To implement such filters, inherit from Filter_Rect
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self):

        Filter.__init__(self)

        # Auxiliary variables that can be computed using the parameters above

        self.bins = []     # ms, vector defining the rectangular basis functions for f(t)

        self.support = []     # ms, centers of bins used to define the filter

    ############################################################################
    # IMPLEMENT SOME OF THE ABSTRACT METHODS OF FILTER
    ############################################################################

    def getLength(self):
        """
        Return filter length (in ms).
        """

        return self.bins[-1]

    def setFilter_Function(self, f):
        """
        Given a function of time f(t), the bins of the filer are initialized accordingly.
        For example, if f(t) is an exponential function, the filter will approximate an exponential using rectangular basis functions.
        """

        self.computeBins()
        self.filter_coeff = f(self.support)

    def computeInterpolatedFilter(self, dt):
        """
        Given a particular dt, the function compute the interpolated filter as well as its temporal support vector.
        """

        self.computeBins()

        bins_i = Tools.timeToIndex(self.bins, dt)

        if self.filter_coeffNb == len(self.filter_coeff):

            filter_interpol = np.zeros((bins_i[-1] - bins_i[0]))

            for i in range(len(self.filter_coeff)):

                lb = int(bins_i[i])
                ub = int(bins_i[i+1])
                filter_interpol[lb:ub] = self.filter_coeff[i]

            filter_interpol_support = np.arange(len(filter_interpol))*dt

            self.filtersupport = filter_interpol_support
            self.filter = filter_interpol

        else:

            print "Error: value of the filter coefficients does not match the number of basis functions!"

    ###################################################################################
    # OTHER FUNCTIONS
    ###################################################################################

    def computeSupport(self):
        """
        Based on the rectangular basis functions defined in sefl.bins compute self.support
        (ie, the centers of rectangular basis functions).
        """

        self.support = np.array([(self.bins[i]+self.bins[i+1])/2 for i in range(len(self.bins)-1)])

    @abc.abstractmethod
    def computeBins(self):
        """
        Given metaparametres compute bins associated to the rectangular basis functions.
        """


class Filter_Rect_LogSpaced(Filter_Rect):

    """
    This class defines a temporal filter defined as a linear combination of log-spaced rectangular basis functions.
    """

    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):

        Filter_Rect.__init__(self)

        # Metaparamters

        self.p_length = length           # ms, filter length

        self.p_binsize_lb = binsize_lb       # ms, min size for bin

        self.p_binsize_ub = binsize_ub       # ms, max size for bin

        self.p_slope = slope            # exponent for log-scaling

        # Initialize

        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.

        self.setFilter_toZero()              # initialize filter to 0

    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0):
        """
        Set the parameters defining the rectangular basis functions.
        Each time meta parameters are changeD, the value of the filer is reset to 0.
        """

        self.p_length = length                  # ms, filter length

        self.p_binsize_lb = binsize_lb              # ms, min size for bin

        self.p_binsize_ub = binsize_ub              # ms, max size for bin

        self.p_slope = slope                   # exponent for log-scale binning

        self.computeBins()

        self.setFilter_toZero()                     # initialize filter to 0

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################

    def computeBins(self):
        """
        This function compute log-spaced bins and support given the metaparameters.
        """

        self.bins = []
        self.bins.append(0)

        cnt = 1
        total_length = 0

        while (total_length <= self.p_length):
            tmp = min(self.p_binsize_lb*np.exp(cnt/self.p_slope), self.p_binsize_ub)
            total_length = total_length + tmp
            self.bins.append(total_length)

            cnt += 1

        self.bins = np.array(self.bins)

        self.computeSupport()

        self.filter_coeffNb = len(self.bins)-1

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        """
        Filter spike train spks with the set of rectangular basis functions defining the Filter.
        """

        T_i = int(T/dt)

        bins_i = Tools.timeToIndex(self.bins, dt)
        spks_i = Tools.timeToIndex(spks, dt)

        nb_bins = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, nb_bins))

        # Fill matrix
        for l in np.arange(nb_bins):

            tmp = np.zeros(T_i + bins_i[-1] + 1)

            for s in spks_i:
                lb = s + bins_i[l]
                ub = s + bins_i[l+1]
                tmp[lb:ub] += 1

            X[:, l] = tmp[:T_i]

        return X

    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        """
        Filter continuous input I with the set of rectangular basis functions defining the Filter.
        """

        T_i = len(I)

        bins_i = Tools.timeToIndex(self.bins, dt)
        bins_l = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, bins_l))
        I_tmp = np.array(I, dtype='float64')

        # Fill matrix
        for l in np.arange(bins_l):

            window = np.ones(bins_i[l+1] - bins_i[l])
            window = np.array(window, dtype='float64')

            F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
            F_star_I = F_star_I[: int(len(I))]

            F_star_I_shifted = np.concatenate((np.zeros(int(bins_i[l])), F_star_I))

            X[:, l] = np.array(F_star_I_shifted[:T_i], dtype='double')

        return X


class Filter_Rect_LogSpaced_AEC(Filter_Rect_LogSpaced):

    """
    This class define a function of time expanded using log-spaced rectangular basis functions.
    Using the metaparameter p_clamp_period, one can force the rectangular basis functions covering
    the first p_clamp_period ms to have a to have a specific size binsize_lb.
    Log-spacing only starts after p_clamp_period.
    """

    def __init__(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=1.0):

        # Metaparameters

        self.p_clamp_period = clamp_period

        Filter_Rect_LogSpaced.__init__(self, length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)

        # Initialize

        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.

        self.setFilter_toZero()              # initialize filter to 0

    ################################################################
    # OVERVRITE METHODS OF Filter_Rect_LogSpaced
    ################################################################

    def setMetaParameters(self, length=1000.0, binsize_lb=2.0, binsize_ub=1000.0, slope=7.0, clamp_period=10.0):

        # Set metaparameters inherited from  Filter_Rect_LogSpaced

        super(Filter_Rect_LogSpaced_AEC, self).setMetaParameters(length=length, binsize_lb=binsize_lb, binsize_ub=binsize_ub, slope=slope)

        # Set paramters which are specific to this class

        self.p_clamp_period = clamp_period

        self.computeBins()

        self.setFilter_toZero()

    def computeBins(self):
        """
        This function compute bins and support given metaparameters.
        """

        self.bins = []
        self.bins.append(0)

        total_length = 0

        for i in np.arange(int(self.p_clamp_period/self.p_binsize_lb)):
            total_length = total_length + self.p_binsize_lb
            self.bins.append(total_length)

        cnt = 1

        while (total_length <= self.p_length):
            tmp = min(self.p_binsize_lb*np.exp(cnt/self.p_slope), self.p_binsize_ub)
            total_length = total_length + tmp
            self.bins.append(total_length)

            cnt += 1

        self.bins = np.array(self.bins)

        self.computeSupport()

        self.filter_coeffNb = len(self.bins)-1


class Filter_Rect_LinSpaced(Filter_Rect):

    """
    This class defines a temporal filter defined as a linear combination of linearly-spaced rectangular basis functions.
    A filter f(t) is defined in the form

    f(t) = sum_j b_j*rect_j(t),

    where b_j is a set of coefficient and rect_j is a set of linearly spaced rectangular basis functions,
    meaning that the width of all basis functions is the same.
    """

    def __init__(self, length=1000.0, nbBins=30):

        Filter_Rect.__init__(self)

        # Metaparameters

        self.p_length = length         # ms, filter length

        self.filter_coeffNb = nbBins         # integer, define the number of rectangular basis functions being used

        # Initialize

        self.computeBins()                   # using meta parameters self.metaparam_subthreshold define bins and support.

        self.setFilter_toZero()              # initialize filter to 0

    def setMetaParameters(self, length=1000.0, nbBins=10):
        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """

        self.p_length = length

        self.filter_coeffNb = nbBins

        self.computeBins()

        self.setFilter_toZero()

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################

    def computeBins(self):
        """
        This function compute self.bins and self.support given the metaparameters.
        """

        self.bins = np.linspace(0.0, self.p_length, self.filter_coeffNb+1)

        self.computeSupport()

        self.filter_coeffNb = len(self.bins)-1

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):
        """
        Filter spike train spks with the set of rectangular basis functions defining the Filter.
        Since all the basis functions have the same width calculation can be made efficient by filter just ones and shifting.
        """

        T_i = int(T/dt)

        bins_i = Tools.timeToIndex(self.bins, dt)
        spks_i = Tools.timeToIndex(spks, dt)
        nb_bins = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, nb_bins))

        # Filter the spike train with the first rectangular function (for the other simply shift the solution
        tmp = np.zeros(T_i + bins_i[-1] + 1)

        for s in spks_i:
            lb = s + bins_i[0]
            ub = s + bins_i[1]
            tmp[lb:ub] += 1

        tmp = tmp[:T_i]

        # Fill the matrix by shifting the vector tmp
        for l in np.arange(nb_bins):
            tmp_shifted = np.concatenate((np.zeros(int(bins_i[l])), tmp))
            X[:, l] = tmp_shifted[:T_i]

        return X

    def convolution_ContinuousSignal_basisfunctions(self, I, dt):
        """
        Filter continuous signal I with the set of rectangular basis functions defining the Filter.
        Since all the basis functions have the same width calculation can be made efficient by filter just ones and shifting.
        """

        T_i = len(I)

        bins_i = Tools.timeToIndex(self.bins, dt)
        bins_l = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, bins_l))
        I_tmp = np.array(I, dtype='float64')

        window = np.ones(bins_i[1] - bins_i[0])
        window = np.array(window, dtype='float64')

        F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
        F_star_I = np.array(F_star_I[:T_i], dtype='double')

        for l in np.arange(bins_l):

            F_star_I_shifted = np.concatenate((np.zeros(int(bins_i[l])), F_star_I))
            X[:, l] = np.array(F_star_I_shifted[:T_i], dtype='double')

        return X


class Filter_Rect_ArbitrarilySpaced(Filter_Rect):

    """
    This class define a function of time expanded using a set of arbitrarily rectangular basis functions.
    A filter f(t) is defined in the form

    f(t) = sum_j b_j*rect_j(t),

    where b_j is a set of coefficient and rect_j is a set of rectangular basis functions.
    The width and size of each rectangular basis function is free (it is not restricted to, eg, lin spaced).
    """

    def __init__(self, bins=np.array([0.0, 10.0, 50.0, 100.0, 1000.0])):

        Filter_Rect.__init__(self)

        # Initialize

        self.bins = bins

        self.filter_coeffNb = len(bins)-1

        self.computeSupport()

        self.setFilter_toZero()              # initialize filter to 0

    def setBasisFunctions(self, bins):
        """
        Set the parameters defining the rectangular basis functions.
        Attention, each time meta parameters are changes, the value of the filer is reset to 0.
        """

        self.bins = np.array(bins)

        self.computeSupport()

        self.filter_coeffNb = len(bins)-1

        self.setFilter_toZero()

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter_Rect
    ################################################################

    def computeBins(self):
        """
        This filter implementation does not have metaparameters. Filters are direcly set and don't need to be computed.
        """

        pass

    ################################################################
    # IMPLEMENT ABSTRACT METHODS OF Filter
    ################################################################

    def convolution_Spiketrain_basisfunctions(self, spks, T, dt):

        T_i = int(T/dt)

        bins_i = Tools.timeToIndex(self.bins, dt)
        spks_i = Tools.timeToIndex(spks, dt)
        nb_bins = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, nb_bins))

        # Filter the spike train with the first rectangular function (for the other simply shift the solution
        tmp = np.zeros(T_i + bins_i[-1] + 1)

        for s in spks_i:
            lb = s + bins_i[0]
            ub = s + bins_i[1]
            tmp[lb:ub] += 1

        tmp = tmp[:T_i]

        # Fill the matrix by shifting the vector tmp
        for l in np.arange(nb_bins):
            tmp_shifted = np.concatenate((np.zeros(int(bins_i[l])), tmp))
            X[:, l] = tmp_shifted[:T_i]

        return X

    def convolution_ContinuousSignal_basisfunctions(self, I, dt):

        T_i = len(I)

        bins_i = Tools.timeToIndex(self.bins, dt)
        bins_l = self.getNbOfBasisFunctions()

        X = np.zeros((T_i, bins_l))
        I_tmp = np.array(I, dtype='float64')

        window = np.ones(bins_i[1] - bins_i[0])
        window = np.array(window, dtype='float64')

        F_star_I = fftconvolve(window, I_tmp, mode='full')*dt
        F_star_I = np.array(F_star_I[:T_i], dtype='double')

        for l in np.arange(bins_l):

            F_star_I_shifted = np.concatenate((np.zeros(int(bins_i[l])), F_star_I))
            X[:, l] = np.array(F_star_I_shifted[:T_i], dtype='double')

        return X
