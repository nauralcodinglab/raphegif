import abc
import re
from copy import deepcopy

import numpy as np

from .SpikingModel import SpikingModel
from .Filter import constructMedianFilter
from . import Tools


class ThresholdModel(SpikingModel):

    """
    Abstract class to define a threshold model.
    A threshold model is a model that explicitly describe the membrane potential V and the firing threshold Vt.
    The GIF model is a Threshold model, the GLM model is not.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def simulateVoltageResponse(self, I, dt):
        """
        Simulate the model response to an input current I and return:
        spks : list of spike times (in ms)
        V    : voltage trace (in mV)
        Vt   : voltage threshold trace (in mV)
        """

    def copyParametersFrom(self, other):
        """Copy constructor.

        Parameters listed in `scalarParameters` and `filterParameters` are
        initialized by value to match `other`.

        """
        Tools.assertHasAttributes(self, ['scalarParameters', 'filterParameters'])
        Tools.assertHasAttributes(other, self.scalarParameters)
        Tools.assertHasAttributes(other, self.filterParameters)

        for scalarParam in self.scalarParameters:
            setattr(self, scalarParam, deepcopy(getattr(other, scalarParam)))

        for filterParam in self.filterParameters:
            setattr(self, filterParam, deepcopy(getattr(other, filterParam)))

    def computeRateAndThreshold_vs_I(self, mu, sigma, tau, dt, T, ROI, nbRep=10):
        """
        Compute standard FI curve using multiple OU processes with means mu (list), standard deviations sigma (list)
        temporal correlation tau (in ms) and of duration T.
        Use ROI to define the region of interest over which to compute the average firing rate. ROI can be usful when
        one wants to evaluate the transient or the steady state FI.
        This function also compute the theta-I curve, that the average value of the firing threshold as a function of the
        input statistics.
        """

        self.setDt(dt)

        FI_all = np.zeros((len(sigma), len(mu), nbRep))
        thetaI_all = np.zeros((len(sigma), len(mu), nbRep))
        thetaI_VT_all = np.zeros((len(sigma), len(mu), nbRep))

        s_cnt = -1
        for s in sigma:
            s_cnt += 1

            m_cnt = -1
            for m in mu:

                m_cnt += 1

                for r in np.arange(nbRep):

                    I_tmp = Tools.generateOUprocess(T=T, tau=tau, mu=m, sigma=s, dt=dt)

                    (spks_t, V, V_T) = self.simulateVoltageResponse(I_tmp, dt)
                    spks_i = Tools.timeToIndex(spks_t, dt)

                    spiks_i_sel = np.where(((spks_t > ROI[0]) & (spks_t < ROI[1])))[0]
                    spiks_i_sel = spks_i[spiks_i_sel]

                    rate = 1000.0*len(spiks_i_sel)/(ROI[1]-ROI[0])
                    FI_all[s_cnt, m_cnt, r] = rate

                    theta = np.mean(V[spiks_i_sel])
                    thetaI_all[s_cnt, m_cnt, r] = theta

                    theta_VT = np.mean(V_T[spiks_i_sel])
                    thetaI_VT_all[s_cnt, m_cnt, r] = theta_VT

        return (FI_all, thetaI_all, thetaI_VT_all)


def constructMedianModel(modelType, models, nan_behavior='default'):
    # INPUT CHECKS
    if nan_behavior not in ('default', 'ignore'):
        raise ValueError(
            'Expected `default` or `ignore` for argument `nan_behavior`; '
            'got {} instead.'.format(
                nan_behavior
            )
        )

    # Ensure all models are of type modelType.
    for mod in models:
        if not isinstance(mod, modelType):
            raise TypeError(
                'All models must be of type {}; got instance of '
                'type {}'.format(modelType, type(mod))
            )

    # CONSTRUCT MEDIANMODEL
    medianModel = modelType()

    # Set values of scalar parameters.
    for paramName in modelType.scalarParameters:
        paramValues = []
        for mod in models:
            paramValues.append(getattr(mod, paramName, np.nan))
        if nan_behavior == 'default':
            setattr(medianModel, paramName, np.median(paramValues))
        elif nan_behavior == 'ignore':
            setattr(medianModel, paramName, np.nanmedian(paramValues))
        else:
            raise RuntimeError('Unexpectedly reached end of switch.')

    # Set values of filter parameters.
    for filtName in modelType.filterParameters:
        filtInstances = []
        for mod in models:
            filtInstances.append(getattr(mod, filtName, None))
        setattr(
            medianModel,
            filtName,
            constructMedianFilter(type(filtInstances[0]), filtInstances),
        )

    return medianModel


def modelsToRecords(models):
    """Get a list of dicts of model parameters from list of models.

    Vectorized wrapper for ThresholdModel.modelToRecord; see its documentation
    for details.

    """
    records = [modelToRecord(mod) for mod in models]
    return records


def modelToRecord(model):
    """Get a dict of model parameters from model.

    #TODO: implement for filter types other than Exponentials.

    Arguments
    ---------
    model : object with `scalarParameters` and `filterParameters` attributes

    Returns
    -------
    Dict of model parameters. For scalarParameters, each dict key is the name
    of the parameter. For filterParameters, the dict keys are strings of the
    form `<filterName>_<tau>` where tau is the timescale of the coefficient.

    """
    Tools.assertHasAttributes(model, ['scalarParameters', 'filterParameters'])

    # Extract model name and type.
    modelIdentifiers = {
        'name': getattr(model, 'name', ''),
        'type': re.search(r'(\w+)\'', str(type(model))).groups()[-1]
    }

    # Extract scalar parameters.
    scalarParams = {
        paramname: getattr(model, paramname, None)
        for paramname in model.scalarParameters
    }

    try:
        scalarParams['resting_potential'] = getRestingMembranePotential(
            model, 1e-6
        )
    except:
        scalarParams['resting_potential'] = np.nan

    # Extract filter coefficients.
    filterParams = {}
    for filterName in model.filterParameters:
        Tools.assertHasAttributes(getattr(model, filterName), ['taus'])
        for i, tau in enumerate(getattr(model, filterName).taus):
            filterParams['{}_{:.1f}'.format(filterName, tau)] = getattr(
                model, filterName
            ).getCoefficients()[i]

    # Merge scalar parameters and filter coeffs into single record.
    record = {}
    record.update(modelIdentifiers)
    record.update(scalarParams)
    record.update(filterParams)

    return record


def getRestingMembranePotential(model, tol=1e-3):
    """Estimate resting membrane potential numerically.

    Arguments
    ---------
    model: GIF or subclass
        Model for which to estimate the resting membrane potential.
    tol: float
        Numerical tolerance for the resting membrane potential estimate. Try
        setting this to a larger value of algorithm fails to converge.

    Returns
    -------
    Numerically-estimated resting membrane potential in mV.

    """
    # Implementation note:
    # Model capacitance changes the integration time constant, but doesn't
    # affect reversal. We can therefore change capacitance to make the
    # numerical estimate of resting potential converge faster. The procedure is
    # therefore to temporarily change model capacitance inside of a `try`
    # block, let the model run with no input to get resting potential, then
    # put the original capacitance back.
    originalCapacitance = model.C

    try:
        model.C = min(
            model.gl, model.C
        )  # Alter C so that time constant becomes very short.
        effectiveTau = model.C / model.gl  # Effective tau in ms.
        timesteps = int(effectiveTau * 5.0 / model.dt)
        inputCurrent = np.zeros(timesteps)

        last_dV = (
            tol + 1
        )  # Initialize with nonsense value to make sure "do-while" loop runs at least once.
        last_V = model.El
        while np.abs(last_dV) > tol:
            t, V, _ = model.simulateDeterministic_forceSpikes(
                inputCurrent, last_V, []
            )
            last_V = V[-1]
            last_dV = V[-1] - V[-2]

    finally:
        model.C = originalCapacitance

    return last_V
