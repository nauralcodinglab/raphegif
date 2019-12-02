"""Reduce the size of Experiment datasets.

Can be used to evaluate the effect of dataset size on model performance.

"""

import abc
from copy import deepcopy
import warnings

import numpy as np

from .Tools import raiseExpectedGot, timeToIndex


class Minifier(object):
    """Abstract class for objects that subsample Experiments."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def minify(self, obj_):
        """Return a copy of obj_ with less data."""


class ExperimentMinifier(Minifier):

    def __init__(self, trainingsetMinifier, testsetMinifier):
        self.trainingsetMinifier = trainingsetMinifier
        self.testsetMinifier = testsetMinifier

    def minify(self, experiment):
        minifiedExpt = deepcopy(experiment)
        minifiedExpt.trainingset_traces = self.trainingsetMinifier.minify(experiment.trainingset_traces)
        minifiedExpt.testset_traces = self.testsetMinifier.minify(experiment.testset_traces)

        return minifiedExpt


class TraceMinifier(Minifier):
    """Abstract class for Minifier objects acting on Traces."""

    def _trimROI(self, trace, startTime, stopTime):
        useTrace = trace.useTrace
        trace.setROI([[startTime, stopTime]])
        trace.useTrace = useTrace  # Calling `setROI()` sets useTrace=True.


class Dummy_TraceMinifier(TraceMinifier):
    """Implement TraceMinifier interface but don't do anything."""

    def __init__(self):
        pass

    def minify(self, traces):
        minifiedTraces = deepcopy(traces)
        return minifiedTraces


class SpikeCount_TraceMinifier(TraceMinifier):
    """Trim traces based on spike count."""

    def __init__(self, numberOfSpikes, padding=20., mode='valid'):
        """Initialize SpikeCount_TraceMinifier.

        Arguments
        ---------
        numberOfSpikes: int > 0
            Number of spikes to include in each minified trace.
        padding: float >= 0, optional
            Length of analogsignals to include before/after the first/last
            spikes (ms).
        mode: `valid` (default) or `same`, optional
            Method for determining how sets of traces should be minified.
            `valid` trims each trace to a valid interval independently. `same`
            trims all traces to the same interval, determined based on a
            randomly-selected trace.

        """
        if numberOfSpikes < 1 or int(numberOfSpikes) != numberOfSpikes:
            raiseExpectedGot(
                'positive integer', '`numberOfSpikes`', numberOfSpikes
            )
        else:
            self.numberOfSpikes = numberOfSpikes

        if padding < 0.:
            raiseExpectedGot('value >= 0.', '`padding`', padding)
        else:
            self.padding = padding

        if mode not in {'valid', 'same'}:
            raiseExpectedGot('`valid` or `same`', '`mode`', mode)
        else:
            self.mode = mode

    def minify(self, traces):
        minifiedTraces = deepcopy(traces)

        # Trim traces inplace according to defined methods.
        if self.mode == 'valid':
            self._validMultiMinify(minifiedTraces)
        elif self.mode == 'same':
            self._sameMultiMinify(minifiedTraces)
        else:
            # Cannot get here.
            raise RuntimeError('Unexpectedly reached end of switch.')

        # Re-run spike detection inplace.
        self._redetectSpikes(minifiedTraces)

        return minifiedTraces

    def _validMultiMinify(self, tracesCopy):
        """Trim all traces in a list independently."""
        for tr in tracesCopy:
            startIndex, stopIndex = self._getSegmentLimits(tracesCopy)
            startTime = startIndex * tr.dt
            stopTime = stopIndex * tr.dt
            self._trimROI(tr, startTime, stopTime)

    def _sameMultiMinify(self, tracesCopy):
        """Trim all traces in a list to the same interval.

        Interval to trim traces to is determined based on a randomly selected
        basis trace.

        """
        basisTrace = np.random.choice(tracesCopy)
        startIndex, stopIndex = self._getSegmentLimits(basisTrace)
        startTime = startIndex * basisTrace.dt
        stopTime = stopIndex * basisTrace.dt

        spikesPerTrace = []
        for tr in tracesCopy:
            self._trimROI(tr, startTime, stopTime)
            spikesPerTrace.append(tr.getSpikeNbInROI())

        if any(~np.isclose(spikesPerTrace, self.numberOfSpikes, rtol=0.1, atol=0.)):
            warnings.warn(
                'Not all traces are within 10\% of {} spike target; '
                '{} spikes in each trace.'.format(
                    self.numberOfSpikes, spikesPerTrace
                )
            )

    def _getSegmentLimits(self, trace):
        firstSpikeIndCandidates = np.random.permutation(
            trace.getSpikeIndicesInROI()[:-self.numberOfSpikes]
        )

        # Find a spike index followed by `numberOfSpikes` spikes within the
        # trace ROI.
        foundValidSegment = False
        for firstSpikeIndCandidate in firstSpikeIndCandidates:
            if self._enoughSpikesFromIndAreInROI(trace, firstSpikeIndCandidate):
                startIndex = firstSpikeIndCandidate - timeToIndex(self.padding, trace.dt)[0]
                stopIndex = trace.getSpikeIndicesInROI()[
                    np.argwhere(trace.getSpikeIndicesInROI() == firstSpikeIndCandidate)[0]
                    + self.numberOfSpikes
                ] + timeToIndex(self.padding, trace.dt)[0]
                foundValidSegment = True
                break
            else:
                continue

        # If there aren't enough spikes within the trace ROI to get a
        # contiguous segment an error should be raised.
        if foundValidSegment:
            return (startIndex, stopIndex)
        else:
            raise InsufficientSpikesError(
                'Could not find a contiguous trace segment with {} '
                'spikes.'.format(self.numberOfSpikes)
            )

    def _enoughSpikesFromIndAreInROI(self, trace, ind):
        lastSpikeIndex_spikesInROI = trace.getSpikeIndicesInROI()[
            np.argwhere(trace.getSpikeIndicesInROI() == ind)
            + self.numberOfSpikes
        ]
        lastSpikeIndex_allSpikes = trace.getSpikeIndices()[
            np.argwhere(trace.getSpikeIndices() == ind) + self.numberOfSpikes
        ]
        if lastSpikeIndex_spikesInROI == lastSpikeIndex_allSpikes:
            return True
        else:
            return False

    def _redetectSpikes(self, traces):
        """Re-detect spikes using `Trace.detectSpikes()` defaults."""
        for tr in traces:
            tr.detectSpikes()


class InsufficientSpikesError(Exception):
    pass
