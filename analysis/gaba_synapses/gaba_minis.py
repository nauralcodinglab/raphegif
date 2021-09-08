#%% IMPORT MODULES

from __future__ import division

import os
import pickle

from grr.cell_class import Cell
from grr import MiniDetector


#%% LOAD DATA

DATA_PATH = os.path.join('data', 'raw', '5HT', 'GABA_synapses')
mIPSC_fnames = ['18o23003.abf', '18o23004.abf', '18o23005.abf']

mini_recs = Cell().read_ABF(
    [os.path.join(DATA_PATH, fname) for fname in mIPSC_fnames]
)


#%% DETECT & INSPECT MINIS

mini_detectors = []

for i, rec in enumerate(mini_recs):

    # Load data and name MiniDetector based on filename.
    mini_detector = MiniDetector.MiniDetector(rec, remove_n_sweeps=3)
    mini_detector.set_name(mIPSC_fnames[i][:-4] + '_MiniDetector')

    # Scrape minis.
    mini_detector.compute_gradient(100)
    mini_detector.find_grad_peaks(4.5, 450, width=10)
    mini_detector.extract_minis((-50, 400))

    # Inspect scraped minis.
    mini_detector.plot_signal(10)
    mini_detector.plot_minis(25)

    # Save data.
    mini_detectors.append(mini_detector)


#%% DUMP DATA

MINI_SAVE_PATH = os.path.join(
    'data', 'processed', '5HT', 'GABA_synapses', 'detected_minis'
)

for MinDet in mini_detectors:
    with open(os.path.join(MINI_SAVE_PATH, MinDet.name + '.dat'), 'wb') as f:
        pickle.dump(MinDet, f)
