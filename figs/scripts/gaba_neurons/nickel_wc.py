#%% IMPORT MODULES

from __future__ import division

import os
import shutil

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import pandas as pd

import sys
sys.path.append('./analysis/spk_timing/')

import OhmicSpkPredictor as osp
from grr.cell_class import Cell
import src.pltools as pltools


#%% LOAD DATA

DATA_PATH = './data/raw/GABA/nickel/'

fnames = {
    'bl_steps': ['18n16006.abf', '18n16007.abf', '18n16008.abf'],
    'ni_washin': ['18n16009.abf'],
    'ni_steps': ['18n16010.abf'],
    'wash': ['18n16011.abf']
}
recs = {}

for key in fnames.keys():
    recs[key] = Cell().read_ABF([DATA_PATH + f for f in fnames[key]])

#%%

recs['bl_steps'][0].plot()
recs['ni_washin'][0].plot()
