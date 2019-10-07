#%% IMPORT MODULES

from __future__ import division

import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
import pandas as pd

from grr.Experiment import Experiment
from grr.AEC import AEC_Badel
from grr.GIF import GIF
from grr.AugmentedGIF import AugmentedGIF
from src.Filter_Rect_LogSpaced import Filter_Rect_LogSpaced
from grr.Filter_Exps import Filter_Exps

import src.pltools as pltools
from grr.Tools import gagProcess


#%% LOAD DATA

DATA_PATH = './data/raw/5HT/fast_noise/'

with open(DATA_PATH + './gamma_filter_tests_output.pyc', 'rb') as f:
    test_out = pickle.load(f)

with open(DATA_PATH + './5HT_gamma_rect.pyc', 'rb') as f:
    rect_gamma = pickle.load(f)


#%% EXTRACT MD FROM RECTANGULAR GAMMA

precision = 8.
Md_vals_rect = {'GIF': [], 'KGIF': []}

for i, GIF_ls in enumerate([rect_gamma['GIFs'], rect_gamma['AugmentedGIFs']]):

    for expt, GIF_ in zip(rect_gamma['experiments'], GIF_ls):

        if not np.isnan(GIF_.Vt_star):

            with gagProcess():

                # Use the myGIF model to predict the spiking data of the test data set in myExp
                tmp_prediction = expt.predictSpikes(GIF_, nb_rep=500)

                # Compute Md* with a temporal precision of +/- 4ms
                Md = tmp_prediction.computeMD_Kistler(precision, 0.1)

        else:

            tmp_prediction = np.nan
            Md = np.nan

        if i == 0:
            tmp_label = 'GIF'
        elif i == 1:
            tmp_label = 'KGIF'
        Md_vals_rect[tmp_label].append(Md)

        print '{} {} MD* {}ms: {:.2f}'.format(expt.name, tmp_label, precision, Md)

Md_vals_rect = pd.DataFrame(Md_vals_rect)
Md_vals_rect = Md_vals_rect.melt(var_name = 'Model', value_name = 'Md*')
Md_vals_rect['gamma'] = 'Rect. basis'
Md_vals_rect

#%% CLEAN DATA FROM EXP GAMMA TESTS

def scrape_md(md_nested_list, models, gamma_label):

    if not len(models) == len(md_nested_list):
        raise ValueError('Number of model labels must match length of md_nested_list.')

    df = pd.DataFrame(md_nested_list, index = models).T
    df = df.melt(var_name = 'Model', value_name = 'Md*')
    df['gamma'] = gamma_label

    return df

md_df = Md_vals_rect

for i, tst_ in enumerate(test_out):

    md_df = md_df.append(
        scrape_md(tst_['Md'], ['GIF', 'KGIF'], str(tst_['timescales'])[1:-1]),
        ignore_index = True
    )

md_df = md_df[['gamma', 'Model', 'Md*']]


#%% CREATE FIGURE

IMG_PATH = './figs/ims/'

kgif_rect_md = Md_vals_rect['Md*'][Md_vals_rect['Model'] == 'KGIF'].tolist()

plt.figure(figsize = (8, 4))
plt.axhspan(
    np.percentile(kgif_rect_md, 25), np.percentile(kgif_rect_md, 75),
    facecolor = 'gray', edgecolor = 'none', alpha = 0.3, zorder = -2
)
plt.axhline(np.percentile(kgif_rect_md, 50), color = 'gray', zorder = -1, lw = 0.9)
sns.boxplot(x = 'gamma', y = 'Md*', hue = 'Model', data = md_df, width = 0.6, dodge = True)
plt.ylim(0, 1)
plt.xticks(rotation = 45)
plt.ylabel('$M_d^*$ (8ms precision)')
plt.xlabel(r'Exponential spike-triggered threshold movement timescales')
plt.legend(loc = (0.1, 0.75))
plt.tight_layout()

if IMG_PATH is not None:
    plt.savefig(IMG_PATH + 'gamma_regularization.png', dpi = 300)

plt.show()
