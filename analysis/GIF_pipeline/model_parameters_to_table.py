import argparse
import pickle

import pandas as pd

from grr.ThresholdModel import modelsToRecords


parser = argparse.ArgumentParser()
parser.add_argument(
    'models', nargs='+', help='Paths to pickled lists of models.'
)
parser.add_argument(
    '-o',
    '--output',
    type=str,
    required=True,
    help='Where to save table of model parameters.',
)
parser.add_argument('-v', '--verbose', action='store_true')

args = parser.parse_args()

# Load models and extract parameters.
records = []
for fname in args.models:
    if args.verbose:
        print('Scraping parameters from models in {}'.format(fname))
    with open(fname, 'rb') as f:
        modList = pickle.load(f)
        f.close()

    recordsFromFile = modelsToRecords(modList)

    # Add field for cell type.
    if '5HT' in fname:
        for rec in recordsFromFile:
            rec['cell_type'] = '5HT'
    elif 'mPFC' in fname:
        for rec in recordsFromFile:
            rec['cell_type'] = 'mPFC'
    elif 'GABA' in fname:
        for rec in recordsFromFile:
            rec['cell_type'] = 'GABA'
    else:
        raise RuntimeError('Unrecognized cell type.')

    records.extend(recordsFromFile)

    del modList, recordsFromFile
if args.verbose:
    print('Done scraping model parameters!')


# Save output as table.
if args.verbose:
    print('Saving scraped parameters to {}'.format(args.output))
paramTable = pd.DataFrame(records)
paramTable.to_csv(args.output, index=False)
if args.verbose:
    print('Finished! Exiting.')
