import argparse
import json

from grr.Tools import check_dict_fields

generate_models_argparser = argparse.ArgumentParser()
generate_models_argparser.add_argument(
    '--mods', type=str, help='Pickled principal cell models.',
)
generate_models_argparser.add_argument(
    '--sermods', type=str, help='Pickled 5-HT cell models.',
)
generate_models_argparser.add_argument(
    '--gabamods', type=str, help='Pickled GABA neuron models.'
)
generate_models_argparser.add_argument(
    '--prefix',
    type=str,
    required=True,
    help='Path to save GIF_network models.',
)
generate_models_argparser.add_argument(
    '--opts', type=str, required=True, help='Path to opts JSON file.'
)
generate_models_argparser.add_argument(
    '-r',
    '--replicates',
    default=1,
    type=int,
    help='No. of randomized models to generate.',
)
generate_models_argparser.add_argument(
    '--seed', type=int, default=42, help='Random seed (default 42).'
)
generate_models_argparser.add_argument(
    '--overwrite', action='store_true', help='Overwrite existing models.'
)
generate_models_argparser.add_argument(
    '-v',
    '--verbose',
    action='store_true',
    help='Print information about progress.',
)


def load_generate_models_opts(fname, required_fields):
    """Load JSON file with options for generating models.

    Parameters
    ----------
    fname: str
        Path to JSON file to load.
    required_fields: dict
        Expected structure of the loaded JSON. Only field names are validated.

    Returns
    -------
    dict

    """
    with open(fname, 'r') as f:
        opts = json.load(f)
        f.close()
    check_dict_fields(opts, required_fields)
    return opts
