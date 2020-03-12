import seaborn as sns

colors = {
    'ser': sns.color_palette()[0],
    'som': sns.color_palette()[1],
    'pyr': 'gray',
    'fit': 'black',
    'm': sns.color_palette()[2],
    'h': sns.color_palette()[3],
    'n': sns.color_palette()[4],
    '4AP': sns.color_palette()[5],
    'input': 'gray',
}

pvalue_thresholds = [[1e-4, '****'], [1e-3, '***'], [1e-2, '**'], [5e-2, '*'], [0.1, 'o'], [1., 'ns']]

sbarlw = 1.5  # Scale bar line width
