import seaborn as sns

colors = {
    'ser': sns.color_palette()[0],
    'som': sns.color_palette()[1],
    'pyr': 'gray',
    'fit': 'black',
    'm': sns.color_palette()[2],
    'h': sns.color_palette()[3],
    'n': sns.color_palette()[4],
    '4AP': (207./255., 2./255., 52./255.),  # Cherry red
    'input': 'gray',
    'gif': (146./255., 10./255., 78./255.),  # Mulberry
    'agif': (79./255., 145./255., 83./255.),  # Light forest
    'igif': 'gray'
}

pvalue_thresholds = [
    [1e-4, '****'],
    [1e-3, '***'],
    [1e-2, '**'],
    [5e-2, '*'],
    [0.1, 'o'],
    [1.0, 'ns'],
]

sbarlw = 1.5  # Scale bar line width
insetlw = 0.75  # Line width for dashed border used to mark insets.

fliersize=2
