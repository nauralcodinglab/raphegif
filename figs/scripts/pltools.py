#%% IMPORT MODULES

import matplotlib.pyplot as plt
import numpy as np

#%% DEFINE HANDY TOOLS

# Function to make a set of scalebars for a mpl plot
def add_scalebar(x_units = None, y_units = None, anchor = (0.98, 0.02),
text_spacing = (0.02, 0.02), bar_spacing = 0.06, linewidth = 5,
remove_frame = True, omit_x = False, omit_y = False, ax = None):

    """
    Automagically add a set of x and y scalebars to a matplotlib plot

    Inputs:

        x_units: str or None

        y_units: str or None

        anchor: tuple of floats
        --  bottom right of the bbox (in axis coordinates)

        text_spacing: tuple of floats
        --  amount to offset labels from respective scalebars (in axis units)

        bar_spacing: float
        --  amount to separate bars from eachother (in axis units)

        linewidth: numeric
        --  thickness of the scalebars

        remove_frame: bool (default False)
        --  remove the bounding box, axis ticks, etc.

        omit_x/omit_y: bool (default False)
        --  skip drawing the x/y scalebar

        ax: matplotlib.axes object
        --  manually specify the axes object to which the scalebar should be added
    """

    # Basic input processing.
    text_spacer_point = (anchor[0] - text_spacing[0], anchor[1] + text_spacing[1])

    if ax is None:
        ax = plt.gca()

    if x_units is None:
        x_units = ''
    if y_units is None:
        y_units = ''

    # Do y scalebar.
    if not omit_y:
        y_span = ax.get_yticks()[:2]
        y_length = y_span[1] - y_span[0]
        y_span_ax = ax.transLimits.transform(np.array([[0, 0], y_span]).T)[:, 1]
        y_length_ax = y_span_ax[1] - y_span_ax[0]

        # y-scalebar label
        ax.text(
        text_spacer_point[0], anchor[1] + y_length_ax / 2 + bar_spacing,
        '{}{}'.format(y_length, y_units),
        verticalalignment = 'center', horizontalalignment = 'right',
        transform = ax.transAxes
        )

        # y scalebar
        ax.plot(
        [anchor[0], anchor[0]],
        [anchor[1] + bar_spacing, anchor[1] + y_length_ax + bar_spacing],
        'k-', linewidth = linewidth,
        transform = ax.transAxes
        )

    # Do x scalebar.
    if not omit_x:
        x_span = ax.get_xticks()[:2]
        x_length = x_span[1] - x_span[0]
        x_span_ax = ax.transLimits.transform(np.array([x_span, [0, 0]]).T)[:, 0]
        x_length_ax = x_span_ax[1] - x_span_ax[0]

        # x-scalebar label
        ax.text(
        anchor[0] - x_length_ax / 2 - bar_spacing, text_spacer_point[1],
        '{}{}'.format(x_length, x_units),
        verticalalignment = 'bottom', horizontalalignment = 'center',
        transform = ax.transAxes
        )

        # x scalebar
        ax.plot(
        [anchor[0] - x_length_ax - bar_spacing, anchor[0] - bar_spacing],
        [anchor[1], anchor[1]],
        'k-', linewidth = linewidth,
        transform = ax.transAxes
        )

    if remove_frame:
        ax.axis('off')


def hide_ticks(ax = None):

    """
    Delete the x and y ticks of the specified axes. If no axes object is provided, defaults to the current axes.
    """

    if ax is None:
        ax = plt.gca()

    ax.set_xticks([])
    ax.set_yticks([])


def hide_border(sides = 'a', ax = None):

    """
    Sides should be set to `a` for all, or a string containing `r/l/t/b` as needed.
    """

    # Check for correct input
    if not any([letter in sides for letter in 'arltb']):
        raise ValueError('sides should be passed a string with `a` for all sides, or r/l/t/b as-needed for other sides.')

    if ax is None:
        ax = plt.gca()

    if sides == 'a':
        sides = 'rltb'

    sidekeys = {
    'r': 'right',
    'l': 'left',
    't': 'top',
    'b': 'bottom'
    }

    for key, side in sidekeys.iteritems():

        if key not in sides:
            continue
        else:
            ax.spines[side].set_visible(False)


def p_to_string(p):

    """
    Takes a p-value and converts it to a pretty LaTeX string.

    p is presented to three decimal places if p >= 0.05, and as p < 0.05/0.01/0.001 otherwise.
    """

    p_rounded = np.round(p, 3)

    if p_rounded >= 0.05:
        p_str = '$p = {}$'.format(p_rounded)
    elif p_rounded < 0.05 and p_rounded >= 0.01:
        p_str = '$p < 0.05$'
    elif p_rounded < 0.01 and p_rounded >= 0.001:
        p_str = '$p < 0.01$'
    else:
        p_str = '$p < 0.001$'

    return p_str


def subplots_in_grid(shape, loc, ratio = 2, colspan = 1, top_bigger = True, fig = None):

    """
    Generates a set of two vertically-stacked subplots with a given size ratio.

    Useful for generating a set of subplots to show command and recorded voltages from an electrophysiological recording while keeping gridspec grids simple.


    Inputs:

        shape: tuple of ints
        --  Shape of the matplotlib gridspec grid into which to insert the pair of subplots

        loc: tuple of ints
        --  Location within the gridspec grid to insert the paid of subplots

        ratio: int
        --  Size ratio between the top and bottom plot

        colspan: int

        top_bigger: bool (default True)
        --  Make the top plot the bigger of the two plots

        fig
        --  matplotlib figure into which to insert the subplots (set to None to get current figure)

    Returns:

        (top_ax, bottom_ax)


    Example:

    >>> top_ax, bottom_ax = pltools.subplots_in_grid((3, 2), (1, 1))
    >>> top_ax.plot(np.arange(0, 10))
    >>> bottom_ax.plot(np.random.normal(size = 20))
    """

    # Check for correct input.
    if not (int(ratio) == ratio and ratio > 0):
        raise TypeError('ratio must be a positive integer')

    if fig is None:
        fig = plt.gcf()

    new_shape = (shape[0] * (ratio + 1), shape[1])
    new_loc = (loc[0] * (ratio + 1), loc[1])

    if top_bigger:
        top_ax = plt.subplot2grid(new_shape, new_loc, rowspan = ratio, colspan = colspan, fig = fig)
        bottom_ax = plt.subplot2grid(new_shape, (new_loc[0] + ratio, new_loc[1]), colspan = colspan, fig = fig)
    else:
        top_ax = plt.subplot2grid(new_shape, new_loc, colspan = colspan, fig = fig)
        bottom_ax = plt.subplot2grid(new_shape, (new_loc[0] + 1, new_loc[1]), rowspan = ratio, colspan = colspan, fig = fig)

    return top_ax, bottom_ax
