# License: BSD-3-Clause

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..impute import get_observed_mod_indicator


def plot_missing_modality(Xs, ax: matplotlib.axes.Axes = None, figsize: tuple = None, sort: bool = True):
    r"""
    Plot modality missing. Missing modalities appear as white, while black indicates available modalities.

    Parameters
    ----------
    Xs : list of array-likes objects, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. If rus is provided, it will not be used.
    ax : matplotlib.axes.Axes, default=None
        Axes where to draw the figure.
    figsize : tuple, default=None
        Figure size (tuple) in inches.
    sort : bool, default=True
        If True, samples will be sort based on their available modalities.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object.
    ax : `matplotlib.axes.Axes`
        Axes object.
    """
    if not isinstance(Xs, list):
        raise ValueError(f"Invalid Xs. It must be a list. A {type(Xs)} was passed.")
    if any(len(X) == 0 for X in Xs):
        raise ValueError("Invalid Xs. All elements must have at least one sample.")
    if len(set(len(X) for X in Xs)) > 1:
        raise ValueError("Invalid Xs. All elements must have the same number of samples.")
    if (ax is not None) and (not isinstance(ax, matplotlib.axes.Axes)):
        raise ValueError(f"Invalid ax. It must be a matplotlib.axes.Axes. A {type(ax)} was passed.")
    if (figsize is not None) and (not isinstance(figsize, tuple)):
        raise ValueError(f"Invalid figsize. It must be a tuple. A {type(figsize)} was passed.")
    if not isinstance(sort, bool):
        raise ValueError(f"Invalid sort. It must be a bool. A {type(sort)} was passed.")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = None
    xlabel, ylabel = "Modality", "Samples"
    observed_view_indicator = get_observed_mod_indicator(Xs)
    observed_view_indicator = pd.DataFrame(observed_view_indicator)
    if sort:
        observed_view_indicator = observed_view_indicator.sort_values(list(range(len(Xs))))
    observed_view_indicator.columns = observed_view_indicator.columns + 1
    ax.pcolor(observed_view_indicator, cmap="binary", edgecolors="black", vmin=0., vmax=2.)
    ax.set_xticks(np.arange(0.5, len(observed_view_indicator.columns), 1), observed_view_indicator.columns)
    _ = ax.set_xlabel(xlabel), ax.set_ylabel(ylabel)
    return fig, ax
