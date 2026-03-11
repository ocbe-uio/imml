# License: BSD-3-Clause

import pandas as pd

from ..explore import get_summary


def plot_summary(Xs: list = None, summary: pd.DataFrame = None, mod_names: list = None,
                 figsize: tuple = None, title: str = None,
                 xlabel: str = None, ylabel: str = "Count"):
    r"""
    Plot a bar chart summarizing completeness across modalities in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-like objects, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. Only used when ``summary`` is not provided.
    summary : pd.DataFrame, default=None
        A summary dataframe as returned by ``imml.explore.get_summary``. If provided, it will be plotted directly.
        If None, the summary will be computed from ``Xs``.
    mod_names : list, default=None
        Names of each modality to use when computing the summary from ``Xs``. If ``None``, it will default to the
        modality index.
    figsize : tuple, default=None
        Figure size in inches passed to ``pd.DataFrame.plot``.
    title : str, default="Summary of the multi-modal dataset"
        Title of the plot.
    xlabel : str, default="Samples"
        Label for the x-axis.
    ylabel : str, default="Count"
        Label for the y-axis.

    Returns
    -------
    matplotlib.axes.Axes
        The matplotlib Axes containing the bar plot.

    See Also
    --------
    :class:`~imml.explore.get_summary`
    :class:`~imml.visualize.plot_missing_modality`
    :class:`~imml.visualize.plot_combinations`

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.visualize import plot_summary
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.3, random_state=42).fit_transform(Xs)
    >>> plot_summary(Xs = Xs)
    """
    if (figsize is not None) and (not isinstance(figsize, tuple)):
        raise ValueError(f"Invalid figsize. It must be a tuple. A {type(figsize)} was passed.")
    if summary is None:
        summary = get_summary(Xs=Xs, mod_names=mod_names, compute_pct=False, return_df=True)
    if not isinstance(summary, pd.DataFrame):
        raise ValueError(f"Invalid summary. It should be a pd.DataFrame. A {type(summary)} was passed. ")
    summary.index = summary.index.str.replace(" samples", "")
    ax = summary[[c for c in summary.columns if not c.startswith('%')]].plot(
        kind="bar", xlabel=xlabel, ylabel=ylabel, rot=0, title=title, figsize=figsize)
    return ax