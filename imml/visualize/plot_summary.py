# License: BSD-3-Clause

import pandas as pd

from ..explore import get_summary


def plot_summary(Xs: list = None, summary: pd.DataFrame = None, modalities: list = None,
                 title: str = "Summary of the multi-modal dataset",
                 xlabel: str = "Samples", ylabel: str = "Count"):
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
    modalities : list, default=None
        Names of each modality to use when computing the summary from ``Xs``. If ``None``, it will default to the
        modality index.
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
    `Statistics and interaction structure of a multi-modal dataset
    <https://imml.readthedocs.io/stable/auto_tutorials/multi_modal_data_statistics.html#sphx-glr-auto-tutorials-multi-modal-data-statistics-py>`__:
    Tutorial demonstrating its usage on a multi-modal dataset.

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
    if summary is None:
        summary = get_summary(Xs=Xs, modalities=modalities, compute_pct=False, return_df=True)
    if not isinstance(summary, pd.DataFrame):
        raise ValueError(f"Invalid summary. It should be a pd.DataFrame. A {type(summary)} was passed. ")
    summary.index = summary.index.str.replace(" samples", "")
    ax = summary[[c for c in summary.columns if not c.startswith('%')]].plot(
        kind="bar", xlabel=xlabel, ylabel=ylabel, rot=0, title=title)
    return ax