# License: BSD-3-Clause
from itertools import combinations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..utils import check_Xs_y


def plot_combinations(Xs: list, modalities: list = None, figsize: tuple = None, max_combs: int = 10):
    r"""
    Plot the number of samples per modality combination.

    This function summarizes how many samples are present in each intersection of modalities
    (i.e., samples that are available simultaneously across two or more modalities). The
    resulting figure is similar to an UpSet plot, but it displays the exact counts of
    intersections.

    Parameters
    ----------
    Xs : list of array-like objects, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. Only used when ``summary`` is not provided.
    modalities : list of str, default=None
        Names of each modality (length must match ``len(Xs)``). If ``None``, modality names
        default to indices: ``["0", "1", ...]``.
    figsize : tuple, default=None
        Figure size in inches passed to ``matplotlib.pyplot.subplots``.
    max_combs : int, default=10
        Maximum number of intersections to display. If fewer intersections are available, all will be shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created matplotlib Figure.
    axes : numpy.ndarray of matplotlib.axes.Axes
        2 x 2 array of Axes as described in the layout above.

    See Also
    --------
    `Statistics and interaction structure of a multi-modal dataset
    <https://imml.readthedocs.io/stable/auto_tutorials/multi_modal_data_statistics.html#sphx-glr-auto-tutorials-multi-modal-data-statistics-py>`__:
    Tutorial demonstrating its usage on a multi-modal dataset.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.visualize import plot_combinations
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for _ in range(3)]
    >>> Xs = Amputer(p=0.3, random_state=42).fit_transform(Xs)
    >>> fig, axes = plot_combinations(Xs=Xs, modalities=['RNA', 'Protein', 'Metabolite'], max_combs=8)
    """
    Xs = check_Xs_y(Xs=Xs)
    if (figsize is not None) and (not isinstance(figsize, tuple)):
        raise ValueError(f"Invalid figsize. It must be a tuple. A {type(figsize)} was passed.")
    if not isinstance(max_combs, int):
        raise ValueError(f"Invalid max_combs. It must be a bool. A {type(max_combs)} was passed.")
    if modalities is None:
        modalities = [str(i) for i in range(len(Xs))]
    else:
        if not isinstance(modalities, list):
            raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
        if len(modalities) != len(Xs):
            raise ValueError(f"Invalid modalities. It must have the same number of elements as Xs. Got {len(modalities)} modalities")
        if set(type(mod) for mod in modalities) != set("str"):
            raise ValueError(f"Invalid modalities. List of strings should be passes. Got {[type(mod) for mod in modalities]} modalities")

    if not isinstance(Xs[0], pd.DataFrame):
        Xs = [pd.DataFrame(X) for X in Xs]
    Xs = [X.loc[~X.isna().all(1)] for X in Xs]

    common_indices = {}
    for size in range(2, len(Xs) + 1):
        for combo in combinations(range(len(Xs)), size):
            common_idx = Xs[combo[0]].index
            for idx in combo[1:]:
                common_idx = common_idx.intersection(Xs[idx].index)
            combo = tuple([modalities[i] for i in combo])
            common_indices[combo] = len(common_idx)

    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=figsize, gridspec_kw={'width_ratios': [1, 2]})

    ax = axes[0, 0]
    ax.axis("off")

    combs = pd.Series(common_indices).sort_values(ascending=False)
    ax = axes[0, 1]
    ax = combs.iloc[:max_combs].plot(kind="bar", ax=ax, ylabel="Intersection size", color="black")
    ax.get_xaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container)

    ax = axes[1, 0]
    mod_counts = pd.Series([len(X) for X in Xs], index=modalities).sort_values(ascending=True)
    selected_combs = np.unique(np.array([list(i) for i in combs.iloc[:max_combs].index.values]).flatten())
    mod_counts = mod_counts.loc[[id for id in mod_counts.index if id in selected_combs]]
    ax = mod_counts.plot(kind="barh", ax=ax, xlabel="Set size", color="black")
    ax.invert_xaxis()
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    for container in ax.containers:
        ax.bar_label(container, padding=2)

    ax = axes[1, 1]
    combs = combs.reset_index().drop(columns=0).map(lambda x: modalities[int(x)])
    # combs = combs.loc[mod_counts.index.astype(int)[::-1]].iloc[:max_combs]
    combs = combs.iloc[:max_combs]
    for col in combs.columns:
        ax = combs.reset_index().plot(kind="scatter", ax=ax, x="index", y=col, s=200, ylabel="", c="black")
    ax.get_xaxis().set_visible(False)
    ax.set_xlim(axes[0,1].get_xlim())
    ax.set_ylim(axes[1,0].get_ylim())
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    return fig, axes