# License: BSD-3-Clause

import math
from matplotlib.patches import Rectangle, Circle
from matplotlib import pyplot as plt

from ..statistics import pid


def plot_pid(rus = None, Xs = None, y = None,
             mod_names: list = ["Modality A", "Modality B"], colors: list = ["#780000", "#669BBC", "#FDF0D5"],
             abb: bool = True, figsize : tuple = None, **kwargs):
    r"""
    Plot PID statistics (redundancy, uniqueness and synergy) of a multi-modal dataset as a Venn diagram.

    Parameters
    ----------
    rus : list or dict, default=None
        The output of the ``pid`` function.
    Xs : list of array-likes objects, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mod_names. If rus is provided, it will not be used.
    y : array-like of shape (n_samples,), default=None
        Target vector relative to Xs. If rus is provided, it will not be used.
    mod_names : list, default=["Modality A", "Modality B"]
        Name of each modality.
    colors : list, default=["#780000", "#669BBC", "#FDF0D5"]
        Colors used for the regions.
    abb : bool, default=True
        Whether to use abbreviations (S, U1, U2 and R) for "Synergy", "Uniquesness1", "Uniqueness2" and "Redundancy",
        respectively.
    figsize : tuple, default=None
        Figure size (tuple) in inches.
    **kwargs : dict, default=None
        Additional keyword arguments are passed to the ``pid`` function.

    Returns
    -------
    fig : `matplotlib.figure.Figure`
        Figure object.
    ax : `matplotlib.axes.Axes`
        Axes object.

    See Also
    --------
    `Statistics and interaction structure of a multi-modal dataset
    <https://imml.readthedocs.io/stable/auto_tutorials/multil_modal_data_statistics.html#sphx-glr-auto-tutorials-multil-modal-data-statistics-py>`__:
    Tutorial demonstrating its usage on a multi-modal dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> y = pd.Series(np.random.default_rng(42).uniform(low=0, high=2, size=len(Xs[0])))
    >>> plot_pid(Xs = Xs, y=y, **{"random_state":42})
    """
    if Xs is not None:
        rus = pid(Xs=Xs, y=y, **kwargs)
    if any(key not in rus.keys() for key in ["Redundancy", "Uniqueness1", "Uniqueness2", "Synergy"]) or (len(rus) != 4):
        raise ValueError(f"Invalid rus. It should have the keys 'Redundancy', 'Uniqueness1', 'Uniqueness2' and 'Synergy'."
                         f" {rus} was provided.")
    a_only = float(rus.get("Uniqueness1", 0))
    b_only = float(rus.get("Uniqueness2", 0))
    inter  = float(rus.get("Redundancy", 0))
    outside = float(rus.get("Synergy", 0))
    A = a_only + inter
    B = b_only + inter
    r1 = math.sqrt(A / math.pi) if A>0 else 0.0
    r2 = math.sqrt(B / math.pi) if B>0 else 0.0
    d = _solve_distance_for_overlap(r1, r2, inter)
    max_r = max(r1, r2)
    k = 1.15

    fig, ax = plt.subplots(figsize=figsize)

    x_min = -r1*(k+outside)
    w = (r1 + r2 + d)*(k+outside)
    y_min = -max_r*(k+outside)
    h = (2*max_r)*(k+outside)
    rect = Rectangle((x_min, y_min), w, h,
                     facecolor=colors[2], edgecolor="black", alpha=0.5)
    ax.add_patch(rect)

    ax.add_patch(Circle((0, 0), r1, facecolor=colors[0], alpha=0.5, edgecolor="black", linewidth=2))
    ax.add_patch(Circle((d, 0), r2, facecolor=colors[1], alpha=0.5, edgecolor="black", linewidth=2))

    if abb:
        u1, u2, r, s = "U1", "U2", "R", "S"
    else:
        u1, u2, r, s = "Uniqueness1", "Uniqueness2", "Redundancy", "Synergy"
    if a_only <= 0.03:
        x_pos_label1 = (d-r1)
    else:
        x_pos_label1 = (d-r1-r2)/2
    if b_only <= 0.03:
        x_pos_label2 = (r1+r2-d)/2
    else:
        x_pos_label2 = (d+r2+r1)/2
    ax.text(x_pos_label1, 0, f"{u1}\n{round(a_only, 2)}", ha='center', va='center')
    ax.text(x_pos_label2, 0, f"{u2}\n{round(b_only, 2)}", ha='center', va='center')
    ax.text(-r1 + d + r2, 0, f"{r}\n{round(inter, 2)}", ha='center', va='center')
    ax.text(-r1 + d + r2, (h + y_min + max_r)/2, f"{s} {round(outside, 2)}",
            ha='center', va='center')

    ax.text(-r1, (-max_r+y_min)/2, mod_names[0], ha='left', va='bottom')
    ax.text(d+r2, (-max_r+y_min)/2, mod_names[1], ha='right', va='bottom')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.autoscale()
    return fig, ax


def _overlap_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return math.pi * min(r1, r2)**2
    r1_2, r2_2 = r1*r1, r2*r2
    alpha = math.acos((d*d + r1_2 - r2_2) / (2*d*r1))
    beta  = math.acos((d*d + r2_2 - r1_2) / (2*d*r2))
    return r1_2*alpha + r2_2*beta - d*r1*math.sin(alpha)


def _solve_distance_for_overlap(r1, r2, target_overlap, tol=1e-6, max_iter=100):
    lo = max(0.0, abs(r1 - r2))
    hi = r1 + r2
    max_overlap = math.pi * min(r1, r2)**2
    target_overlap = max(0.0, min(target_overlap, max_overlap))
    if target_overlap <= 0:
        return hi
    if abs(target_overlap - max_overlap) < tol:
        return lo
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        ov = _overlap_area(r1, r2, mid)
        if abs(ov - target_overlap) < tol:
            return mid
        if ov > target_overlap:
            lo = mid
        else:
            hi = mid
    return 0.5*(lo+hi)
