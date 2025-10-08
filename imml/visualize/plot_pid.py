# License: BSD-3-Clause

import math
from matplotlib.patches import Rectangle, Circle
from matplotlib import pyplot as plt

from ..statistics import pid


def plot_pid(rus = None, Xs = None, y = None,
             modalities: list = ["Modality A", "Modality B"], colors: list = ["#780000", "#669BBC", "#FDF0D5"],
             abb: bool = True, figsize : tuple = None, **kwargs):
    r"""
    Plot PID statistics (redundancy, uniqueness and synergy) of a multi-modal dataset as a Venn diagram.

    Parameters
    ----------
    rus : list or dict, default=None
        The output of the `̀`̀pid̀̀̀̀`̀`̀ function.
    Xs : list of array-likes objects, default=None
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities. If rus is provided, it will not be used.
    y : array-like of shape (n_samples,), default=None
        Target vector relative to Xs. If rus is provided, it will not be used.
    modalities : list, default=["Modality A", "Modality B"]
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

    Examples
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
    a_only = round(float(rus.get("Uniqueness1", 0)), 2)
    b_only = round(float(rus.get("Uniqueness2", 0)), 2)
    inter  = round(float(rus.get("Redundancy", 0)), 2)
    outside = round(float(rus.get("Synergy", 0)), 2)
    A = a_only + inter
    B = b_only + inter
    r1 = math.sqrt(A / math.pi) if A>0 else 0.0
    r2 = math.sqrt(B / math.pi) if B>0 else 0.0
    d = _solve_distance_for_overlap(r1, r2, inter)
    max_r = max(r1, r2)

    fig, ax = plt.subplots(figsize=figsize)

    rect = Rectangle((-r1/outside, -max_r/outside), (d+r2)/outside, 2*max_r/outside,
                     facecolor=colors[2], edgecolor="black", alpha=0.5)
    ax.add_patch(rect)

    ax.add_patch(Circle((0, 0), r1, facecolor=colors[0], alpha=0.5, edgecolor="black", linewidth=2))
    ax.add_patch(Circle((d, 0), r2, facecolor=colors[1], alpha=0.5, edgecolor="black", linewidth=2))

    if abb:
        u1, u2, r, s = "U", "U", "R", "S"
    else:
        u1, u2, r, s = "Uniqueness", "Uniqueness", "Redundancy", "Synergy"
    ax.text(-r1/2, 0, f"{u1}\n{a_only}", ha='center', va='center')
    ax.text(d + r2/2, 0, f"{u2}\n{b_only}", ha='center', va='center')
    ax.text(max_r -d/2, 0, f"{r}\n{inter}", ha='center', va='center')
    ax.text(max_r -d/2, max_r*1.2, f"{s} {outside}", ha='center', va='bottom')

    ax.text(0, -(max_r*1.1), modalities[0], ha='center', va='top')
    ax.text(d, -(max_r*1.1), modalities[1], ha='center', va='top')

    padding = max_r * 1.3 + d*0.1
    ax.set_xlim(-padding, d + padding)
    ax.set_ylim(-(max_r*1.6), max_r * 1.6)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
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
