import numpy as np
import pandas as pd

from .. import Tensor


def as_pd_Xs(Xs):
    if isinstance(Xs[0], Tensor):
        Xs = [X.detach().cpu().numpy() for X in Xs]
    if not isinstance(Xs[0], pd.DataFrame):
        Xs = [pd.DataFrame(X) for X in Xs]
    return Xs


def as_numpy_Xs(Xs):
    if isinstance(Xs[0], pd.DataFrame):
        Xs = [X.values for X in Xs]
    elif isinstance(Xs[0], Tensor):
        Xs = [X.detach().cpu().numpy() for X in Xs]
    return Xs