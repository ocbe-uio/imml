import pandas as pd

from .. import Tensor


def as_pd_Xs(Xs):
    if isinstance(Xs[0], Tensor):
        Xs = [X.detach().cpu().numpy() for X in Xs]
    if not isinstance(Xs[0], pd.DataFrame):
        Xs = [pd.DataFrame(X) for X in Xs]
    return Xs


def as_np_Xs(Xs):
    if isinstance(Xs[0], pd.DataFrame):
        Xs = [X.values for X in Xs]
    elif isinstance(Xs[0], Tensor):
        Xs = [X.detach().cpu().numpy() for X in Xs]
    return Xs


def as_pd_indicator(indicator):
    if isinstance(indicator, Tensor):
        indicator = indicator.detach().cpu().numpy()
    if not isinstance(indicator, pd.DataFrame):
        indicator = pd.DataFrame(indicator)
    return indicator
