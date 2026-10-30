import pandas as pd

from .. import Tensor, deepmodule_installed

if deepmodule_installed:
    import torch


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


def as_pd_X(X):
    if isinstance(X, Tensor):
        X = X.detach().cpu().numpy()
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    return X


def as_tensor_Xs(Xs):
    if isinstance(Xs[0], pd.DataFrame):
        Xs = [X.values for X in Xs]
    if not isinstance(Xs[0], Tensor):
        Xs = [torch.as_tensor(X) for X in Xs]
    return Xs


def as_tensor_X(X):
    if isinstance(X, pd.DataFrame):
        X = X.values
    if not isinstance(X, Tensor):
        X = torch.as_tensor(X)
    return X
