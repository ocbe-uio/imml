import numpy as np
import pandas as pd

from imml import deepmodule_installed

if deepmodule_installed:
    import torch


def make_sample_data(n_samples=40, n_features=10, splits=(3, 5), random_state=42):
    X = np.random.default_rng(random_state).random((n_samples, n_features))
    X = pd.DataFrame(X)
    split_points = (0, *splits, n_features)
    Xs_pandas = [X.iloc[:, start:end] for start, end in zip(split_points[:-1], split_points[1:])]
    Xs_numpy = [X.to_numpy() for X in Xs_pandas]
    sample_data = [Xs_pandas, Xs_numpy]
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        sample_data.append(Xs_torch)
    return sample_data
