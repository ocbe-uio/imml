import numpy as np
import pandas as pd

from ..utils import check_Xs_y
from .. import deepmodule_installed, deepmodule_error, Dataset

if deepmodule_installed:
    import torch


class MultiSurvDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `MultiSurv`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    y : array-like of shape (n_samples, 2)
        Survival labels. The first column is event indicator, where 1 means observed event and 0 means censored, and
        the second column is survival time.        .
    transform : list of callable, default=None
        A list of transformations to apply to each modality sample.

    Returns
    -------
    Xs: list of tensors
        A list of modalities for one sample.
    y: tensor of shape (2,)
        Event indicator and survival time for one sample.

    See Also
    --------
    :class:`~imml.survival.MultiSurv`

    Example
    --------
    >>> import numpy as np
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import MultiSurvDataset
    >>> Xs = [torch.rand((20, 10)) for _ in range(3)]
    >>> y = torch.tensor(np.column_stack((np.random.default_rng(42).uniform(0.5, 5, 20),
    ...                                   np.random.default_rng(42).integers(0, 2, 20)))).float()
    >>> train_data = MultiSurvDataset(Xs=Xs, y=y)
    >>> train_dataloader = DataLoader(dataset=train_data)
    >>> next(iter(train_dataloader))
    """

    def __init__(self, Xs: list, y, transform = None):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)
        Xs = check_Xs_y(Xs=Xs, y=y, supervised=True)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError(f"Invalid y. It must have shape (n_samples, 2). Got {tuple(y.shape)}.")

        Xs_ = []
        for X in Xs:
            if isinstance(X, pd.DataFrame):
                X = X.values
            if isinstance(X, np.ndarray):
                X = torch.from_numpy(X).float()
            elif torch.is_tensor(X):
                X = X.float()
            X[X.isnan().all(dim=1)] = 0
            Xs_.append(X)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        if not torch.is_tensor(y):
            raise ValueError(f"Invalid y. It must be array-like. A {type(y)} was passed.")
        y = y.float()
        if y[:, 0].gt(1).any() or y[:, 0].lt(0).any():
            raise ValueError(f"Invalid y. The first column should be the event indicator: values shoud be 0 or 1.")

        if transform is not None:
            if not isinstance(transform, list):
                raise ValueError(f"Invalid transform. It must be a list. A {type(transform)} was passed.")
            if len(transform) != len(Xs_):
                raise ValueError(f"Invalid transform. It must have the same length as Xs. Got {len(transform)} transforms and {len(Xs_)} modalities")
            if not all(callable(func) for func in transform):
                raise ValueError("Invalid transform. All transforms must be callable.")

        self.Xs = Xs_
        self.y = y
        self.transform = transform


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        if self.transform is not None:
            Xs = [transform(X[idx]) for transform, X in zip(self.transform, self.Xs)]
        else:
            Xs = [X[idx] for X in self.Xs]
        output = (Xs, self.y[idx])
        return output
