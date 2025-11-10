# License: BSD-3-Clause
import numpy as np
import pandas as pd

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs_y

try:
    import lightning as L
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

Dataset = torch.utils.data.Dataset if deepmodule_installed else object


class M3CareDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `M3Care`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods

        A list of different modalities.
    y : array-like of shape (n_samples,)
        Target vector relative to X.

    Returns
    -------
    Xs: list of array-likes objects
        - Xs length: n_mods

        A list of different modalities for one sample.
    y: array-like of shape (n_samples,)
        Target vector relative to the sample.
    observed_mod_indicator: array-like of shape (1, n_mods)
        Boolean array-like indicating observed modalities for the sample.

    Example
    --------
    >>> from torch.utils.data import DataLoader
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.load import M3CareDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> y = torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float()
    >>> train_data = M3CareDataset(Xs=Xs, y=y)
    >>> train_dataloader = DataLoader(dataset=train_data)
    >>> next(iter(train_dataloader))
    """

    def __init__(self, Xs, y):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)

        if y is None:
            raise ValueError("Invalid y. It cannot be None.")
        if len(y) != len(Xs[0]):
            raise ValueError(f"Invalid y. It must have the same length as each element in Xs. Got {len(y)} vs {len(Xs[0])}")

        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        observed_mod_indicator = get_observed_mod_indicator(Xs)
        if isinstance(observed_mod_indicator, pd.DataFrame):
            observed_mod_indicator = observed_mod_indicator.values
        if isinstance(observed_mod_indicator, np.ndarray):
            observed_mod_indicator = torch.from_numpy(observed_mod_indicator)
        observed_mod_indicator = observed_mod_indicator.bool()

        Xs_ = []
        for X in Xs:
            if isinstance(Xs[0], pd.DataFrame):
                X = X.values
            if isinstance(X, np.ndarray):
                if X[:,0].dtype == object:
                    X = X.tolist()
                    X = [sent if isinstance(sent[0], str) else [""] for sent in X]
                else:
                    X = torch.from_numpy(X).float()
            Xs_.append(X)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        self.Xs = Xs_
        self.y = y
        self.observed_mod_indicator = observed_mod_indicator


    def __len__(self):
        return len(self.observed_mod_indicator)


    def __getitem__(self, idx):
        Xs = [X[idx][0] if isinstance(X[idx][0], str) else X[idx] for X in self.Xs]
        sample = Xs, self.y[idx], self.observed_mod_indicator[idx]
        return sample
