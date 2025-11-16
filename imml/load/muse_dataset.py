# License: BSD-3-Clause
import numpy as np
import pandas as pd

from ..impute import get_missing_mod_indicator
from ..utils import check_Xs_y

try:
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

Dataset = torch.utils.data.Dataset if deepmodule_installed else object


class MUSEDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `MUSE`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods

        A list of different modalities.
    y : array-like of shape (n_samples,)
        Target vector relative to X.

    Returns
    -------
    Xs_idx: list of array-likes objects
        - Xs length: n_mods

        A list of different modalities for one sample.
    y_idx: array-like of shape (n_samples,)
        Target vector relative to the sample.
    missing_mod_indicator: array-like of shape (1, n_mods)
        Boolean array-like indicating missing modalities for the sample.
    y_indicator: array-like of shape (1,)
        Boolean array-like indicating observed label for the sample.

    See Also
    --------
    `Classify an incomplete tabular–language dataset
    <https://imml.readthedocs.io/stable/auto_tutorials/classify_incomplete_tabular_language.html>`__:
    Tutorial demonstrating its usage on an incomplete tabular-language dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import torch
    >>> from torch.utils.data import DataLoader
    >>> from imml.load import MUSEDataset
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> y = torch.from_numpy(np.random.default_rng(42).integers(0, 2, len(Xs[0]))).float()
    >>> train_data = MUSEDataset(Xs=Xs, y=y)
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
        missing_mod_indicator = get_missing_mod_indicator(Xs)
        if isinstance(missing_mod_indicator, pd.DataFrame):
            missing_mod_indicator = missing_mod_indicator.values
        if isinstance(missing_mod_indicator, np.ndarray):
            missing_mod_indicator = torch.from_numpy(missing_mod_indicator)
        missing_mod_indicator = missing_mod_indicator.bool()

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
        y_indicator = torch.logical_not(torch.isnan(y))

        self.Xs = Xs_
        self.y = y
        self.missing_mod_indicator = missing_mod_indicator
        self.y_indicator = y_indicator


    def __len__(self):
        return len(self.missing_mod_indicator)


    def __getitem__(self, idx):
        Xs = [X[idx][0] if isinstance(X[idx][0], str) else X[idx] for X in self.Xs]
        sample = Xs, self.y[idx], self.missing_mod_indicator[idx],  self.y_indicator[idx]
        return sample
