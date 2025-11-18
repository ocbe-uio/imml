# License: BSD-3-Clause

from ..utils import check_Xs_y
from .. import deepmodule_installed, deepmodule_error, Dataset

if deepmodule_installed:
    import torch


class MRGCNDataset(Dataset):
    r"""
    This class provides a `torch.utils.data.Dataset` implementation for handling multi-modal datasets with `MRGCN`.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    transform : list of callable, defult=None
        A list of functions or transformations to apply to each sample in the dataset.

    Example
    --------
    >>> import numpy as np
    >>> import torch
    >>> from imml.load import MRGCNDataset
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> train_data = MRGCNDataset(Xs=Xs)
    """

    def __init__(self, Xs: list, transform = None):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)
        Xs = check_Xs_y(Xs=Xs)

        self.Xs = Xs
        self.transform = transform


    def __len__(self):
        return len(self.Xs[0])


    def __getitem__(self, idx):
        if self.transform is not None:
            Xs = [self.transform[idx](X[idx]) for X in self.Xs]
        else:
            Xs = [X[idx] for X in self.Xs]
        Xs = tuple(Xs)
        return Xs
