# License: BSD-3-Clause

try:
    import torch
    DLBaseDataset = torch.utils.data.Dataset
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
    DLBaseDataset = object


class MRGCNDataset(DLBaseDataset):
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
        if not isinstance(Xs, list):
            raise ValueError(f"Invalid Xs. It must be a list of array-likes objects. A {type(Xs)} was passed.")
        if len(Xs) < 2:
            raise ValueError(f"Invalid Xs. It must have at least two modalities. Got {len(Xs)} modalities.")
        if any(len(X) == 0 for X in Xs):
            raise ValueError("Invalid Xs. All elements must have at least one sample.")
        if len(set(len(X) for X in Xs)) > 1:
            raise ValueError("Invalid Xs. All elements must have the same number of samples.")

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
