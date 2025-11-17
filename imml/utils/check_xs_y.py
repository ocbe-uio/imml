# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.utils import check_array

try:
    import torch
except ImportError:
    torch = object


def check_Xs_y(Xs: list, y = None, modalities : list = None, mod_types : list = None, copy=False,
               ensure_all_finite="allow-nan", return_dimensions=False, supervised: bool = False):
    r"""
    Checks Xs and y and ensures they have the correct format.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    y : array-like of shape (n_samples,), (default=None)
        Target vector relative to X.
    modalities : list of str, default=None
        If provided, ensures the number of modalities. Otherwise not checked.
    mod_types : list of str, default=None
        If provided, ensures the type of modalities. Otherwise not checked.
    copy : boolean, default=False
        If True, the returned Xs is a copy of the input Xs, and operations on the output will not affect the input.
        If False, the returned Xs is a modality of the input Xs, and operations on the output will change the input.
    ensure_all_finite : bool or 'allow-nan', default='allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
    return_dimensions : boolean, default=False
        If True, the function also returns the dimensions of the multi-modal dataset. The dimensions are n_mods,
        n_samples, n_features where n_samples and n_mods are respectively the number of modalities and the number of
        samples, and n_features is a list of length n_mods containing the number of features of each modality.
    supervised : bool, default=False
        If True, it checks y.

    Returns
    -------
    Xs_converted : object
        The converted and validated Xs (list of data arrays).
    n_mods : int
        The number of modalities in the dataset. Returned only if
        ``return_dimensions`` is ``True``.
    n_samples : int
        The number of samples in the dataset. Returned only if
        ``return_dimensions`` is ``True``.
    n_features : list
        List of length ``n_mods`` containing the number of features in
        each modality. Returned only if ``return_dimensions`` is ``True``.
    """
    if not isinstance(Xs, list):
        raise ValueError(f"Invalid Xs. It must be a list. A {type(Xs)} was passed.")
    n_mods = len(Xs)
    if len(Xs) < 2:
        raise ValueError(f"Invalid Xs. It must have at least two modalities. Got {n_mods} modalities.")
    if any(len(X) == 0 for X in Xs):
        raise ValueError(f"Invalid Xs. All elements must have at least one sample. Got {[len(X) for X in Xs]}.")
    if (modalities is not None) and (not isinstance(modalities, list)):
        raise ValueError(f"Invalid modalities. It must be a list. A {type(modalities)} was passed.")
    if isinstance(modalities, list) and (n_mods != len(modalities)):
        raise ValueError(f"Invalid modalities. Wrong number of modalities. Expected {len(modalities)} but found {n_mods}")
    if (mod_types is not None) and (not isinstance(mod_types, list)):
        raise ValueError(f"Invalid mod_types. It must be a list. A {type(mod_types)} was passed.")
    if isinstance(mod_types, list) and (n_mods != len(mod_types)):
        raise ValueError(f"Invalid mod_types. Wrong number of mod_types. Expected {len(mod_types)} but found {n_mods}")
    if isinstance(mod_types, list) and (not all(mod in mod_types for mod in modalities)):
        raise ValueError(f"Invalid modalities. Expected options are: {mod_types}")
    if len(set([len(X) for X in Xs])) != 1:
        raise ValueError(f"Invalid Xs. All modalities should have the same number of samples. Got {[len(X) for X in Xs]}.")
    dtype = type(Xs[0])
    if not all(isinstance(X, dtype) for X in Xs):
        raise ValueError(f"Invalid Xs. All modalities should be the same data type. Got {[type(X) for X in Xs]}.")
    if pd.concat([pd.DataFrame(X) for X in Xs], axis=1).isna().all(1).any():
        raise ValueError(f"Invalid Xs. There are samples with no available data.")

    if supervised:
        if y is None:
            raise ValueError("Invalid y. It cannot be None.")
        if len(y) != len(Xs[0]):
            raise ValueError(f"Invalid y. It must have the same length as each element in Xs. Got {len(y)} vs {len(Xs[0])}")

    if isinstance(Xs[0],pd.DataFrame):
        Xs = [pd.DataFrame(check_array(X, allow_nd=False, copy=copy, ensure_all_finite=ensure_all_finite, dtype=None),
                           index=X.index, columns=X.columns) for X_idx, X in enumerate(Xs)]
    elif isinstance(Xs[0], np.ndarray) or isinstance(Xs[0], list):
        Xs = [check_array(X, allow_nd=False, copy=copy, ensure_all_finite=ensure_all_finite, dtype=None) for X in Xs]
    elif isinstance(Xs[0], torch.Tensor):
        Xs = [torch.from_numpy(check_array(X, allow_nd=False, copy=copy, ensure_all_finite=ensure_all_finite, dtype=None))
              for X in Xs]

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_mods, n_samples, n_features
    else:
        return Xs
