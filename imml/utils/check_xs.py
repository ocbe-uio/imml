# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.utils import check_array


def check_Xs(Xs, enforce_modalities=None, copy=False, ensure_all_finite="allow-nan",return_dimensions=False):
    r"""
    Checks Xs and ensures it to be a list of 2D matrices. Adapted from `̀mvlearn` [#checkxspaper]_ [#checkxscode]_ .

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    enforce_modalities : int, (default=not checked)
        If provided, ensures this number of modalities in Xs. Otherwise not checked.
    copy : boolean, (default=False)
        If True, the returned Xs is a copy of the input Xs, and operations on the output will not affect the input.
        If False, the returned Xs is a modality of the input Xs, and operations on the output will change the input.
    ensure_all_finite : bool or 'allow-nan', default='allow-nan'
        Whether to raise an error on np.inf, np.nan, pd.NA in array. The possibilities are:

        - True: Force all values of array to be finite.
        - False: accepts np.inf, np.nan, pd.NA in array.
        - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
          cannot be infinite.
    return_dimensions : boolean, (default=False)
        If True, the function also returns the dimensions of the multi-modal dataset. The dimensions are n_mods,
        n_samples, n_features where n_samples and n_mods are respectively the number of modalities and the number of
        samples, and n_features is a list of length n_mods containing the number of features of each modality.

    References
    ----------
    .. [#checkxspaper] Perry, Ronan, et al. "mvlearn: Multiview Machine Learning in Python." Journal of Machine
                      Learning Research 22.109 (2021): 1-7.
    .. [#checkxscode] https://mvlearn.github.io/references/utils.html

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
        if not isinstance(Xs, np.ndarray):
            msg = f"If not list, input must be of type np.ndarray,\
                not {type(Xs)}"
            raise ValueError(msg)
        if Xs.ndim == 2:
            Xs = [Xs]
        else:
            Xs = list(Xs)

    n_mods = len(Xs)
    if n_mods == 0:
        msg = "Length of input list must be greater than 0"
        raise ValueError(msg)

    if enforce_modalities is not None and n_mods != enforce_modalities:
        msg = "Wrong number of modalities. Expected {} but found {}".format(
            enforce_modalities, n_mods
        )
        raise ValueError(msg)

    pandas_format = True if isinstance(Xs[0],pd.DataFrame) else False
    if pandas_format:
        Xs = [pd.DataFrame(check_array(X, allow_nd=False, copy=copy, ensure_all_finite=ensure_all_finite),
                           index=X.index, columns=X.columns) for X_idx, X in enumerate(Xs)]
    else:
        Xs = [check_array(X, allow_nd=False, copy=copy, ensure_all_finite=ensure_all_finite) for X in Xs]

    if return_dimensions:
        n_samples = Xs[0].shape[0]
        n_features = [X.shape[1] for X in Xs]
        return Xs, n_mods, n_samples, n_features
    else:
        return Xs
