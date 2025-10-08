# License: BSD-3-Clause

from typing import Union

from . import check_Xs


def convert_dataset_format(Xs: list, keys: list = None) -> Union[list, dict]:
    r"""
    Convert the format of a multi-modal dataset. If it is a dict, it will be converted to dict, and if it is a list,
    it will be converted to dict.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    keys : list, default=None
        keys for the dict. If None, it will use numbers starting from 0. Only used when to_dict is True.

    Returns
    -------
    transformed_Xs: dict of array-likes objects.
        - Xs length: n_mods
        - Xs[key] shape: (n_samples, n_features_i)

    Examples
    --------
    >>> from imml.utils.convert_dataset_format import convert_dataset_format    >>> import numpy as np
    >>> import pandas as pd
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> convert_dataset_format(Xs = Xs)
    """
    if isinstance(Xs, dict):
        check_Xs(Xs=list(Xs.values()), ensure_all_finite="allow-nan")
        transformed_Xs = list(Xs.values())
    elif isinstance(Xs, list):
        if keys is None:
            keys = list(range(len(Xs)))
        transformed_Xs = {key:X for key,X in zip(keys, Xs)}
    else:
        raise ValueError(f"Invalid Xs. It must be a list or dict of array-likes objects. A {type(Xs)} was passed.")

    return transformed_Xs
