# License: BSD-3-Clause

import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ..utils import check_Xs


class RemoveMods(FunctionTransformer):
    r"""
    A transformer that generates block-wise missingness patterns in complete multi-modal datasets. Apply
    `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_ (from `Scikit-learn`) with `remove_modalities` as a function.

    Parameters
    ----------
    observed_mod_indicator: array-like of shape (n_samples, n_mods)
        Boolean array-like indicating observed modalities for each sample.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.ampute import RemoveMods
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> observed_mod_indicator = np.random.default_rng(42).choice(2, size=(len(Xs[0]), len(Xs)))
    >>> transformer = RemoveMods(observed_mod_indicator = observed_mod_indicator)
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self, observed_mod_indicator):
        self.observed_mod_indicator = observed_mod_indicator
        super().__init__(remove_mods, kw_args={"observed_mod_indicator": self.observed_mod_indicator})


def remove_mods(Xs: list, observed_mod_indicator):
    r"""
    A function that generates block-wise missingness patterns in complete multi-modal datasets.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mods.

    Returns
    -------
    transformed_X : list of array-likes objects (n_samples, n_features_i)
        The transformed multi-modal dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.ampute import remove_mods
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> observed_mod_indicator = np.random.default_rng(42).choice(2, size=(len(Xs[0]), len(Xs)))
    >>> remove_mods(Xs=Xs, observed_mod_indicator = observed_mod_indicator)
    """

    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    transformed_Xs = []
    if isinstance(observed_mod_indicator, pd.DataFrame):
        observed_mod_indicator = observed_mod_indicator.values
    for X_idx, X in enumerate(Xs):
        idxs_to_remove = observed_mod_indicator[:, X_idx] == False
        if isinstance(X, pd.DataFrame):
            X = X.values
        transformed_X = copy.deepcopy(X).astype(float)
        transformed_X[idxs_to_remove, :] = np.nan
        transformed_Xs.append(transformed_X)
    if isinstance(Xs[0], pd.DataFrame):
        transformed_Xs = [pd.DataFrame(transformed_X, columns=X.columns, index=X.index) for X, transformed_X in
                          zip(Xs, transformed_Xs)]

    return transformed_Xs
