# License: BSD-3-Clause

import numpy as np
from sklearn.preprocessing import FunctionTransformer

from ..utils import check_Xs


class RemoveIncomSamplesByMod(FunctionTransformer):
    r"""
    Remove incomplete samples from each specific modality. Apply `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_ (from `Scikit-learn`)
    with `remove_samples_missing_mods` as a function.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import RemoveMissingSamplesByMod
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> transformer = RemoveMissingSamplesByMod()
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self):
        super().__init__(remove_incom_samples_by_mod)


def remove_incom_samples_by_mod(Xs: list) -> list:
    r"""
    Remove incomplete samples from each specific modality.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    transformed_Xs: list of array-likes objects.
        - Xs length: n_mods
        - Xs[i] shape: (n_samples_i, n_features_i)

        A list of different modalities.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.ampute import Amputer
    >>> from imml.preprocessing import remove_missing_samples_by_mod
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> remove_missing_samples_by_mod(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    transformed_Xs = [X[np.isfinite(X).all(axis=1)] for X in Xs]
    return transformed_Xs
