# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

try:
    import torch
    from torch import Tensor
except ImportError:
    Tensor = str
from ..utils import check_Xs_y


class ObservedModIndicator(FunctionTransformer):
    r"""
    Binary indicator for observed modalities. Apply FunctionTransformer (from Scikit-learn) with
    get_observed_mod_indicator as a function.

    Note that this component typically should not be used in a vanilla Pipeline consisting of preprocessing and
    an estimator.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import ObservedModIndicator
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p= 0.2, random_state=42).fit_transform(Xs)
    >>> indicator = ObservedModIndicator()
    >>> observed_mod = indicator.fit_transform()(Xs)
    """


    def __init__(self):
        super().__init__(get_observed_mod_indicator)


def get_observed_mod_indicator(Xs : list, y = None):
    r"""
    Return a binary indicator for observed modalities.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features)

        A list of different modalities.
    y : Ignored
        Not used, present here for API consistency by convention.

    Returns
    -------
    transformed_X : array-likes objects, shape (n_samples, n_mods)
        Binary indicator for observed modalities.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import get_observed_mod_indicator
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p= 0.2, random_state=42).fit_transform(Xs)
    >>> observed_mod = get_observed_mod_indicator()(Xs)
    """
    transformed_Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
    if isinstance(transformed_Xs[0], Tensor):
        transformed_Xs = [X.numpy() for X in transformed_Xs]
    transformed_X = np.vstack([pd.isna(X).all(1) for X in transformed_Xs]).T
    transformed_X = np.logical_not(transformed_X)
    if isinstance(Xs[0], pd.DataFrame):
        transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)
    elif isinstance(Xs[0], Tensor):
        transformed_X = torch.from_numpy(transformed_X).bool()
    return transformed_X

