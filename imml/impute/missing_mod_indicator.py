# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ..utils import check_Xs

class MissingModIndicator(FunctionTransformer):
    r"""
    Binary indicators for missing modalities. Apply FunctionTransformer (from Scikit-learn) with
    get_missing_mod_indicator as a function.

    Note that this component typically should not be used in a vanilla Pipeline consisting of preprocessing and
    an estimator.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import MissingModIndicator
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p= 0.2, random_state=42).fit_transform(Xs)
    >>> indicator = MissingModIndicator()
    >>> missing_mod = indicator.fit_transform()(Xs)
    """

    def __init__(self):
        super().__init__(get_missing_mod_indicator)


def get_missing_mod_indicator(Xs : list, y = None):
    r"""
    Return a binary indicator for missing modalities.

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
        Binary indicator for missing modalities.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import get_missing_mod_indicator
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p= 0.2, random_state=42).fit_transform(Xs)
    >>> missing_mod = get_missing_mod_indicator()(Xs)
    """
    Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
    transformed_X = np.vstack([np.isnan(X).all(1) for X in Xs]).T
    if isinstance(Xs[0], pd.DataFrame):
        transformed_X = pd.DataFrame(transformed_X, index=Xs[0].index)
    return transformed_X

