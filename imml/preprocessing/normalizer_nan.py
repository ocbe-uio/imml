# License: BSD-3-Clause

import numpy as np
from sklearn.preprocessing import Normalizer
from sklearn.utils import check_array
from sklearn.utils.validation import validate_data


class NormalizerNaN(Normalizer):
    r"""
    Similar to sklearn.preprocessing.Normalizer but handles NaN values.

    Parameters
    ----------
    norm : {‘l1’, ‘l2’, ‘max’}, default=’l2’
        The norm to use to normalize each non zero sample. If norm=’max’ is used, values will be rescaled by
        the maximum of the absolute values.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.ampute import Amputer
    >>> from imml.preprocessing import NormalizerNaN, MultiModTransformer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> amp = Amputer(p=0.3)
    >>> Xs = amp.fit_transform(Xs)
    >>> transformer = MultiModTransformer(NormalizerNaN())
    >>> transformer.fit_transform(Xs)
    """


    def __init__(self, norm : str = 'l2'):

        super().__init__(norm)
        values = ['l1', 'l2', 'max']
        if norm not in values:
            raise ValueError(f"Invalid value. Expected one of: {values}")
        self.norm = norm


    def fit(self, X, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.
        Returns
        -------
        self :  returns an instance of self.
        """

        validate_data(self, X=X, **{"ensure_all_finite":'allow-nan'})
        return self


    def transform(self, X, y=None):
        r"""
        Scale each non zero row of X to unit norm.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
        y : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-like of shape (n_samples, n_features)
            Transformed data.
        """

        X = check_array(X, ensure_all_finite='allow-nan')
        if self.norm == "l1":
            norms = np.nansum(np.abs(X), axis=1)
        elif self.norm == "l2":
            norms = np.nansum(X**2, axis=1)
            norms = np.sqrt(norms, norms)
        elif self.norm == "max":
            norms = np.nanmax(np.abs(X), axis=1)
        norms[norms == 0] = 1.
        transformed_X = X / norms[:,None]
        return transformed_X

