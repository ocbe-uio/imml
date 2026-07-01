# License: BSD-3-Clause

from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..utils import check_Xs_y


class UMTransformer(BaseEstimator, TransformerMixin):
    r"""
    A transformer that applies a unimodal transformation. It first concatenates the modalities, then applies
    the transformation, and finally split the concatenated data back to the original modalities.

    Parameters
    ----------
    transformer : scikit-learn transformer object
        A scikit-learn transformer object that will be used to transform the data.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import UMTransformer
    >>> from sklearn.impute import KNNImputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = UMTransformer(transformer = KNNImputer())
    >>> transformer.fit_transform(Xs)
    """


    def __init__(self, transformer):
        try:
            assert hasattr(transformer, "fit") and callable(getattr(transformer, "fit"))
        except AssertionError:
            raise ValueError("transformer must be a scikit-learn transformer like object")

        self.transformer = transformer


    def fit(self, Xs, y = None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : array-like, shape (n_samples,)
            Labels for each sample. Only used by supervised algorithms.

        Returns
        -------
        self :  returns an instance of self.
        """

        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        transformed_Xs = np.concatenate(Xs, axis=1)
        self.transformer.fit(transformed_Xs)
        return self


    def transform(self, Xs):
        r"""
        Transform the input data using the transformers.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs : list of array-likes objects, shape (n_samples, n_features_i)
            A list of transformed mods of data, one for each input modality.
        """

        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')
        shapes = [X.shape[1] for X in Xs]
        transformed_Xs = np.concatenate(Xs, axis=1)
        transformed_Xs = self.transformer.transform(transformed_Xs)
        transformed_Xs = [transformed_Xs[:, sum(shapes[:i]):sum(shapes[:i]) + shape] for i, shape in enumerate(shapes)]
        return transformed_Xs
