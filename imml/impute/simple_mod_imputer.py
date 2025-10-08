# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ..impute import get_observed_mod_indicator
from ..utils import check_Xs


class SimpleModImputer(BaseEstimator, TransformerMixin):
    r"""
    Fill incomplete samples of a dataset using a specified method.

    Parameters
    ----------
    value : str, optional (default='mean')
        The method to use for filling missing modalities. Possible values:
        - 'mean': replace missing samples with the mean of each feature in the corresponding modality.
        - 'zeros': replace missing samples with zeros.

    Attributes
    ----------
    features_mod_mean_list_ : array-like of shape (n_mods,)
        The mean value of each feature in the corresponding modality, if value='mean'
    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import SimpleModImputer
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> amp = Amputer(p=0.3, random_state=42)
    >>> Xs = amp.fit_transform(Xs)
    >>> transformer = SimpleModImputer(value = 'mean')
    >>> transformer.fit_transform(Xs)
    """


    def __init__(self, value : str = 'mean'):

        values = ['mean', 'zeros']
        if value not in values:
            raise ValueError(f"Invalid value. Expected one of: {values}")
        self.value = value


    def fit(self, Xs : list, y=None):
        r"""
        Fit the transformer to the input data.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.
        y : Ignored
                Not used, present here for API consistency by convention.

        Returns
        -------
        self :  returns an instance of self.
        """

        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        if self.value == "mean":
            self.features_mod_mean_list_ = [np.nanmean(X, axis=0) for X in Xs]
        elif self.value == "zeros":
            pass
        return self


    def transform(self, Xs : list):
        r"""
        Transform the input data by filling missing samples.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs : list of array-likes objects, shape (n_samples, n_features_i)
            The transformed data with filled missing samples.
        """

        Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
        pandas_format = isinstance(Xs[0], pd.DataFrame)
        if pandas_format:
            samples = Xs[0].index
            features = [X.columns for X in Xs]
            dtypes = [X.dtypes.to_dict() for X in Xs]
            Xs = [X.values for X in Xs]
        observed_mod_indicator = get_observed_mod_indicator(Xs = Xs)
        n_samples = len(observed_mod_indicator)

        transformed_Xs = []
        for X_idx, X in enumerate(Xs):
            n_features = X.shape[1]
            if self.value == "mean":
                features_mod_mean = self.features_mod_mean_list_[X_idx]
                transformed_X = np.tile(features_mod_mean, (n_samples ,1))
            elif self.value == "zeros":
                transformed_X = np.zeros((n_samples, n_features))
            transformed_X[observed_mod_indicator[:, X_idx]] = X[observed_mod_indicator[:, X_idx]]
            if pandas_format:
                transformed_X = pd.DataFrame(transformed_X, index=samples, columns=features[X_idx])
                transformed_X = transformed_X.astype(dtypes[X_idx])
            transformed_Xs.append(transformed_X)
        return transformed_Xs


def simple_mod_imputer(Xs : list, y = None, value : str = 'mean'):
    r"""
    Transform the input data by filling missing samples.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features)
        A list of different modalities.
    y : Ignored
            Not used, present here for API consistency by convention.
    value : str, optional (default='mean')
        The method to use for filling missing modalities. Possible values:
        - 'mean': replace missing samples with the mean of each feature in the corresponding modality.
        - 'zeros': replace missing samples with zeros.

    Returns
    -------
    transformed_Xs : list of array-likes objects, shape (n_samples, n_features_i)
        The transformed data with filled missing samples.
    """
    Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
    pandas_format = isinstance(Xs[0], pd.DataFrame)
    if pandas_format:
        samples = Xs[0].index
        features = [X.columns for X in Xs]
        dtypes = [X.dtypes.to_dict() for X in Xs]
        Xs = [X.values for X in Xs]
    observed_mod_indicator = get_observed_mod_indicator(Xs=Xs)
    n_samples = len(observed_mod_indicator)

    transformed_Xs = []
    for X_idx, X in enumerate(Xs):
        n_features = X.shape[1]
        if value == "mean":
            features_mod_mean = np.nanmean(X, axis=0)
            transformed_X = np.tile(features_mod_mean, (n_samples, 1))
        elif value == "zeros":
            transformed_X = np.zeros((n_samples, n_features))
        else:
            raise ValueError(f"Invalid value. Expected one of: ['mean', 'zeros']")

        transformed_X[observed_mod_indicator[:, X_idx]] = X[observed_mod_indicator[:, X_idx]]
        if pandas_format:
            transformed_X = pd.DataFrame(transformed_X, index=samples, columns=features[X_idx])
            transformed_X = transformed_X.astype(dtypes[X_idx])
        transformed_Xs.append(transformed_X)
    return transformed_Xs
