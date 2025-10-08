# License: BSD-3-Clause

import pandas as pd

from ..decomposition import DFMF
from ..utils import check_Xs


class DFMFImputer(DFMF):
    r"""
    Impute missing data in a dataset using the `DFMF` method. [#dfmfpaper]_ [#dfmfcode]_

    This class extends the `DFMF` class to provide functionality for filling in incomplete samples by
    addressing both block-wise and feature-wise missing data. As a subclass of DFMF, `DFMFImputer` inherits all
    input parameters and attributes from `DFMF`. Consequently, it uses the same `fit` method as DFMF for
    training the model.

    References
    ----------
    .. [#dfmfpaper] M. Žitnik and B. Zupan, "Data Fusion by Matrix Factorization," in IEEE Transactions on Pattern
                    Analysis and Machine Intelligence, vol. 37, no. 1, pp. 41-53, 1 Jan. 2015,
                    doi: 10.1109/TPAMI.2014.2343973.
    .. [#dfmfcode] https://github.com/mims-harvard/scikit-fusion/tree/master

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import DFMFImputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = DFMFImputer(n_components = 5)
    >>> labels = transformer.fit_transform(Xs)
    """


    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def fit_transform(self, Xs, y = None, **fit_params):
        r"""
        Fit to data, then impute them.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples_i, n_features_i)

            A list of different mods.
        y : Ignored
            Not used, present here for API consistency by convention.
        fit_params : Ignored
            Not used, present here for API consistency by convention.

        Returns
        -------
        transformed_X : array-likes objects of shape (n_samples, n_components)
            The transformed data with filled missing samples.
        """

        self.fit(Xs)
        imputed_Xs = [self.fuser_.complete(relation) for relation in self.fuser_.fusion_graph.relations]
        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        transformed_Xs = []
        for X, transformed_X in zip(Xs, imputed_Xs):
            transformed_X = pd.DataFrame(transformed_X, columns=X.columns)
            transformed_Xs.append(X.fillna(transformed_X))

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        return transformed_Xs
