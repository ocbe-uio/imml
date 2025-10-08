# License: BSD-3-Clause

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer

from ..decomposition import JNMF


class JNMFImputer(JNMF):
    r"""
    Impute missing data in a dataset using the `JNMF` method.
    [#jnmfpaper1]_ [#jnmfpaper2]_ [#jnmfpaper3]_ [#jnmfpaper4]_ [#jnmfpaper5]_ [#jnmfpaper6]_ [#jnmfcode1]_ [#jnmfcode2]_

    This class extends the `JNMF` class to provide functionality for filling in incomplete samples by
    addressing both block-wise and feature-wise missing data. As a subclass of `JNMF`, `JNMFImputer` inherits all
    input parameters and attributes from `JNMF`. Consequently, it uses the same `fit` method as `JNMF`
    training the model.

    References
    ----------
    .. [#jnmfpaper1] Tsuyuzaki et al., (2023). nnTensor: An R package for non-negative matrix/tensor decomposition.
                     Journal of Open Source Software, 8(84), 5015, https://doi.org/10.21105/joss.05015
    .. [#jnmfpaper2] Liviu Badea, (2008) Extracting Gene Expression Profiles Common to Colon and Pancreatic
                     Adenocarcinoma using Simultaneous nonnegative matrix factorization. Pacific Symposium on
                     Biocomputing 13:279-290.
    .. [#jnmfpaper3] Shihua Zhang, et al. (2012) Discovery of multi-dimensional modules by integrative analysis of
                     cancer genomic data. Nucleic Acids Research 40(19), 9379-9391.
    .. [#jnmfpaper4] Zi Yang, et al. (2016) A non-negative matrix factorization method for detecting modules in
                     heterogeneous omics multi-modal data, Bioinformatics 32(1), 1-8.
    .. [#jnmfpaper5] Y. Kenan Yilmaz et al., (2010) Probabilistic Latent Tensor Factorization, International Conference
                     on Latent Variable Analysis and Signal Separation 346-353.
    .. [#jnmfpaper6] N. Fujita et al., (2018) Biomarker discovery by integrated joint non-negative matrix factorization
                     and pathway signature analyses, Scientific Report.
    .. [#jnmfcode1] https://rdrr.io/cran/nnTensor/man/JNMF.html
    .. [#jnmfcode2] https://github.com/rikenbit/nnTensor

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.impute import JNMFImputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = JNMFImputer(n_components = 5)
    >>> labels = transformer.fit_transform(Xs)
    """


    def __init__(self, filling: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.filling = filling


    def transform(self, Xs):
        r"""
        Impute unseen data.

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
        transformed_Xs = [np.dot(transformed_X + V, H.T)
                          for transformed_X,V,H in zip(super().transform(Xs), self.V_, self.H_)]

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        return transformed_Xs


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

        if self.filling:
            transformed_Xs_jnmf = [SimpleImputer().set_output(transform="pandas").fit_transform(X) for X in Xs]
            transformed_Xs_jnmf = super().fit_transform(transformed_Xs_jnmf)
        else:
            transformed_Xs_jnmf = super().fit_transform(Xs)
        transformed_Xs = []
        for X, V, H in zip(Xs, self.V_, self.H_):
            transformed_X = np.dot(transformed_Xs_jnmf + V, H.T)
            if isinstance(Xs[0], pd.DataFrame):
                transformed_X = X.fillna(pd.DataFrame(transformed_X, index=X.index, columns=X.columns))
            else:
                transformed_X = pd.DataFrame(X).fillna(pd.DataFrame(transformed_X))
            transformed_Xs.append(transformed_X)

        if self.transform_ == "pandas":
            transformed_Xs = [pd.DataFrame(transformed_X, index=X.index, columns=X.columns)
                              for transformed_X, X in zip(transformed_Xs, Xs)]
        elif self.transform_ == "numpy":
            transformed_Xs = [transformed_X.values for transformed_X in transformed_Xs]

        return transformed_Xs

