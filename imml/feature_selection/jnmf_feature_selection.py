# License: BSD-3-Clause

import pandas as pd
import numpy as np

from ..decomposition import JNMF


class JNMFFeatureSelector(JNMF):
    r"""
    Feature selection for multi-modal datasets using the Joint Non-negative Matrix Factorization (JNMF) method.
    [#jnmfpaper1]_ [#jnmfpaper2]_ [#jnmfpaper3]_ [#jnmfpaper4]_ [#jnmfpaper5]_ [#jnmfpaper6]_ [#jnmfcode1]_ [#jnmfcode2]_

    This class extends the functionality of the `JNMF` method to perform feature selection across multiple modalities or
    blocks of data. The selected features are those with the highest contributions to the derived components from
    JNMF. This feature selection can be based on either the largest contribution for each component, the maximum
    overall contribution, or the average contribution across all components.

    Parameters
    ----------
    select_by : str, default="component"
        Criterion used to select features. Must be one of ["component", "max", "average"]:

        - "component": Selects the feature with the largest contribution for each component.
        - "max": Selects the features with the largest overall contribution.
        - "average": Selects the features with the highest average contribution across all components.

    f_per_component : int, default=1
        Number of features to select per component.

        - If `select_by="component"`, this controls how many features are selected for each component.
        - If `select_by="max"`, the top `n_components` * `f_per_component` features across all components are selected.
        - If `select_by="average"`, it selects `n_components` * `f_per_component` features with the highest average contribution for each component.

    kwargs : dict
        Arguments passed to the `JNMF` method.

    Attributes
    ----------
    selected_features_ : list of str of shape (n_components * f_per_component,)
        List of selected features.
    weights_ : list of float of shape (n_components * f_per_component,)
        The importance or contribution scores of the selected features in absolute values. These scores reflect how
        strongly each feature contributes to the components derived from JNMF.

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
    >>> from imml.feature_selection import JNMFFeatureSelector
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).uniform(size=(20, 10))) for i in range(3)]
    >>> transformer = JNMFFeatureSelector(n_components = 5)
    >>> transformed_Xs = transformer.fit_transform(Xs)
    """


    def __init__(self, select_by: str = "component", f_per_component: int = 1, **kwargs):
        select_by_options = ["max", "component", "average"]
        if select_by not in select_by_options:
            raise ValueError(f"Invalid select_by. Expected one of {select_by}. {select_by_options} was passed.")

        super().__init__(**kwargs)
        self.select_by = select_by
        self.f_per_component = f_per_component


    def fit(self, Xs, y = None):
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
        super().fit(Xs)
        hs = self.H_
        if isinstance(Xs[0], pd.DataFrame):
            hs = [pd.DataFrame(h, index=X.columns) for h,X in zip(hs,Xs)]
            hs = pd.concat(hs, axis=0)
        elif isinstance(Xs[0], np.ndarray):
            hs = [pd.DataFrame(h) for h in hs]
            hs = pd.concat(hs, axis=0)
            hs.columns = range(hs.columns.size)
        hs = hs.abs()
        selected_features = {}
        components = []
        if self.select_by == "component":
            for n in range(self.f_per_component):
                hs = hs.loc[:, hs.max().sort_values(ascending=False).index]
                for col in hs:
                    components.append(col)
                    component = hs[col]
                    feature = component.idxmax()
                    selected_features[feature] = component.max()
                    hs = hs.drop(labels=feature)
            self.component_ = components
        elif self.select_by == "average":
            hs = hs.mean(axis=1)
            for i in range(self.n_components * self.f_per_component):
                feature = hs.idxmax()
                selected_features[feature] = hs.max()
                hs = hs.drop(labels=feature)
        elif self.select_by == "max":
            hs = hs.stack().reset_index(drop=True, level=1)
            for i in range(self.n_components * self.f_per_component):
                feature = hs.idxmax()
                selected_features[feature] = hs.max()
                hs = hs.drop(labels=feature)
        self.selected_features_ = list(selected_features.keys())
        self.weights_ = list(selected_features.values())
        return self


    def transform(self, Xs):
        r"""
        Project data into the learned space.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs : list of array-likes objects, shape (n_samples, n_components)
            The projected data.
        """
        if isinstance(Xs[0], pd.DataFrame):
            transformed_Xs = [X.iloc[:, X.columns.isin(self.selected_features_)] for X in Xs]
        elif isinstance(Xs[0], np.ndarray):
            selected_features = np.array(self.selected_features_)
            dims = [X.shape[1] for X in Xs]
            dims = np.cumsum(dims)
            transformed_Xs = [X[:, selected_features[(selected_features >= dim - X.shape[1])
                                                     & (selected_features < dim)]] for X,dim in zip(Xs,dims)]
        return transformed_Xs


    def fit_transform(self, Xs, y = None, **fit_params):
        r"""
        Fit to data, then transform it.

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
            The projected data.
        """
        transformed_X = self.fit(Xs).transform(Xs)
        return transformed_X

