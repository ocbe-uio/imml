# License: BSD-3-Clause

import copy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import remove_mods


class Amputer(BaseEstimator, TransformerMixin):
    r"""
    Simulate an incomplete multi-modal dataset with block-wise missing data from a fully observed multi-modal dataset.

    Parameters
    ----------
    p: float, default=0.1
        Percentage of incomplete samples.
    mechanism: str, default="mem"
        One of ["mem", 'mcar', 'mnar', 'pm'], corresponding to mutually exclusive missing, missing completely at random,
        missing not at random, and partial missing, respectively.
    weights: list, default=None
        The probabilities associated with each number of missing modalities. If not given, the sample
        assumes a uniform distribution. Only used if mechanism = "mnar" or mechanism = "mem".
    random_state: int, default=None
        If int, random_state is the seed used by the random number generator.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = Amputer(p= 0.2, random_state=42)
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self, p: float = 0.1, mechanism: str = "mem", weights: list = None, random_state: int = None):

        mechanisms_options = ["mem", "mcar", "mnar", "pm"]
        if mechanism not in mechanisms_options:
            raise ValueError(f"Invalid mechanism. Expected one of: {mechanisms_options}")
        if p < 0 or p > 1:
            raise ValueError(f"Invalid p. Expected between 0 and 1.")

        self.mechanism = mechanism
        self.p = p
        self.weights = weights
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)


    def fit(self, Xs: list, y=None):
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
        self.n_mods = len(Xs)
        return self


    def transform(self, Xs: list):
        r"""
        Ampute a fully observed multi-modal dataset.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        transformed_Xs : list of array-likes objects, shape (n_samples, n_features), length n_mods
            The amputed multi-modal dataset.
        """
        if self.p > 0:
            pandas_format = isinstance(Xs[0], pd.DataFrame)
            if pandas_format:
                rownames = Xs[0].index
                colnames = [X.columns for X in Xs]
                Xs = [X.values for X in Xs]
            sample_names = pd.Index(list(range(len(Xs[0]))))

            if self.mechanism == "mem":
                pseudo_observed_mod_indicator = self._mem_mask(sample_names=sample_names)
            elif self.mechanism == "mcar":
                pseudo_observed_mod_indicator = self._mcar_mask(sample_names=sample_names)
            elif self.mechanism == "pm":
                pseudo_observed_mod_indicator = self._pm_mask(sample_names=sample_names)
            elif self.mechanism == "mnar":
                pseudo_observed_mod_indicator = self._mnar_mask(sample_names=sample_names)

            pseudo_observed_mod_indicator = pseudo_observed_mod_indicator.astype(bool)
            transformed_Xs = remove_mods(Xs=Xs, observed_mod_indicator=pseudo_observed_mod_indicator)

            if pandas_format:
                transformed_Xs = [pd.DataFrame(X, index=rownames, columns=colnames[X_idx])
                                  for X_idx, X in enumerate(transformed_Xs)]
        else:
            transformed_Xs = Xs

        return transformed_Xs


    def _mem_mask(self, sample_names):
        pseudo_observed_mod_indicator = pd.DataFrame(np.ones((len(sample_names), self.n_mods)), index=sample_names)
        common_samples = pd.Series(sample_names, index=sample_names).sample(frac=1 - self.p, replace=False,
                                                                            random_state=self.random_state).index
        sampled_names = copy.deepcopy(common_samples)
        if self.weights is None:
            n_missings = int(len(sample_names.difference(sampled_names)) / self.n_mods)
            n_missings = [n_missings] * self.n_mods
        else:
            n_missings = [int(len(sample_names.difference(sampled_names)) * w) for w in self.weights]
        for X_idx,n_missing in enumerate(n_missings):
            x_per_mod = sample_names.difference(sampled_names)
            if X_idx != self.n_mods - 1:
                x_per_mod = pd.Series(x_per_mod, index=x_per_mod).sample(n=n_missing,
                                                                            replace=False,
                                                                            random_state=self.random_state).index
            sampled_names = sampled_names.append(x_per_mod)
            idxs_to_remove = common_samples.append(x_per_mod)
            idxs_to_remove = sample_names.difference(idxs_to_remove)
            pseudo_observed_mod_indicator.loc[idxs_to_remove, X_idx] = 0
        return pseudo_observed_mod_indicator


    def _mcar_mask(self, sample_names):
        pseudo_observed_mod_indicator = pd.DataFrame(np.ones((len(sample_names), self.n_mods)), index=sample_names)
        common_samples = pd.Series(sample_names, index=sample_names).sample(frac=1 - self.p, replace=False,
                                                                            random_state=self.random_state).index
        idxs_to_remove = sample_names.difference(common_samples)
        shape = pseudo_observed_mod_indicator.loc[idxs_to_remove].shape
        mask = self.rng.choice(2, size=shape)
        mask = pd.DataFrame(mask, index=idxs_to_remove)
        samples_to_fix = mask.nunique(axis=1).eq(1)
        if samples_to_fix.any():
            samples_to_fix = samples_to_fix[samples_to_fix]
            mods_to_fix = self.rng.integers(low=0, high=self.n_mods, size=len(samples_to_fix))
            for mod_idx in np.unique(mods_to_fix):
                samples = mods_to_fix == mod_idx
                samples = samples_to_fix[samples].index
                mask.loc[samples, mod_idx] = np.invert(mask.loc[samples, mod_idx].astype(bool)).astype(int)

        pseudo_observed_mod_indicator.loc[idxs_to_remove] = mask.astype(int)
        return pseudo_observed_mod_indicator


    def _mnar_mask(self, sample_names):
        mask = pd.DataFrame(np.ones((len(sample_names), self.n_mods)), index=sample_names)
        common_samples = pd.Series(sample_names, index=sample_names).sample(frac=1 - self.p, replace=False,
                                                                            random_state=self.random_state).index
        idxs_to_remove = sample_names.difference(common_samples)
        reference_var = self.rng.choice(range(1, self.n_mods), p = self.weights, size=len(idxs_to_remove))
        reference_var = pd.Series(reference_var, index=idxs_to_remove)
        n_mods_to_remove = {n_mods_to_remove: self.rng.choice(self.n_mods, size=n_mods_to_remove, replace=False)
                            for n_mods_to_remove in np.unique(reference_var)}
        for keys,values in n_mods_to_remove.items():
            mask.loc[reference_var[reference_var == keys].index, values] = 0

        return mask


    def _pm_mask(self, sample_names):
        pseudo_observed_mod_indicator = pd.DataFrame(np.ones((len(sample_names), self.n_mods)), index=sample_names)
        common_samples = pd.Series(sample_names, index=sample_names).sample(frac=1 - self.p, replace=False,
                                                                            random_state=self.random_state).index
        idxs_to_remove = sample_names.difference(common_samples)
        n_incomplete_modalities = self.rng.choice(np.arange(1, self.n_mods), size=1)[0]
        if (self.n_mods == 2) or (n_incomplete_modalities == 1):
            col = self.rng.choice(self.n_mods)
            pseudo_observed_mod_indicator.loc[idxs_to_remove, col] = 0
        else:
            mask = self.rng.choice(2, size=(len(idxs_to_remove), n_incomplete_modalities))
            mask = pd.DataFrame(mask, index=idxs_to_remove,
                                columns=self.rng.choice(self.n_mods, size=n_incomplete_modalities, replace=False))
            samples_to_fix = mask.nunique(axis=1).eq(1)
            if samples_to_fix.any():
                samples_to_fix = samples_to_fix[samples_to_fix]
                mods_to_fix = self.rng.choice(mask.columns, size=len(samples_to_fix))
                for mod_idx in np.unique(mods_to_fix):
                    samples = mods_to_fix == mod_idx
                    samples = samples_to_fix[samples].index
                    mask.loc[samples, mod_idx] = np.invert(mask.loc[samples, mod_idx].astype(bool)).astype(int)
            pseudo_observed_mod_indicator.loc[idxs_to_remove, mask.columns] = mask.astype(int)
        return pseudo_observed_mod_indicator
