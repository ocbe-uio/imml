# License: BSD-3-Clause

import os.path
from os.path import dirname
from typing import Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import SpectralClustering
from sklearn.manifold import spectral_embedding

from ._snf import get_n_clusters, make_affinity
from ..impute import get_observed_mod_indicator
from ..utils import check_Xs_y
from ..preprocessing import remove_missing_samples_by_mod

try:
    from rpy2.robjects.packages import importr, PackageNotInstalledError
    import rpy2.robjects as robjects
    from ..utils import _convert_df_to_r_object
    rmodule_installed = True
except ImportError:
    rmodule_installed = False
    rmodule_error = "Module 'r' needs to be installed to use r engine. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"

if rmodule_installed:
    rbase = importr("base")
    r_folder = dirname(__file__)
    r_folder = os.path.join(r_folder, "_" + (os.path.basename(__file__).split(".")[0]))
    robjects.r['source'](os.path.join(r_folder, 'NEMO.R'))
    try:
        snftool = importr("SNFtool")
        snftool_installed = True
    except PackageNotInstalledError:
        snftool_installed = False
        snftool_module_error = "SNFtool needs to be installed in R to use r engine."


class NEMO(BaseEstimator, ClusterMixin):
    r"""
    NEighborhood based Multi-Omics clustering (NEMO). [#nemopaper]_ [#nemocode]_

    NEMO is a method used for clustering data from multiple modalities sources. This algorithm operates
    through three main stages. Initially, it constructs a similarity matrix for each modality that represents the
    similarities between different samples. Then, it merges these individual modality matrices into a unified one,
    combining the information from all modalities. Finally, the algorithm performs the actual clustering process on this
    integrated network, grouping similar samples together based on their multi-modal data patterns.

    Parameters
    ----------
    n_clusters : int or list-of-int
        The number of clusters to generate. If it is a list, the number of clusters will be estimated by the algorithm
        with this range of number of clusters to choose between.
    num_neighbors : list or int, default=None
        The number of neighbors to use for each modality. It can either be a number, a list of numbers or None. If it is a
        number, this is the number of neighbors used for all modalities. If this is a list, the number of neighbors are
        taken for each modality from that list. If it is None, each modality chooses the number of neighbors to be the number
        of samples divided by num_neighbors_ratio.
    num_neighbors_ratio : int, default=6
        The number of clusters to generate. If it is not provided, it will be estimated by the algorithm.
    metric : str or list-of-str, default="sqeuclidean"
        Distance metric to compute. Must be one of available metrics in :py:func`scipy.spatial.distance.pdist`. If
        multiple arrays a provided an equal number of metrics may be supplied.
    random_state : int, default=None
        Determines the randomness. Use an int to make the randomness deterministic.
    engine : str, default='python'
        Engine to use for computing the model. Must be one of ["python", "r"].
    verbose : bool, default=False
        Verbosity mode.

    Attributes
    ----------
    labels_ : array-like of shape (n_samples,)
        Labels of each point in training data.
    embedding_ : array-like of shape (n_samples, n_clusters)
        The final representation of the data to be used as input for the clustering step.
    n_clusters_ : int
        Final number of clusters.
    num_neighbors_ : int
        Final number of neighbors.
    affinity_matrix_ : np.array (n_samples, n_samples)
        Affinity matrix.

    References
    ----------
    .. [#nemopaper] Rappoport Nimrod, Shamir Ron. NEMO: Cancer subtyping by integration of partial multi-omic data.
                    Bioinformatics. 2019;35(18):3348–3356. doi: 10.1093/bioinformatics/btz058.
    .. [#nemocode] https://github.com/Shamir-Lab/NEMO

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.cluster import NEMO
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> estimator = NEMO(n_clusters = 2)
    >>> labels = estimator.fit_predict(Xs)
    """

    def __init__(self, n_clusters: Union[int,list] = 8, num_neighbors = None, num_neighbors_ratio: int = 6,
                 metric='sqeuclidean', random_state:int = None, engine: str = "python", verbose = False):
        engines_options = ["python", "r"]
        if engine not in engines_options:
            raise ValueError(f"Invalid engine. Expected one of {engines_options}. {engine} was passed.")
        if engine == "r":
            if not rmodule_installed:
                raise ImportError(rmodule_error)
            elif not snftool_installed:
                raise ImportError(snftool_module_error)

        if n_clusters is None:
            n_clusters = list(range(2, 16))
        self.n_clusters = n_clusters
        self.num_neighbors = num_neighbors
        self.num_neighbors_ratio = num_neighbors_ratio
        self.metric = metric
        self.random_state = random_state
        self.engine = engine
        self.verbose = verbose


    def fit(self, Xs, y=None):
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
        self :  Fitted estimator.
        """
        Xs = check_Xs_y(Xs, ensure_all_finite='allow-nan')

        if not isinstance(Xs[0], pd.DataFrame):
            Xs = [pd.DataFrame(X) for X in Xs]
        if self.engine == 'python':
            observed_mod_indicator = get_observed_mod_indicator(Xs)
            samples = observed_mod_indicator.index

            if self.num_neighbors is None:
                self.num_neighbors_ = [round(len(X)/self.num_neighbors_ratio) for X in Xs]
            elif not isinstance(self.num_neighbors, list):
                self.num_neighbors_ = [self.num_neighbors]*len(Xs)
            else:
                self.num_neighbors_ = self.num_neighbors

            # Work with numpy arrays for speed
            affinity_matrix_np = np.zeros((len(samples), len(samples)))

            for X, neigh, mod_idx in zip(Xs, self.num_neighbors_, range(len(Xs))):
                # Get mask for samples with this modality
                mod_mask = observed_mod_indicator.iloc[:, mod_idx].values
                X_filtered = X.values[mod_mask]

                # Compute affinity matrix
                sim_data = make_affinity(X_filtered, metric=self.metric, K=neigh, normalize=False)

                # Apply KNN threshold - vectorized
                non_sym_knn = self._apply_knn_threshold_vectorized(sim_data, neigh).T

                # Symmetrize
                sym_knn = non_sym_knn + non_sym_knn.T

                # Add to full affinity matrix at correct positions
                indices = np.where(mod_mask)[0]
                affinity_matrix_np[np.ix_(indices, indices)] += sym_knn

            # Compute shared modality count for normalization - vectorized
            # Convert to numpy int array for matrix multiplication (bool @ bool gives bool!)
            obs_mod_np = observed_mod_indicator.values.astype(int)
            # Use matrix multiplication: each element (i,j) is the dot product of row i and row j
            shared_omic_count = obs_mod_np @ obs_mod_np.T

            # Divide by shared count (this will create NaN where shared_omic_count is 0)
            with np.errstate(divide='ignore', invalid='ignore'):
                affinity_matrix_np = affinity_matrix_np / shared_omic_count

            # Fill entries where samples share no modalities with mean of lower triangular
            lower_tri_mask = np.tril(np.ones_like(affinity_matrix_np), k=-1).astype(bool)
            lower_tri_values = affinity_matrix_np[lower_tri_mask]
            # Exclude NaN/Inf values (these are where shared count was 0)
            lower_tri_values = lower_tri_values[np.isfinite(lower_tri_values)]
            if len(lower_tri_values) > 0:
                fill_value = np.mean(lower_tri_values)
                affinity_matrix_np[shared_omic_count == 0] = fill_value

            affinity_matrix = pd.DataFrame(affinity_matrix_np, index=samples, columns=samples)

            self.n_clusters_ = self.n_clusters if isinstance(self.n_clusters, int) else \
                get_n_clusters(arr= affinity_matrix.values, n_clusters= self.n_clusters)[0]

            model = SpectralClustering(n_clusters= self.n_clusters_, random_state= self.random_state,
                                       affinity="precomputed")
            labels = model.fit_predict(X= affinity_matrix)
            transformed_Xs = spectral_embedding(model.affinity_matrix_, n_components=self.n_clusters_,
                                                eigen_solver=model.eigen_solver, random_state=self.random_state,
                                                eigen_tol=model.eigen_tol, drop_first=False)
            self.embedding_ = transformed_Xs


        elif self.engine == "r":
            # Store original sample order
            original_samples = Xs[0].index if isinstance(Xs[0], pd.DataFrame) else list(range(len(Xs[0])))

            transformed_Xs = remove_missing_samples_by_mod(Xs=Xs)
            transformed_Xs = [X.T for X in transformed_Xs]
            transformed_Xs = _convert_df_to_r_object(transformed_Xs)
            num_neighbors = np.nan if self.num_neighbors is None else self.num_neighbors
            output = robjects.globalenv['nemo.affinity.graph'](transformed_Xs, num_neighbors,
                                                                        self.num_neighbors_ratio)
            affinity_matrix_r, self.num_neighbors_ = output[0], list(output[1])

            # Get row/column names from R affinity matrix
            r_sample_order = list(robjects.r['rownames'](affinity_matrix_r))

            if isinstance(self.n_clusters, list):
                self.n_clusters_ = int(robjects.globalenv['nemo.num.clusters'](affinity_matrix_r)[0])
            else:
                self.n_clusters_ = self.n_clusters
            if self.random_state is not None:
                rbase.set_seed(self.random_state)
            labels_r = snftool.spectralClustering(affinity_matrix_r, self.n_clusters_)
            labels_r = np.array(labels_r) - 1
            affinity_matrix_np = np.array(affinity_matrix_r)

            # Create mapping from R sample order to original sample order
            # R returns samples as strings, convert to same type as original
            if isinstance(original_samples, pd.Index):
                r_sample_order = pd.Index([type(original_samples[0])(x) for x in r_sample_order])
            else:
                r_sample_order = [type(original_samples[0])(x) for x in r_sample_order]

            # Reorder affinity matrix to match original sample order
            reorder_idx = [r_sample_order.tolist().index(s) if hasattr(r_sample_order, 'tolist')
                           else r_sample_order.index(s) for s in original_samples]
            affinity_matrix = affinity_matrix_np[np.ix_(reorder_idx, reorder_idx)]
            labels = labels_r[reorder_idx]

        self.labels_ = labels
        self.affinity_matrix_ = affinity_matrix

        return self


    def _predict(self, Xs):
        r"""
        Return clustering results for samples.

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.labels_


    def fit_predict(self, Xs, y=None):
        r"""
        Fit the model and return clustering results.
        Convenience method; equivalent to calling fit(X) followed by predict(X).

        Parameters
        ----------
        Xs : list of array-likes objects
            - Xs length: n_mods
            - Xs[i] shape: (n_samples, n_features_i)

            A list of different modalities.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """

        labels = self.fit(Xs)._predict(Xs)
        return labels


    def _apply_knn_threshold(self, row, neigh):
        threshold = np.sort(row.values)[::-1][neigh - 1] if neigh <= len(row) else 0
        row_filtered = row.copy()
        row_filtered[row < threshold] = 0
        row_sum = row_filtered.sum()
        if row_sum > 0:
            row_filtered[row >= threshold] = row_filtered[row >= threshold] / row_sum
        return row_filtered

    def _apply_knn_threshold_vectorized(self, matrix, neigh):
        """Vectorized version of _apply_knn_threshold for numpy arrays."""
        n = matrix.shape[0]
        result = np.zeros_like(matrix)

        # For each row, find the k-th largest value (threshold)
        # Sort in descending order and get the neigh-th element (index neigh-1)
        sorted_rows = np.sort(matrix, axis=1)[:, ::-1]
        thresholds = sorted_rows[:, min(neigh - 1, n - 1)][:, np.newaxis]

        # Keep only values >= threshold
        mask = matrix >= thresholds
        result[mask] = matrix[mask]

        # Normalize each row by its sum
        row_sums = result.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        result = result / row_sums

        return result
