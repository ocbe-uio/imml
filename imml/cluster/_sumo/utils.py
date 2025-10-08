# License: BSD-3-Clause

from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import adjusted_rand_score, mutual_info_score
from sklearn.metrics.cluster import entropy
from .constants import CLUSTER_METRICS
from typing import Union
import numpy as np
import os
import pandas as pd
import warnings


def docstring_formatter(*args, **kwargs):
    """ Decorator allowing for printing variable values in docstrings"""

    def dec(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec

def check_matrix_symmetry(m: np.ndarray, tol=1e-8, equal_nan=True):
    """ Check symmetry of numpy array, after removal of missing samples"""
    # filter missing samples
    row_idx = np.array([i for i in range(m.shape[0]) if not np.all(np.isnan(m[i, :]))])
    col_idx = np.array([i for i in range(m.shape[1]) if not np.all(np.isnan(m[:, i]))])
    filtered = m[row_idx[:, None], col_idx]
    if filtered.shape[0] != filtered.shape[1]:
        return False
    return np.allclose(filtered, filtered.T, atol=tol, equal_nan=equal_nan)


def check_categories(a: np.ndarray):
    """ Check categories in data"""
    return list(np.unique(a[~np.isnan(a)]))


def load_npz(file_path: str):
    """ Load data from .npz file

    Args:
        file_path (str): path to .npz file

    Returns:
        dictionary with arrays as values and their indices used during saving to .npz file as keys

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("File {} does not exist".format(file_path))

    not_npz = False
    if not os.path.isfile(file_path):
        not_npz = True
    else:
        try:
            npz = np.load(file_path, allow_pickle=True)
        except OSError:
            not_npz = True

    if not_npz:
        raise ValueError("Failed to load file - file {} was saved incorrectly or is not an .npz file".format(file_path))

    return {i: npz[i] for i in npz.files}


def save_arrays_to_npz(data: Union[dict, list], file_path: str):
    """ Save numpy arrays to .npz file

    Args:
        data (dict/list): list of numpy arrays or dictionary with specified keywords for every array
        file_path (str): optional path to output file

    """
    arrays = list(data.values()) if isinstance(data, dict) else data
    if not all([isinstance(arrays[i], np.ndarray) for i in range(len(arrays))]):
        raise ValueError("Incorrect data arrays")

    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_path += ".npz" if ".npz" != file_path[-4:] else ""

    if isinstance(data, dict):
        np.savez(file=file_path, **data)
    else:
        args = {str(i): arrays[i] for i in range(len(arrays))}
        np.savez(file=file_path, **args)


def extract_ncut(a: np.ndarray, k: int, random_state):
    """ Select clusters using normalized cut based on graph similarity matrix

    Args:
        a (Numpy.ndarray): symmetric similarity matrix
        k (int): number of clusters

    Returns:
        one dimensional array containing clusters ids for every node
    """
    assert check_matrix_symmetry(a)

    d = np.diag(np.power(np.sum(a, axis=1) + np.spacing(1), -0.5))
    u, s, vh = np.linalg.svd(np.eye(a.shape[0]) - d @ a @ d)

    k = min(u.shape[1], k)
    v = u[:, u.shape[1] - k:]

    kmeans = KMeans(n_clusters=k, n_init="auto", random_state=random_state).fit(v)
    return kmeans.labels_, v


def extract_max_value(h: np.ndarray):
    """ Select clusters based on maximum value in feature matrix H for every sample/row

    Args:
        h (Numpy.ndarray): feature matrix from optimization algorithm run of shape (n,k), where 'n' is a number of nodes
            and 'k' is a number of clusters

    Returns:
        one dimensional array containing clusters ids for every node

    """
    return np.argmax(h, axis=1)


def extract_spectral(h: np.ndarray, assign_labels: str = "kmeans", n_neighbors: int = 10, n_clusters: int = None,
                     random_state: int = None):
    """ Select clusters using spectral clustering of feature matrix H

    Args:
        h (Numpy.ndarray): feature matrix from optimization algorithm run of shape (n,k), where 'n' is a number of \
            nodes and 'k' is a number of clusters
        assign_labels : {'kmeans', 'discretize'}, strategy to use to assign labels in the embedding space
        n_neighbors (int): number of neighbors to use when constructing the affinity matrix
        n_clusters (int): number of clusters, if not set use number of columns of 'a'

    Returns:
        one dimensional array containing clusters ids for every node

    """
    if not n_clusters:
        n_clusters = h.shape[1]
    if not n_neighbors:
        n_neighbors = 10
    args = {"n_clusters": n_clusters, "assign_labels": assign_labels, 'n_init': 100,
            "affinity": 'nearest_neighbors', 'n_neighbors': n_neighbors, "random_state": random_state}

    with warnings.catch_warnings():
        warnings.filterwarnings('error', message="Graph is not fully connected")
        clustering = None

        while n_neighbors <= h.shape[0]:
            try:
                clustering = SpectralClustering(**args).fit(h)
                break
            except UserWarning:
                args['n_neighbors'] += 1

        if not clustering:
            clustering = SpectralClustering(**args).fit(h)

    return clustering.labels_


def purity(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure representing percentage of total number of nodes classified correctly """
    assert cl.shape == org.shape

    acc = 0
    for label in np.unique(cl):
        labels = {}
        for node in range(len(org)):
            if cl[node] == label:
                if org[node] not in labels.keys():
                    labels[org[node]] = 0
                labels[org[node]] += 1
        acc += max(labels.values()) if labels.keys() else 0
    return acc / len(org)


def adjusted_rand_index(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure calculated by considering all pairs of samples and counting pairs
        that are assigned in the same or different clusters in the predicted and true clusterings """
    assert cl.shape == org.shape

    return adjusted_rand_score(org, cl)  # (labels_true, labels_pred)


def normalized_mutual_information(cl: np.ndarray, org: np.ndarray):
    """ Clustering accuracy measure, which takes into account mutual information between two clusterings and entropy
        of each cluster """
    assert cl.shape == org.shape

    return mutual_info_score(org, cl) / (abs(entropy(cl) + entropy(org)) / 2)


@docstring_formatter(metrics=CLUSTER_METRICS)
def check_accuracy(cl: np.ndarray, org: np.ndarray, method="purity"):
    """ Check clustering accuracy

    Args:
        cl (Numpy.ndarray): one dimensional array containing computed clusters ids for every node
        org (Numpy.ndarray): one dimensional array containing true classes ids for every node
        method (str): accuracy assessment function from {metrics}

    """
    methods = {"NMI": normalized_mutual_information, "purity": purity, "ARI": adjusted_rand_index}
    assert all([m in methods.keys() for m in CLUSTER_METRICS])
    if method not in CLUSTER_METRICS:
        raise ValueError("Could not find selected method")

    if cl.shape != org.shape or cl.ndim != 1 or org.ndim != 1:
        raise ValueError("Incorrect dimensions of arrays")

    return methods[method](cl, org)


def is_standardized(a: np.ndarray, axis: int = 1, atol: float = 1e-3):
    """ Check if matrix values are standardized (have mean equal 0 and standard deviation equal 1)

    Args:
        a (Numpy.ndarray):feature matrix
        axis: either 0 (column-wise standardization) or 1 (row-wise standardization)
        atol (float): absolute tolerance

    Returns:
        is_standard (bool): True if data is standardized
        mean (float): maximum and minimum mean of columns/rows
        std (float): maximum and minimum standard deviation of columns/rows

    """
    if axis not in [0, 1]:
        raise ValueError("Incorrect value of axis, expected either 0 or 1, got {} instead".format(axis))

    mean = np.nanmean(a, axis=axis)
    std = np.nanstd(a, axis=axis)
    return np.allclose(mean, 0, atol=atol) and np.allclose(std, 1, atol=atol), (np.nanmin(mean), np.nanmax(mean)), \
           (np.nanmin(std), np.nanmax(std))


def filter_features_and_samples(data: pd.DataFrame, drop_features: float = 0.1, drop_samples: float = 0.1,
                                verbose = False):
    """ Filter data frame features and samples

    Args:
        data (pandas.DataFrame): data frame (with samples in columns and features in rows)
        drop_features (float): if percentage of missing values for feature exceeds this value, remove this feature
        drop_samples (float): if percentage of missing values for sample (that remains after feature dropping) \
            exceeds this value, remove this sample

    Returns:
        filtered data frame
    """
    # check arguments
    if drop_features < 0 or drop_features >= 1:
        raise ValueError("Incorrect value od 'drop_feature', expected value in range [0,1)")
    if drop_samples < 0 or drop_samples >= 1:
        raise ValueError("Incorrect value od 'drop_samples', expected value in range [0,1)")

    before = data.shape
    # drop features if needed
    nans = pd.isna(data).values
    data.drop(data.index[np.sum(nans, axis=1) / nans.shape[1] > drop_features], axis=0, inplace=True)

    # drop samples if needed
    nans = pd.isna(data).values
    data.drop(data.columns[np.sum(nans, axis=0) / nans.shape[0] > drop_samples], axis=1, inplace=True)

    if verbose:
        print("Number of dropped rows/features: {}".format(before[0] - data.shape[0]))
        print("Number of dropped columns/samples: {}".format(before[1] - data.shape[1]))
        print("Data shape: {}".format(data.values.shape))

    return data


def load_data_text(file_path: str, sample_names: int = None, feature_names: int = None, drop_features: float = 0.1,
                   drop_samples: float = 0.1):
    """ Loads data from text file (with samples in columns and features in rows) into pandas.DataFrame

    Args:
        file_path (str): path to the tab delimited .txt file
        sample_names (int): index of row with sample names
        feature_names (int): index of column with feature names
        drop_features (float): if percentage of missing values for feature exceeds this value, remove this feature
        drop_samples (float): if percentage of missing values for sample (that remains after feature dropping) \
            exceeds this value, remove this sample
    Returns:
        data (pandas.DataFrame): data frame loaded from file, with missing values removed

    """
    if not os.path.exists(file_path):
        raise FileNotFoundError("Data file not found")

    data = pd.read_csv(file_path, delimiter=r'\s+', header=sample_names, index_col=feature_names)
    if data.empty or data.values.shape == (1, 1):
        raise ValueError('File cannot be read correctly, file is not tab delimited or is corrupted')
    elif data.values.dtype == np.object:
        raise ValueError("File contains some non-numerical values other than 'NA'")

    return filter_features_and_samples(data, drop_features=drop_features, drop_samples=drop_samples)
