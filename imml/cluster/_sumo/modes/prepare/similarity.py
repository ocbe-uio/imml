# License: BSD-3-Clause

from math import sqrt
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
import numpy as np

from ...._sumo.constants import SIMILARITY_METHODS, PREPARE_DEFAULTS, CORR_METHODS
from ...._sumo.utils import docstring_formatter


def euclidean_dist(a: np.ndarray, b: np.ndarray, missing: float):
    """ Calculate euclidean distance between two vectors of continuous variables """
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    eps = np.spacing(1)
    vec = np.power(a - b, 2)
    common = ~np.isnan(vec)  # common / not missing values
    dist = (sqrt(np.sum(vec[common])) + eps) / (np.sum(common) + eps) if np.sum(common) > threshold else np.nan
    return dist


def cosine_sim(a: np.ndarray, b: np.ndarray, missing: float):
    """ Calculate cosine similarity between two vectors"""
    assert a.shape == b.shape
    threshold = a.shape[0] * missing
    values = ~np.logical_or(np.isnan(b), np.isnan(a))  # find missing values in either of vectors
    avec = a[values]
    bvec = b[values]

    sim = 1 - cosine(avec, bvec)
    return sim if avec.size > threshold else np.nan


def correlation(a: np.ndarray, b: np.ndarray, missing: float, method="pearson"):
    """ Calculate correlation similarity between two vectors"""
    assert a.shape == b.shape
    assert method in CORR_METHODS
    threshold = a.shape[0] * missing
    values = ~np.logical_or(np.isnan(b), np.isnan(a))  # find missing values in either of vectors
    avec = a[values]
    bvec = b[values]

    cor_func = pearsonr if method == "pearson" else spearmanr
    if avec.size > threshold:
        return cor_func(avec, bvec)[0]
    else:
        return np.nan


@docstring_formatter(missing=PREPARE_DEFAULTS['missing'][0], sim_methods=SIMILARITY_METHODS)
def feature_to_adjacency(f: np.ndarray, missing: float = PREPARE_DEFAULTS['missing'][0],
                         method: str = PREPARE_DEFAULTS['method'][0], n: float = None, alpha: float = None):
    """ Generate similarity matrix from genomic assay

    Args:
        f (Numpy.ndarray): Feature matrix (n x k, where 'n' - samples, 'k' - measurements)
        missing (float): acceptable fraction of values for assessment of distance/similarity between two samples \
            (default of {missing}, means that up to 90 % of missing values is acceptable)
        method (str): similarity method selected from: {sim_methods}
        n (float): parameter of euclidean similarity method, fraction of nearest neighbours of sample
        alpha (float): parameter of euclidean similarity method, RBF kernel hyperparameter

    Returns:
        sim (Numpy.ndarray): symmetric matrix describing similarity between samples (n x n)
    """
    if method not in SIMILARITY_METHODS:
        raise ValueError("Incorrect name of similarity method")

    if method == "euclidean":
        if n is None or alpha is None:
            raise ValueError("Euclidean similarity selected, but not all arguments supplied")
        sim = feature_rbf_similarity(f, missing=missing, n=n, alpha=alpha)
    else:
        # filter missing samples
        samples = np.array([i for i in range(f.shape[0]) if not np.all(np.isnan(f[i, :]))])

        sim = np.zeros((f.shape[0], f.shape[0]))
        sim[:] = np.nan

        for i in [sample for sample in range(f.shape[0]) if sample in samples]:
            for j in [sample for sample in range(i, f.shape[0]) if sample in samples]:
                if method == "cosine":
                    sim[i, j] = cosine_sim(f[i, :], f[j, :], missing=missing)
                else:
                    sim[i, j] = correlation(f[i, :], f[j, :], missing=missing, method=method)
                if i != j:
                    sim[j, i] = sim[i, j]

        vals = sim[~np.isnan(sim)]
        # remove negative values
        vals[vals < 0] = 0
        sim[~np.isnan(sim)] = vals

    return sim


def feature_rbf_similarity(f: np.ndarray, missing: float = PREPARE_DEFAULTS['missing'][0],
                           n: float = PREPARE_DEFAULTS['k'], alpha: float = PREPARE_DEFAULTS['alpha'],
                           distance=euclidean_dist):
    """ Generate similarity matrix using RBF kernel and supplied distance function

    Args:
        f (Numpy.ndarray): Feature matrix (n x k, where 'n' - samples, 'k' - measurements)
        n (float): fraction of nearest neighbours to use for samples similarity calculation
        missing (float): acceptable fraction of values for assessment of distance/similarity between two samples
        alpha (float): hyperparameter of RBF kernel
        distance: distance function accepting two vectors and missing parameter (default of Euclidean distance)

    Returns:
        w (Numpy.ndarray): symmetric matrix describing similarity between samples (n x n)

    """
    if alpha <= 0:
        raise ValueError("Incorrect value of hyperparameter")
    # NOTE: for every method distance is calculated only over features that are not missing for every pair of samples

    # filter missing samples
    samples = np.array([i for i in range(f.shape[0]) if not np.all(np.isnan(f[i, :]))])
    k = int(round(samples.shape[0] * n, 0))

    # euclidean distance matrix
    dist = np.zeros((f.shape[0], f.shape[0]))
    dist[:] = np.nan

    for i in [sample for sample in range(f.shape[0]) if sample in samples]:
        for j in [sample for sample in range(i, f.shape[0]) if sample in samples]:
            if i != j:
                dist[i, j] = distance(f[i, :], f[j, :], missing=missing)
                dist[j, i] = dist[i, j]
            else:
                dist[i, j] = 0

    # weighted edges matrix
    w = np.zeros((f.shape[0], f.shape[0]))
    w[:] = np.nan

    sample_dist = dist[samples[:, None], samples]  # select only available samples
    sample_w = np.zeros((samples.shape[0], samples.shape[0]))

    for i in range(samples.shape[0]):
        for j in range(i, samples.shape[0]):
            num = -1 * (sample_dist[i, j] ** 2)
            neighbours_i = np.sort(sample_dist[i, :])[1:k + 1]
            neighbours_j = np.sort(sample_dist[j, :])[1:k + 1]
            # np.nan means that it is not possible to calculate distance between this sample and any other in layer
            ni_mean = np.nanmean(neighbours_i) if neighbours_i[0] != np.nan else np.nan
            nj_mean = np.nanmean(neighbours_j) if neighbours_j[0] != np.nan else np.nan
            den = alpha * ni_mean * nj_mean
            if np.allclose(den, 0):
                raise ValueError(
                    "Average distance between sample and its all nearest neighbours is 0, please increase" +
                    " fraction of nearest neighbours used for similarity calculation ('-k' parameter)")
            if i != j:
                sample_w[i, j] = np.exp(num / den)
                sample_w[j, i] = sample_w[i, j]
            else:
                sample_w[i, j] = 1.

    w[samples[:, None], samples] = sample_w  # put missing samples back in

    return w
