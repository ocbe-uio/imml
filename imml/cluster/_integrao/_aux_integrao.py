# License: BSD-3-Clause

from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.utils import check_symmetric


def data_indexing(matrices):
    """
    Performs data indexing on input expression matrices

    Parameters
    ----------
    matrices : (M, N) array_like
        Input expression matrices, with gene/feature in columns and sample in row.

    Returns
    -------
    matrices_pure: Expression matrices without the first column and first row
    dict_commonSample: dictionaries that give you the common samples between 2 views
    dict_uniqueSample: dictionaries that give you the unique samples between 2 views
    original_order: the original order of samples for each view
    """

    if len(matrices) < 1:
        print("Input nothing, return nothing")
        return None

    original_order = [0] * (len(matrices))
    dict_original_order = {}
    dict_commonSample = {}
    dict_uniqueSample = {}
    dict_commonSampleIndex = {}
    dict_sampleToIndexs = defaultdict(list)

    for i in range(0, len(matrices)):
        original_order[i] = list(matrices[i].index)
        dict_original_order[i] = original_order[i]
        for sample in original_order[i]:
            dict_sampleToIndexs[sample].append(
                (i, np.argwhere(matrices[i].index == sample).squeeze().tolist())
            )

    for i in range(0, len(original_order)):
        for j in range(i + 1, len(original_order)):
            commonList = list(set(original_order[i]).intersection(original_order[j]))
            dict_commonSample.update(
                dict_commonSample.fromkeys([(i, j), (j, i)], commonList)
            )
            dict_commonSampleIndex[(i, j)] = [
                np.argwhere(matrices[i].index == x).squeeze().tolist()
                for x in commonList
            ]
            dict_commonSampleIndex[(j, i)] = [
                np.argwhere(matrices[j].index == x).squeeze().tolist()
                for x in commonList
            ]

            dict_uniqueSample[(i, j)] = list(
                set(original_order[i]).symmetric_difference(commonList)
            )
            dict_uniqueSample[(j, i)] = list(
                set(original_order[j]).symmetric_difference(commonList)
            )

    return (
        dict_commonSample,
        dict_commonSampleIndex,
        dict_sampleToIndexs,
        dict_uniqueSample,
        original_order,
        dict_original_order,
    )


def dist2(X, C):
    """
    Description: Computes the Euclidean distances between all pairs of data point given

    Usage: dist2(X, C)
    X: A data matrix where each row is a different data point
    C: A data matrix where each row is a different data point. If this matrix is the same as X,
    pairwise distances for all data points in X are computed.

    Return: Returns an N x M matrix where N is the number of rows in X and M is the number of rows in C.

    Author: Dr. Anna Goldenberg, Bo Wang, Aziz Mezlini, Feyyaz Demir
    Python Version Rewrite: Rex Ma

    Examples:
        # Data1 is of size n x d_1, where n is the number of patients, d_1 is the number of genes,
        # Data2 is of size n x d_2, where n is the number of patients, d_2 is the number of methylation
        Dist1 = dist2(Data1, Data1)
        Dist2 = dist2(Data2, Data2)
    """

    ndata = X.shape[0]
    ncentres = C.shape[0]

    sumsqX = np.sum(X * X, axis=1)
    sumsqC = np.sum(C * C, axis=1)

    XC = 2 * (np.matmul(X, np.transpose(C)))

    res = (
        np.transpose(np.reshape(np.tile(sumsqX, ncentres), (ncentres, ndata)))
        + np.reshape(np.tile(sumsqC, ndata), (ndata, ncentres))
        - XC
    )

    return res


def _stable_normalized_pd(W):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)

    W_np = W.values
    np.fill_diagonal(W_np, 0.5)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns)

    W = check_symmetric(W, raise_warning=False)

    return W


def _scaling_normalized_pd(W, ratio):
    """
    Adds `alpha` to the diagonal of pandas dataframe `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = (W / rowSum) * 0.5 * ratio

    W_np = W.values
    np.fill_diagonal(W_np, 1-0.5*ratio)
    W = pd.DataFrame(W_np, index=W.index, columns=W.columns)

    W = check_symmetric(W, raise_warning=False)

    return W


def p_preprocess(P):
    # Make sure P-values are set properly
    np.fill_diagonal(P, 0)  # set diagonal to zero
    P = P + np.transpose(P)  # symmetrize P-values
    P = P / np.sum(P)  # make sure P-values sum to one
    P = P * 4.0  # early exaggeration
    P = np.maximum(P, 1e-12)
    return P


def _stable_normalized(W):
    """
    Adds `alpha` to the diagonal of `W`

    Parameters
    ----------
    W : (N, N) array_like
        Similarity array from SNF

    Returns
    -------
    W : (N, N) np.ndarray
        Stable-normalized similiarity array
    """

    # add `alpha` to the diagonal and symmetrize `W`
    rowSum = np.sum(W, 1) - np.diag(W)
    rowSum[rowSum == 0] = 1

    W = W / (2 * rowSum)
    np.fill_diagonal(W, 0.5)
    W = check_symmetric(W, raise_warning=False)

    return W