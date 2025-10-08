# License: BSD-3-Clause

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer

from ..utils import check_Xs
from ..explore import get_samples


class DropMod(FunctionTransformer):
    r"""
    A transformer that drops a specified modality from a multi-modal dataset. Apply `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_
    (from `Scikit-learn`) with `drop_mod` as a function.

    Parameters
    ----------
    X_idx : int, default=0
        The index of the modality to drop from the input data.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import DropMod
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = DropMod(X_idx = 1)
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self, X_idx: int = 0):
        self.X_idx = X_idx
        super().__init__(drop_mod, kw_args={"X_idx": X_idx})


class ConcatenateMods(FunctionTransformer):
    r"""
    A transformer that concatenates all modalities from a multi-modal dataset. Apply `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_
    (from `Scikit-learn`) with `concatenate_mods` as a function.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import ConcatenateMods
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = ConcatenateMods()
    >>> transformer.fit_transform(Xs)
    """

    def __init__(self):
        super().__init__(concatenate_mods)


class SingleMod(FunctionTransformer):
    r"""
    Transformer that selects a single modality from multi-modal data. Apply `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_ (from `Scikit-learn`)
    with `single_mod` as a function.

    Parameters
    ----------
    X_idx : int, default=0
        The index of the modality to select from the input data.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import SingleMod
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = SingleMod(X_idx = 1)
    >>> transformer.fit_transform(Xs)
    """
    
    def __init__(self, X_idx : int = 0):
        self.X_idx = X_idx
        super().__init__(single_mod, kw_args = {"X_idx": X_idx})


class AddMissingMods(FunctionTransformer):
    r"""
    Transformer to add missing samples in each modality, in a way that all the modalities will have the same samples.
     Apply `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_ (from `Scikit-learn`) with `add_missing_mods` as a function.

    This transformer is applied on individual modalities, so for applying in a multi-modal dataset, we recommend
    to use it with `MultiModTransformer`.

    Parameters
    ----------
    samples : array-like  (n_samples,)
        pd.Index with all samples

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import AddMissingMods
    >>> from imml.explore import get_samples
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> samples = get_samples(Xs= Xs)
    >>> transformer = AddMissingMods(samples= samples)
    >>> transformer.fit_transform(Xs)

    """

    def __init__(self, samples: pd.Index):
        self.samples = samples
        super().__init__(add_missing_mods, kw_args={"samples": samples})


class SortData(FunctionTransformer):
    r"""
    Transformer that establish and assess the order of the incomplete multi-modal dataset. Apply
    `FunctionTransformer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html>`_ (from `Scikit-learn`) with sort_data as a function.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import SortData
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> transformer = SortData()
    >>> transformer.fit_transform(Xs)

    """

    def __init__(self):
        super().__init__(sort_data)


def concatenate_mods(Xs: list):
    r"""
    A function that concatenate all features from a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mods.

    Returns
    -------
    transformed_Xs : array-like, shape (n_samples, n_features)
        The transformed dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import concatenate_mods
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> concatenate_mods(Xs=Xs)
    """

    if isinstance(Xs[0], pd.DataFrame):
        transformed_X = pd.concat(Xs, axis= 1)
    elif isinstance(Xs[0], np.ndarray):
        transformed_X = np.concatenate(Xs, axis= 1)
    return transformed_X


def drop_mod(Xs, X_idx : int = 0):
    r"""
    A function that drops a specified modality from a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mods.
    X_idx : int, default=0
        The index of the mod to drop from the input data.

    Returns
    -------
    transformed_X : list of array-likes objects (n_samples, n_features_i)
        The transformed multi-modal dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import drop_mod
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> drop_mod(Xs=Xs, X_idx = 1)
    """
    if X_idx >= len(Xs):
        raise ValueError("X_idx out of range. Should be between 0 and n_mods - 1")
    Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
    transformed_Xs = Xs[:X_idx] + Xs[X_idx+1 :]
    return transformed_Xs


def single_mod(Xs, X_idx : int = 0):
    r"""
    A function that selects a specified modality from a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mods.
    X_idx : int, default=0
        The index of the mod to select from the input data.

    Returns
    -------
    transformed_Xs : array-like, shape (n_samples, n_features)
        The transformed dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import single_mod
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> single_mod(Xs=Xs, X_idx = 1)
  """
    if X_idx >= len(Xs):
        raise ValueError("X_idx out of range. Should be between 0 and n_mods - 1")
    Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
    transformed_X = Xs[X_idx]
    return transformed_X


def add_missing_mods(Xs, samples):
    r"""
    Add missing samples in each modality, in a way that all the modalities will have the same samples.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different mods.

    Returns
    -------
    transformed_X : list of array-likes objects (n_samples, n_features_i)
        The transformed multi-modal dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import add_missing_mods
    >>> from imml.explore import get_samples
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> samples = get_samples(Xs= Xs)
    >>> add_missing_mods(Xs, samples= samples)
  """
    pandas_format = isinstance(Xs[0], pd.DataFrame)
    if pandas_format:
        transformed_Xs = [X.T for X in Xs]
    else:
        transformed_Xs = [pd.DataFrame(X).T for X in Xs]
    for i,transformed_X in enumerate(transformed_Xs):
        transformed_X[samples.difference(transformed_X.index)] = np.nan
        transformed_Xs[i] = transformed_X.T.loc[samples]
    if not pandas_format:
        transformed_Xs = [transformed_X.values for transformed_X in transformed_Xs]
    return transformed_Xs


def sort_data(Xs: list):
    r"""
    A function that establish and assess the order of the incomplete multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    transformed_X : list of array-likes objects (n_samples, n_features_i)
        The transformed multi-modal dataset.

    Example
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.preprocessing import sort_data
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> sort_data(Xs=Xs)
    """

    Xs = check_Xs(Xs, ensure_all_finite='allow-nan')
    if not isinstance(Xs[0], pd.DataFrame):
        Xs = [pd.DataFrame(X) for X in Xs]
    samples = get_samples(Xs=Xs)
    transformed_X = [X.loc[samples.intersection(X.index)] for X in Xs]
    return transformed_X

