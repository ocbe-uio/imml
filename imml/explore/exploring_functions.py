# License: BSD-3-Clause

from typing import Union
import numpy as np
import pandas as pd

from ..utils import check_Xs
from ..impute import get_observed_mod_indicator


def get_n_mods(Xs: list) -> int:
    r"""
    Get the number of modalities of a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    n_mods: int
        Number of modalities.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_n_mods
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> get_n_mods(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_mods = len(Xs)
    return n_mods


def get_n_samples_by_mod(Xs: list) -> int:
    r"""
    Get the number of samples in each modality.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    n_samples_by_mod: pd.Series
        Number of samples in each modality.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_n_samples_by_mod
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_n_samples_by_mod(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_samples_by_mod = get_observed_mod_indicator(Xs)
    n_samples_by_mod = n_samples_by_mod.sum(axis=0)
    return n_samples_by_mod


def get_com_samples(Xs: list) -> pd.Index:
    r"""
    Get name (index) of complete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    samples: pd.Index
        Sample names with full data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_com_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_com_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    samples = get_observed_mod_indicator(Xs)
    if not isinstance(samples, pd.DataFrame):
        samples = pd.DataFrame(samples)
    samples = samples[samples.all(1)].index
    return samples


def get_incom_samples(Xs: list) -> pd.Index:
    r"""
    Get name (index) of incomplete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    samples: pd.Index
        Sample names with incomplete data.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_incom_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_incom_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    samples = get_observed_mod_indicator(Xs)
    if not isinstance(samples, pd.DataFrame):
        samples = pd.DataFrame(samples)
    samples = samples[~samples.all(1)].index
    return samples


def get_samples(Xs: list) -> pd.Index:
    r"""
    Get name (index) of samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples_i, n_features_i)

        A list of different modalities.

    Returns
    -------
    samples: pd.Index (n_samples,)
        Sample names.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    if not isinstance(Xs[0], pd.DataFrame):
        Xs = [pd.DataFrame(X) for X in Xs]
    samples = [X.index.to_list() for X in Xs]
    samples = [x for xs in samples for x in xs]
    samples = pd.Index(sorted(set(samples), key=samples.index))
    return samples


def get_samples_by_mod(Xs: list, return_as_list: bool = True) -> Union[list, dict]:
    r"""
    Get the samples for each modality in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    return_as_list : bool, default=True
        If True, the function will return a list; a dict otherwise.

    Returns
    -------
    samples: list or dict of pd.Index
        If list, each element in the list is the sample names for each modality. If dict, keys are the modalities and the
        values are the sample names.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_samples_by_mod
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_samples_by_mod(Xs = Xs)
    """
    observed_mod_indicator = get_observed_mod_indicator(Xs)
    if not isinstance(observed_mod_indicator, pd.DataFrame):
        observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
    if return_as_list:
        samples = [mod_profile[mod_profile].index for X_idx, mod_profile in observed_mod_indicator.items()]
    else:
        samples = {X_idx: mod_profile[mod_profile].index for X_idx, mod_profile in observed_mod_indicator.items()}
    return samples


def get_missing_samples_by_mod(Xs: list, return_as_list: bool = True) -> Union[list, dict]:
    r"""
    Get the samples not present in each modality in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    return_as_list : bool, default=True
        If list, each element in the list is the sample names for each modality. If dict, keys are the modalities and the
        values are the sample names.

    Returns
    -------
    samples: dict of pd.Index or list of pd.Index.
        Dictionary or list of missing samples for each modality.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_missing_samples_by_mod
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_missing_samples_by_mod(Xs = Xs)
    """

    observed_mod_indicator = get_observed_mod_indicator(Xs)
    if not isinstance(observed_mod_indicator, pd.DataFrame):
        observed_mod_indicator = pd.DataFrame(observed_mod_indicator)
    if return_as_list:
        samples = [mod_profile[mod_profile == False].index.to_list()
                   for X_idx, mod_profile in observed_mod_indicator.items()]
    else:
        samples = {X_idx: mod_profile[mod_profile == False].index.to_list()
                   for X_idx, mod_profile in observed_mod_indicator.items()}
    return samples


def get_n_com_samples(Xs: list) -> int:
    r"""
    Get the number of complete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    n_samples: int
        number of complete samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_n_com_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_n_com_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_samples = len(get_com_samples(Xs=Xs))
    return n_samples


def get_n_incom_samples(Xs: list) -> int:
    r"""
    Get the number of incomplete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    n_samples: int
        number of incomplete samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_n_incom_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_n_incom_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_samples = len(get_incom_samples(Xs=Xs))
    return n_samples


def get_pct_com_samples(Xs: list) -> float:
    r"""
    Get the percentage of complete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    percentage_samples: float
        percentage of complete samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_pct_com_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_pct_com_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    percentage_samples = get_n_com_samples(Xs=Xs) / len(Xs[0]) * 100
    return percentage_samples


def get_pct_incom_samples(Xs: list) -> float:
    r"""
    Get the percentage of incomplete samples in a multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.

    Returns
    -------
    percentage_samples: float
        percentage of incomplete samples.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_pct_incom_samples
    >>> from imml.ampute import Amputer
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> Xs = Amputer(p=0.2, mechanism="mcar", random_state=42).fit_transform(Xs)
    >>> get_pct_incom_samples(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    percentage_samples = get_n_incom_samples(Xs=Xs) / len(Xs[0]) * 100
    return percentage_samples


def get_summary(Xs: list, modalities: list = None, one_row: bool = False, compute_pct: bool = True,
                return_df: bool = False) -> Union[dict, pd.DataFrame]:
    r"""
    Get a summary of an incomplete multi-modal dataset.

    Parameters
    ----------
    Xs : list of array-likes objects
        - Xs length: n_mods
        - Xs[i] shape: (n_samples, n_features_i)

        A list of different modalities.
    modalities : list, default=None
        Name of each modality. By default, it will be set to the modality index. Only applicable when one_row is False.
    one_row : bool, default=False
        If True, return a one-row summary of the dataset. If False, each row will correspond to a modality.
    compute_pct : bool, default=True
        If True, compute percent of each value.
    return_df : bool, default=False
        If True, it will return a pd.DataFrame. It returns a dict otherwise.

    Returns
    -------
    summary: dict or pd.DataFrame
        Summary of an incomplete multi-modal dataset.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from imml.explore import get_summary
    >>> Xs = [pd.DataFrame(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> get_summary(Xs = Xs)
    """
    Xs = check_Xs(Xs=Xs, ensure_all_finite="allow-nan")
    n_samples = len(Xs[0])
    if one_row:
        summary = {
            "Complete samples": get_n_com_samples(Xs),
            "Incomplete samples": get_n_incom_samples(Xs),
            "Observed samples per modality": [n_samples - len(X_id) for X_id in
                                              get_missing_samples_by_mod(Xs)],
            "Missing samples per modality": [len(X_id) for X_id in
                                             get_missing_samples_by_mod(Xs)],
            "% Observed samples per modality": [round((n_samples - len(X_id)) / n_samples * 100) for X_id in
                                                get_missing_samples_by_mod(Xs)],
            "% Missing samples per modality": [round(len(X_id) / n_samples * 100) for X_id in
                                               get_missing_samples_by_mod(Xs)],
        }
        if compute_pct:
            summary = {
                **summary,
                "% Observed samples per modality": [round((n_samples - len(X_id)) / n_samples * 100) for X_id in
                                                    get_missing_samples_by_mod(Xs)],
                "% Missing samples per modality": [round(len(X_id) / n_samples * 100) for X_id in
                                                   get_missing_samples_by_mod(Xs)],
            }
        if return_df:
            summary = pd.DataFrame.from_dict(summary, orient="index").T

    else:
        if modalities is None:
            modalities = list(range(len(Xs)))
        c_samples, m_samples, i_samples = [], [], []
        summary = {}
        for X, mod in zip(Xs, modalities):
            mod_c_samples = pd.DataFrame(X)[np.isfinite(X).all(axis=1)]
            mod_m_samples = pd.DataFrame(X)[np.isnan(X).all(axis=1)]
            mod_i_samples = pd.DataFrame(X)[np.isnan(X).any(axis=1)]
            summary[mod] = {
                "Complete samples": len(mod_c_samples),
                "Missing samples": len(mod_m_samples),
                "Incomplete samples": len(mod_i_samples),
            }
            c_samples.append(mod_c_samples.index.to_series())
            m_samples.append(mod_m_samples.index.to_series())
            i_samples.append(mod_i_samples.index.to_series())
        summary["Total"] = {
            "Complete samples": (pd.concat(c_samples).value_counts() == len(Xs)).sum(),
            "Missing samples": (pd.concat(m_samples).value_counts() > 0).sum(),
            "Incomplete samples": (pd.concat(i_samples).value_counts() > 0).sum(),
        }
        if compute_pct:
            for mod in summary.keys():
                for k in list(summary[mod].keys()):
                    summary[mod][f"% {k}"] = summary[mod][k] / n_samples * 100
        if return_df:
            summary = pd.DataFrame.from_dict(summary, orient="index")
    return summary