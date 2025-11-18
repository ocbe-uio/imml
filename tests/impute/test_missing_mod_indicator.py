import pytest
import numpy as np
import pandas as pd
from imml.impute import MissingModIndicator, get_missing_mod_indicator
from imml import deepmodule_installed

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    observed_mod_indicator = pd.DataFrame({
        0: [False, False, True, False, True],
        1: [False, True, False, False, False]
    })
    observed_mod_indicator = observed_mod_indicator.values
    output = (Xs_pandas, Xs_numpy)
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        observed_mod_indicator_torch = torch.from_numpy(observed_mod_indicator).bool()
        output = output + (Xs_torch, observed_mod_indicator, observed_mod_indicator_torch,)
    else:
        output = output + (observed_mod_indicator,)
    return output


def test_get_missing_mod_indicator(sample_data):
    if deepmodule_installed:
        observed_mod_indicator = sample_data[3]
        sample_data = sample_data[:3]
    else:
        observed_mod_indicator = sample_data[2]
        sample_data = sample_data[:2]
    for Xs in sample_data:
        indicator = get_missing_mod_indicator(Xs)
        np.equal(indicator, observed_mod_indicator)


def test_missing_mod_indicator_class(sample_data):
    if deepmodule_installed:
        observed_mod_indicator = sample_data[3]
        sample_data = sample_data[:3]
    else:
        observed_mod_indicator = sample_data[2]
        sample_data = sample_data[:2]
    for Xs in sample_data:
        transformer = MissingModIndicator()
        indicator = transformer.fit_transform(Xs)
        np.equal(indicator, observed_mod_indicator)
