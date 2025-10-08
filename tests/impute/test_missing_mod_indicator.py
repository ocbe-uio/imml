import pytest
import numpy as np
import pandas as pd
from imml.impute import MissingModIndicator, get_missing_mod_indicator


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
    return Xs_pandas, Xs_numpy, observed_mod_indicator

def test_get_missing_mod_indicator(sample_data):
    observed_mod_indicator = sample_data[-1]
    for Xs in sample_data[:2]:
        indicator = get_missing_mod_indicator(Xs)
        np.equal(indicator, observed_mod_indicator)

def test_missing_mod_indicator_class(sample_data):
    observed_mod_indicator = sample_data[-1]
    for Xs in sample_data[:2]:
        transformer = MissingModIndicator()
        indicator = transformer.fit_transform(Xs)
        np.equal(indicator, observed_mod_indicator)
