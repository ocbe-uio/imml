import numpy as np
import pandas as pd
import pytest

from imml.ampute import RemoveMods, remove_mods
from imml import deepmodule_installed

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    output = [X1, X2, X3], [X1.values, X2.values, X3.values]
    if deepmodule_installed:
        output = (*output, [torch.from_numpy(X) for X in output[1]])
    return output

def test_remove_mods_transformer(sample_data):
    for Xs in sample_data:
        observed_mod_indicator = np.ones((len(Xs[0]), len(Xs)))
        observed_mod_indicator[0, :3] = 0
        transformer = RemoveMods(observed_mod_indicator=observed_mod_indicator)
        transformed_Xs = transformer.fit_transform(Xs)
        assert len(Xs) == len(transformed_Xs)
        assert np.isnan(transformed_Xs[0]).all(axis=1).sum() == 1


def test_remove_mods_function(sample_data):
    for Xs in sample_data:
        observed_mod_indicator = np.ones((len(Xs[0]), len(Xs)))
        observed_mod_indicator[0, :3] = 0
        transformed_Xs = remove_mods(Xs, observed_mod_indicator=observed_mod_indicator)
        assert np.isnan(transformed_Xs[0]).all(axis=1).sum() == 1

if __name__ == "__main__":
    pytest.main()