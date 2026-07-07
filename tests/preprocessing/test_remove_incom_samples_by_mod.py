import numpy as np
import pandas as pd
import pytest

from imml import deepmodule_installed
from imml.preprocessing import RemoveIncomSamplesByMod, remove_incom_samples_by_mod

if deepmodule_installed:
    import torch


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    X3.loc[5, 8:] = np.nan
    output = [X1, X2, X3], [X1.values, X2.values, X3.values]
    if deepmodule_installed:
        output = (*output, [torch.from_numpy(X) for X in output[1]])
    return output


def test_remove_incom_samples_by_mod_class(sample_data):
    for Xs in sample_data:
        transformer = RemoveIncomSamplesByMod()
        transformed_Xs = transformer.fit_transform(Xs)
        expected_values = [18, 19, 19]
        for transformed_X, expected_value in zip(transformed_Xs, expected_values):
            assert len(transformed_X) == expected_value


def test_remove_incom_samples_by_mod_function(sample_data):
    for Xs in sample_data:
        transformed_Xs = remove_incom_samples_by_mod(Xs)
        expected_values = [18, 19, 19]
        for transformed_X, expected_value in zip(transformed_Xs, expected_values):
            assert len(transformed_X) == expected_value


if __name__ == "__main__":
    pytest.main()