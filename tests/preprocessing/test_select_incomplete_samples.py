import pytest
import numpy as np
import pandas as pd

from imml.preprocessing import SelectIncompleteSamples, select_incomplete_samples
from imml import deepmodule_installed

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
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    output = (Xs_pandas, Xs_numpy)
    if deepmodule_installed:
        Xs_torch = [torch.from_numpy(X) for X in Xs_numpy]
        output = output + (Xs_torch,)
    return output


def test_select_incomplete_samples_class(sample_data):
    for Xs in sample_data:
        transformer = SelectIncompleteSamples()
        transformed_Xs = transformer.fit_transform(Xs)
        type(transformed_Xs[0]) is type(Xs[0])
        expected_values = [4, 4, 4]
        for transformed, expected_value in zip(transformed_Xs, expected_values):
            np.equal(transformed, expected_value)


def test_select_incomplete_samples_function(sample_data):
    for Xs in sample_data:
        transformed_Xs = select_incomplete_samples(Xs)
        type(transformed_Xs[0]) is type(Xs[0])
        expected_values = [4, 4, 4]
        for transformed, expected_value in zip(transformed_Xs, expected_values):
            np.equal(transformed, expected_value)


def test_select_incomplete_samples_multiple_types(sample_data):
    X = np.random.default_rng(42).random((20, 10))
    X = pd.DataFrame(X)
    X1, X2, X3 = X.iloc[:, :3], X.iloc[:, 3:5], X.iloc[:, 5:]
    X1.loc[:,0] = "loren ipsum"
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    X3.loc[5, 8:] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2, X3], [X1.values, X2.values, X3.values]
    sample_data = (Xs_pandas, Xs_numpy)
    for Xs in sample_data:
        transformed_Xs = select_incomplete_samples(Xs)
        type(transformed_Xs[0]) is type(Xs[0])
        expected_values = [4, 4, 4]
        for transformed_X, expected_value in zip(transformed_Xs, expected_values):
            np.equal(transformed_X, expected_value)


if __name__ == "__main__":
    pytest.main()