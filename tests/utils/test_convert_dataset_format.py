import pytest
import numpy as np
import pandas as pd

from imml.utils import convert_dataset_format


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((5, 5))
    X = pd.DataFrame(X)
    X1, X2 = X.iloc[:, :3].copy(), X.iloc[:, 3:].copy()
    X1.loc[[2,4], :] = np.nan
    X2.loc[1, :] = np.nan
    Xs_pandas, Xs_numpy = [X1, X2], [X1.values, X2.values]
    observed_mod_indicator = pd.DataFrame({
        0: [True, True, False, True, False],
        1: [True, False, True, True, True]
    })
    observed_mod_indicator = observed_mod_indicator.values
    return Xs_pandas, Xs_numpy, observed_mod_indicator


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid Xs."):
        convert_dataset_format(Xs="invalid_input")


def test_convert_dataset_format_dict(sample_data):
    for Xs in sample_data[:2]:
        Xs_dict = convert_dataset_format(Xs)
        assert isinstance(Xs_dict, dict)
        assert len(Xs_dict) == len(Xs)
        assert all(isinstance(k, int) for k in Xs_dict.keys())


def test_convert_dataset_format_list(sample_data):
    for Xs in sample_data[:2]:
        Xs = {i:X for i,X in enumerate(Xs)}
        Xs_list = convert_dataset_format(Xs)
        assert isinstance(Xs_list, list)
        assert len(Xs_list) == len(Xs)


if __name__ == "__main__":
    pytest.main()
