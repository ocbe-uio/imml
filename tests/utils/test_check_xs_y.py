import numpy as np
import pandas as pd
import pytest

try:
    import torch
    deepmodule_installed = True
except ImportError:
    deepmodule_installed = False

from imml.utils import check_Xs_y


def test_valid_inputs():
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs_y([X1, X2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    df1 = pd.DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    df2 = pd.DataFrame([[5, 6], [7, 8]], columns=['A', 'B'])
    result = check_Xs_y([df1, df2])
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0].shape == (2, 2)
    assert result[1].shape == (2, 2)

    if deepmodule_installed:
        X1 = torch.from_numpy(np.array([[1, 2], [3, 4]]))
        X2 = torch.from_numpy(np.array([[1, 2], [3, 4]]))
        result = check_Xs_y([X1, X2])
        assert isinstance(result, list)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    # Test with arrays containing NaN values
    X1 = np.array([[1, np.nan], [3, 4]])
    X2 = np.array([[5, 6], [np.nan, 8]])
    result = check_Xs_y([X1, X2], ensure_all_finite='allow-nan')
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.isnan(result[0][0, 1])
    assert np.isnan(result[1][1, 0])


def test_invalid_inputs():
    with pytest.raises(ValueError, match="Invalid Xs. It must be a list"):
        check_Xs_y(123)
    with pytest.raises(ValueError, match="Invalid Xs. It must have at least two modalities"):
        check_Xs_y([])

    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    with pytest.raises(ValueError, match="Invalid Xs. Wrong number of modalities. Expected 3 but found 2"):
        check_Xs_y([X1, X2], enforce_modalities=3)

    with pytest.raises(ValueError, match="Invalid Xs. All modalities should have the same number of samples"):
        check_Xs_y([X1[:-1], X2])
    with pytest.raises(ValueError, match="Invalid Xs. All modalities should be the same data type"):
        check_Xs_y([X1, pd.DataFrame(X2)])


def test_optional_parameters():
    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs_y([X1, X2], copy=True)
    assert not np.may_share_memory(result[0], X1)
    assert not np.may_share_memory(result[1], X2)

    X1 = np.array([[1, np.nan], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs_y([X1, X2], ensure_all_finite='allow-nan')
    assert np.isnan(result[0][0, 1])

    X1 = np.array([[1, 2], [3, 4]])
    X2 = np.array([[5, 6], [7, 8]])
    result = check_Xs_y([X1, X2], return_dimensions=True)
    assert len(result) == 4
    assert result[1] == 2
    assert result[2] == 2
    assert result[3] == [2, 2]


if __name__ == "__main__":
    pytest.main()
