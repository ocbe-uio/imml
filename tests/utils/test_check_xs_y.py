import numpy as np
import pandas as pd
import pytest

from imml.utils import check_Xs_y
from imml import deepmodule_installed

if deepmodule_installed:
    import torch


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
    with pytest.raises(ValueError, match="Invalid modalities."):
        check_Xs_y([X1, X2], modalities=3)

    with pytest.raises(ValueError, match="Invalid Xs. All modalities should have the same number of samples"):
        check_Xs_y([X1[:-1], X2])
    with pytest.raises(ValueError, match="Invalid Xs. All modalities should be the same data type"):
        check_Xs_y([X1, pd.DataFrame(X2)])
    with pytest.raises(ValueError, match="Invalid Xs. All elements must have at least one sample"):
        check_Xs_y([np.empty((0, 2)), np.empty((0, 2))])
    with pytest.raises(ValueError, match="Invalid modalities. Wrong number of modalities"):
        check_Xs_y([X1, X2], modalities=["mod1"])
    with pytest.raises(ValueError, match="Invalid mod_types. It must be a list"):
        check_Xs_y([X1, X2], mod_types=3)
    with pytest.raises(ValueError, match="Invalid mod_types. Wrong number of mod_types"):
        check_Xs_y([X1, X2], modalities=["mod1", "mod2"], mod_types=["mod1"])
    with pytest.raises(ValueError, match="Invalid modalities. Expected options are"):
        check_Xs_y([X1, X2], modalities=["mod1", "mod3"], mod_types=["mod1", "mod2"])
    with pytest.raises(ValueError, match="Invalid Xs. There are samples with no available data"):
        check_Xs_y([np.array([[np.nan, np.nan], [1, 2]]), np.array([[np.nan, np.nan], [3, 4]])])
    with pytest.raises(ValueError, match="Invalid y. It cannot be None"):
        check_Xs_y([X1, X2], supervised=True)
    with pytest.raises(ValueError, match="Invalid y. It must have the same length"):
        check_Xs_y([X1, X2], y=np.array([1]), supervised=True)


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

    result = check_Xs_y([X1.tolist(), X2.tolist()])
    assert isinstance(result, list)
    assert isinstance(result[0], np.ndarray)
    assert result[0].shape == (2, 2)

    result = check_Xs_y([X1, X2], y=np.array([0, 1]), modalities=["mod1", "mod2"],
                        mod_types=["mod1", "mod2"], supervised=True)
    assert len(result) == 2


if __name__ == "__main__":
    pytest.main()
