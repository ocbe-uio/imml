import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import numpy as np

from imml.load import MUSEDataset


@pytest.fixture
def sample_data():
    n_samples = 5
    n_mods = 3
    Xs = [torch.rand((n_samples, 10)) for _ in range(n_mods)]
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float)
    observed_mod_indicator = torch.ones((n_samples, n_mods), dtype=torch.bool)
    y_indicator = torch.ones((n_samples,), dtype=torch.bool)
    return Xs, y, observed_mod_indicator, y_indicator


def test_deepmodule_not_installed(sample_data):
    Xs, y, observed_mod_indicator, y_indicator = sample_data
    MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with patch.dict(sys.modules, {"torch": None}):
        import imml.load.muse_dataset as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y, observed_mod_indicator, y_indicator = sample_data
    dataset = MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    assert len(dataset) == len(y)
    assert hasattr(dataset, 'Xs')
    assert hasattr(dataset, 'y')
    assert hasattr(dataset, 'observed_mod_indicator')
    assert hasattr(dataset, 'y_indicator')
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 4
    assert len(sample[0]) == len(Xs)
    assert isinstance(sample[1], torch.Tensor)
    assert isinstance(sample[2], torch.Tensor)
    assert isinstance(sample[3], torch.Tensor)


def test_invalid_params():
    n_samples = 5
    Xs = [torch.rand((n_samples, 10)) for _ in range(3)]
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float)
    observed_mod_indicator = torch.ones((n_samples, 3), dtype=torch.bool)
    y_indicator = torch.ones((n_samples,), dtype=torch.bool)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs="not_a_list", y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[], y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[torch.rand((n_samples, 10)), torch.rand((0, 10))], y=y,
                   observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid Xs."):
        MUSEDataset(Xs=[torch.rand((n_samples, 10)), torch.rand((n_samples+1, 10))], y=y,
                   observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid y."):
        MUSEDataset(Xs=Xs, y=None, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid y."):
        MUSEDataset(Xs=Xs, y=torch.randint(0, 2, (n_samples+1,)),
                   observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid observed_mod_indicator."):
        MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=None, y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid observed_mod_indicator."):
        MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=torch.ones((n_samples+1, 3)), y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid observed_mod_indicator."):
        MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=torch.ones((n_samples, 2)), y_indicator=y_indicator)
    with pytest.raises(ValueError, match="Invalid y_indicator."):
        MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=None)
    with pytest.raises(ValueError, match="Invalid y_indicator."):
        MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator,
                   y_indicator=torch.ones((n_samples+1,)))


def test_getitem(sample_data):
    Xs, y, observed_mod_indicator, y_indicator = sample_data
    dataset = MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, tuple)
        assert len(sample) == 4
        Xs_idx = sample[0]
        assert len(Xs_idx) == len(Xs)
        for j, X_idx in enumerate(Xs_idx):
            assert X_idx.shape == (Xs[j].shape[1],)
            assert torch.allclose(X_idx, Xs[j][i])
        y_idx = sample[1]
        assert isinstance(y_idx, torch.Tensor)
        assert y_idx.item() == y[i].item()
        observed_mod_indicator_idx = sample[2]
        assert isinstance(observed_mod_indicator_idx, torch.Tensor)
        assert torch.all(observed_mod_indicator_idx == observed_mod_indicator[i])
        y_indicator_idx = sample[3]
        assert isinstance(y_indicator_idx, torch.Tensor)
        assert y_indicator_idx.item() == y_indicator[i].item()


def test_missing_values(sample_data):
    Xs, y, observed_mod_indicator, y_indicator = sample_data
    observed_mod_indicator[0, 0] = False
    observed_mod_indicator[1, 1] = False
    y_indicator[2] = False
    dataset = MUSEDataset(Xs=Xs, y=y, observed_mod_indicator=observed_mod_indicator, y_indicator=y_indicator)
    sample = dataset[0]
    assert not sample[2][0].item()
    assert sample[2][1].item()
    assert sample[2][2].item()
    sample = dataset[1]
    assert sample[2][0].item()
    assert not sample[2][1].item()
    assert sample[2][2].item()
    sample = dataset[2]
    assert not sample[3].item()


if __name__ == "__main__":
    pytest.main()