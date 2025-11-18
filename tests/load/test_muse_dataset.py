import pandas as pd
import pytest
torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader
import importlib
import sys
from unittest.mock import patch

from imml.load import MUSEDataset

custom_dataset = MUSEDataset


@pytest.fixture
def sample_data():
    n_samples = 5
    n_mods = 3
    Xs = [torch.rand((n_samples, 10)) for _ in range(n_mods)]
    y = torch.randint(0, 2, (n_samples,), dtype=torch.float)
    return Xs, y


def test_deepmodule_not_installed(sample_data):
    Xs, y = sample_data
    custom_dataset(Xs=Xs, y=y)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.load.muse_dataset as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            custom_dataset(Xs=Xs, y=y)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    assert len(dataset) == len(y)
    assert hasattr(dataset, 'Xs')
    assert hasattr(dataset, 'y')
    assert hasattr(dataset, 'missing_mod_indicator')
    assert hasattr(dataset, 'y_indicator')
    assert len(dataset) == len(y)
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
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=None)
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.randint(0, 2, (n_samples+1,)))


def test_getitem(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, tuple)
        assert len(sample) == 4
        Xs_idx, y_idx, missing_mod_indicator_idx, y_indicator_idx = sample[0], sample[1], sample[2], sample[3]
        assert len(Xs_idx) == len(Xs)
        for j, X_idx in enumerate(Xs_idx):
            assert X_idx.shape == (Xs[j].shape[1],)
            assert torch.allclose(X_idx, Xs[j][i])
        assert isinstance(y_idx, torch.Tensor)
        assert y_idx.item() == y[i].item()
        assert isinstance(missing_mod_indicator_idx, torch.Tensor)
        assert not torch.all(missing_mod_indicator_idx)
        assert isinstance(y_indicator_idx, torch.Tensor)
        assert y_indicator_idx


def test_loader(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    data_loader = DataLoader(dataset=dataset)
    batch = next(iter(data_loader))
    assert len(batch) == 4
    assert isinstance(batch, list)
    assert isinstance(batch[0], list)
    assert isinstance(batch[1], torch.Tensor)
    assert isinstance(batch[2], torch.Tensor)
    assert isinstance(batch[3], torch.Tensor)
    data_loader = DataLoader(dataset=dataset, batch_size=2)
    batch = next(iter(data_loader))


def test_tab_text(sample_data):
    Xs, y = sample_data
    Xs = [
        pd.DataFrame(Xs[0]),
        pd.DataFrame(["This is the graphical abstract of iMML."]*len(Xs[0])),
    ]
    y = pd.Series(y)
    dataset = custom_dataset(Xs=Xs, y=y)
    data_loader = DataLoader(dataset=dataset)
    batch = next(iter(data_loader))
    assert len(batch) == 4
    assert isinstance(batch, list)
    assert isinstance(batch[0], list)
    assert isinstance(batch[1], torch.Tensor)
    assert isinstance(batch[2], torch.Tensor)
    assert isinstance(batch[3], torch.Tensor)


def test_missing_values(sample_data):
    Xs, y = sample_data
    Xs[0][0, :] = torch.nan
    Xs[1][1, :] = torch.nan
    Xs[2][2, :] = torch.nan
    y[1] = torch.nan
    dataset = custom_dataset(Xs=Xs, y=y)
    sample = dataset[0]
    assert sample[2][0].item()
    assert not sample[2][1].item()
    assert not sample[2][2].item()
    assert sample[3]
    sample = dataset[1]
    assert not sample[2][0].item()
    assert sample[2][1].item()
    assert not sample[2][2].item()
    assert not sample[3]
    sample = dataset[2]
    assert not sample[2][0].item()
    assert not sample[2][1].item()
    assert sample[2][2].item()
    assert sample[3]


if __name__ == "__main__":
    pytest.main()