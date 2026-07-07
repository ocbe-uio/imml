import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from imml.load import MultiSurvDataset

custom_dataset = MultiSurvDataset


@pytest.fixture
def sample_data():
    n_samples = 5
    n_mods = 3
    Xs = [torch.rand((n_samples, 10)) for _ in range(n_mods)]
    y = torch.tensor(
        [[1., 0.5], [0., 1.5], [1., 2.5], [0., 3.5], [1., 4.5]],
        dtype=torch.float,
    )
    return Xs, y


def test_deepmodule_not_installed(sample_data):
    Xs, y = sample_data
    custom_dataset(Xs=Xs, y=y)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.load.multisurv_dataset as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            custom_dataset(Xs=Xs, y=y)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    assert hasattr(dataset, "Xs")
    assert hasattr(dataset, "y")
    assert hasattr(dataset, "transform")
    assert len(dataset) == len(y)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 2
    assert len(sample[0]) == len(Xs)
    assert isinstance(sample[1], torch.Tensor)


def test_invalid_params(sample_data):
    n_samples = 5
    Xs, y = sample_data
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=None)
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.rand((n_samples + 1, 2)))
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.rand(n_samples))
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.rand((n_samples, 3)))
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.column_stack((torch.arange(n_samples), torch.full((n_samples,), 2.0))))
    with pytest.raises(ValueError, match="Invalid y."):
        custom_dataset(Xs=Xs, y=torch.column_stack((torch.arange(n_samples), torch.full((n_samples,), -1.0))))
    with pytest.raises(ValueError, match="Invalid transform."):
        custom_dataset(Xs=Xs, y=torch.rand((n_samples, 2)), transform="not_a_list")
    with pytest.raises(ValueError, match="Invalid transform."):
        custom_dataset(Xs=Xs, y=torch.rand((n_samples, 2)), transform=[lambda x: x])
    with pytest.raises(ValueError, match="Invalid transform."):
        custom_dataset(Xs=Xs, y=torch.rand((n_samples, 2)), transform=[lambda x: x, None, lambda x: x])


def test_loader(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    data_loader = DataLoader(dataset=dataset)
    batch = next(iter(data_loader))
    assert len(batch) == 2
    assert isinstance(batch, list)
    assert isinstance(batch[0], list)
    assert isinstance(batch[1], torch.Tensor)

    data_loader = DataLoader(dataset=dataset, batch_size=2)
    batch = next(iter(data_loader))
    assert batch[0][0].shape == (2, Xs[0].shape[1])
    assert batch[1].shape == (2, 2)


def test_pandas_numpy_inputs(sample_data):
    Xs, y = sample_data
    Xs = [pd.DataFrame(X.numpy()) for X in Xs]
    y = pd.DataFrame(y.numpy(), columns=["time", "event"])
    dataset = custom_dataset(Xs=Xs, y=y)
    sample = dataset[0]

    assert all(torch.is_tensor(X) for X in dataset.Xs)
    assert isinstance(sample[1], torch.Tensor)
    assert sample[1].shape == (2,)


def test_transform(sample_data):
    Xs, y = sample_data
    transforms = [lambda x: x + 1, lambda x: x * 2, lambda x: x - 1]
    dataset = custom_dataset(Xs=Xs, y=y, transform=transforms)
    Xs_idx, y_idx = dataset[0]

    assert torch.allclose(Xs_idx[0], Xs[0][0] + 1)
    assert torch.allclose(Xs_idx[1], Xs[1][0] * 2)
    assert torch.allclose(Xs_idx[2], Xs[2][0] - 1)
    assert torch.allclose(y_idx, y[0])


def test_getitem(sample_data):
    Xs, y = sample_data
    dataset = custom_dataset(Xs=Xs, y=y)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        Xs_idx, y_idx = sample
        assert len(Xs_idx) == len(Xs)
        for j, X_idx in enumerate(Xs_idx):
            assert X_idx.shape == (Xs[j].shape[1],)
            assert torch.allclose(X_idx, Xs[j][i])
        assert isinstance(y_idx, torch.Tensor)
        assert torch.allclose(y_idx, y[i])


def test_example():
    Xs = [torch.rand((20, 10)) for _ in range(3)]
    y = torch.tensor(np.column_stack((np.random.default_rng(42).integers(0, 2, 20),
                                      np.random.default_rng(42).uniform(0.5, 5, 20)))).float()
    train_data = MultiSurvDataset(Xs=Xs, y=y)
    train_dataloader = DataLoader(dataset=train_data)
    next(iter(train_dataloader))


if __name__ == "__main__":
    pytest.main()
