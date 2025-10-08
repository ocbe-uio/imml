import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from imml.cluster import IntegrAO
from imml.load import IntegrAODataset


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    X1, X2, X3 = X[:, :3], X[:, 3:5], X[:, 5:]
    Xs_pandas = [pd.DataFrame(X1), pd.DataFrame(X2), pd.DataFrame(X3)]
    Xs_numpy = [X1, X2, X3]
    Xs_torch = [torch.from_numpy(X.astype(np.float32)) for X in Xs_numpy]
    return Xs_torch, Xs_pandas, Xs_numpy


def test_deepmodule_not_installed(sample_data):
    n_clusters = 3
    Xs = sample_data[0]
    model = IntegrAO(Xs=Xs, n_clusters=n_clusters, random_state=42)
    IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with patch.dict(sys.modules, {"torch": None}):
        import imml.load.integrao_dataset as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    n_clusters = 3
    for Xs in sample_data:
        model = IntegrAO(Xs=Xs, n_clusters=n_clusters, random_state=42)
        dataset = IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
        assert len(dataset) == len(Xs[0])
        assert hasattr(dataset, 'Xs')
        assert len(dataset.Xs) == len(Xs)


def test_invalid_params(sample_data):
    n_clusters = 3
    model = IntegrAO(Xs=sample_data[0], n_clusters=n_clusters, random_state=42)
    with pytest.raises(ValueError, match="Invalid Xs."):
        IntegrAODataset(Xs="invalid_input", neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid Xs."):
        IntegrAODataset(Xs=[], neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid Xs."):
        IntegrAODataset(Xs=[torch.rand((5, 10)), torch.rand((0, 10))],
                        neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid Xs."):
        IntegrAODataset(Xs=[torch.rand((5, 10)), torch.rand((6, 10))],
                        neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid neighbor_size."):
        IntegrAODataset(Xs=[torch.rand((5, 10)), torch.rand((5, 10))],
                        neighbor_size=1.5, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid neighbor_size."):
        IntegrAODataset(Xs=[torch.rand((5, 10)), torch.rand((5, 10))],
                        neighbor_size=0, networks=model.fused_networks_)
    with pytest.raises(ValueError, match="Invalid networks."):
        IntegrAODataset(Xs=[torch.rand((5, 10)), torch.rand((5, 10))],
                        neighbor_size=model.neighbor_size, networks="invalid_input")


def test_getitem(sample_data):
    n_clusters = 3
    Xs = sample_data[0]
    model = IntegrAO(Xs=Xs, n_clusters=n_clusters, random_state=42)
    dataset = IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    sample = dataset[0]
    assert isinstance(sample, tuple)
    assert len(sample) == 4
    for s, X in zip(sample[0], Xs):
        assert s.shape == (X.shape[1],)
    for s, X in zip(sample[2], Xs):
        assert s.shape == (len(X),)
    assert len([dataset[i] for i in range(3)]) == 3


if __name__ == "__main__":
    pytest.main()
