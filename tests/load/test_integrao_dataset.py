import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import pytest
from tests._sample_data import make_sample_data

from imml.cluster import IntegrAO
from imml.load import IntegrAODataset


sample_data = make_sample_data(n_samples=25, n_features=10)


def test_deepmodule_not_installed():
    n_clusters = 3
    Xs = sample_data[0]
    model = IntegrAO(Xs=Xs, n_clusters=n_clusters, random_state=42)
    IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.load.integrao_dataset as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    n_clusters = 3
    model = IntegrAO(Xs=Xs, n_clusters=n_clusters, random_state=42)
    dataset = IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
    assert len(dataset) == len(Xs[0])
    assert hasattr(dataset, 'Xs')
    assert len(dataset.Xs) == len(Xs)


def test_invalid_params():
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


@pytest.mark.parametrize("Xs", sample_data)
def test_getitem(Xs):
    n_clusters = 3
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
