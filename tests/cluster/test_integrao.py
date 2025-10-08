import pytest
torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from lightning import Trainer
from torch.utils.data import DataLoader

from imml.ampute import Amputer
from imml.cluster import IntegrAO
from imml.load import IntegrAODataset

L.seed_everything(42)
estimator = IntegrAO


@pytest.fixture
def sample_data():
    X = np.random.default_rng(42).random((25, 10))
    X1, X2, X3 = X[:, :3], X[:, 3:5], X[:, 5:]
    Xs_pandas = [pd.DataFrame(X1), pd.DataFrame(X2), pd.DataFrame(X3)]
    Xs_numpy = [X1, X2, X3]
    Xs_torch = [torch.from_numpy(X.astype(np.float32)) for X in Xs_numpy]
    return Xs_torch, Xs_pandas, Xs_numpy


def test_deepmodule_not_installed(sample_data):
    estimator(Xs=sample_data[1])
    with patch.dict(sys.modules, {"torch": None}):
        import imml.cluster.integrao as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(Xs=sample_data[1])
    importlib.reload(module_mock)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with torch_geometric and MPS")
def test_default_params(sample_data):
    n_clusters = 3
    for Xs in sample_data:
        n_samples = len(Xs[0])
        model = estimator(Xs=Xs, n_clusters=n_clusters, random_state=42)
        train_data = IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
        train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=True)
        trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
        trainer.fit(model, train_dataloader, val_dataloaders=train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=False)
        trainer.test(model, train_dataloader)
        labels = trainer.predict(model, train_dataloader)[0]
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        assert len(model.embedding_) == n_samples


def test_invalid_params(sample_data):
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid', Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(Xs="invalid")
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=0., Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=-1, Xs=sample_data[0])


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with torch_geometric and MPS")
def test_missing_values_handling(sample_data):
    n_clusters = 2
    for Xs in sample_data[1:]:
        n_samples = len(Xs[0])
        amputer = Amputer(p= 0.3, random_state=42)
        Xs = amputer.fit_transform(Xs)
        model = estimator(Xs=Xs, n_clusters=n_clusters, random_state=42)
        train_data = IntegrAODataset(Xs=Xs, neighbor_size=model.neighbor_size, networks=model.fused_networks_)
        train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=True)
        trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
        trainer.fit(model, train_dataloader)
        train_dataloader = DataLoader(dataset=train_data, batch_size=len(Xs[0]), shuffle=False)
        labels = trainer.predict(model, train_dataloader)[0]
        assert labels is not None
        assert len(labels) == n_samples
        assert len(np.unique(labels)) == model.n_clusters
        assert min(labels) == 0
        assert max(labels) == (model.n_clusters - 1)
        assert not np.isnan(labels).any()
        embedding_ = model.embedding_
        assert not np.isnan(embedding_).any().any()
        assert len(embedding_) == n_samples


if __name__ == "__main__":
    pytest.main()
