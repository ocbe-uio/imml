import pytest
torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch
import numpy as np
from lightning import Trainer
from torch.utils.data import DataLoader

from imml.ampute import Amputer
from imml.cluster import IntegrAO
from imml.load import IntegrAODataset
from tests._sample_data import make_sample_data

L.seed_everything(42)
estimator = IntegrAO
sample_data = make_sample_data(n_samples=25, n_features=10, splits=(3, 5))


def test_deepmodule_not_installed():
    estimator(Xs=sample_data[-1])
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.cluster.integrao as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(Xs=sample_data[-1])
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with MPS")
@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    n_clusters = 3
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


def test_invalid_params():
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


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with MPS")
@pytest.mark.parametrize("Xs", sample_data)
def test_missing_values_handling(Xs):
    n_clusters = 2
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


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with MPS")
def test_example():
    import numpy as np
    import torch
    from imml.cluster import IntegrAO
    from lightning import Trainer
    from torch.utils.data import DataLoader
    from imml.load import IntegrAODataset
    Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    estimator = IntegrAO(Xs=Xs, random_state=42)
    train_data = IntegrAODataset(Xs=Xs, neighbor_size=estimator.neighbor_size, networks=estimator.fused_networks_)
    train_dataloader = DataLoader(dataset=train_data, batch_size=len(Xs[0]))
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(estimator, train_dataloader)
    labels = trainer.predict(estimator, train_dataloader)[0]

if __name__ == "__main__":
    pytest.main()
