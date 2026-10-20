import pytest
torch = pytest.importorskip("torch")
L = pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from torch.utils.data import DataLoader
from lightning import Trainer

from imml.ampute import Amputer
from imml.preprocessing import MMTransformer
from imml.cluster import MRGCN
from imml.load import MRGCNDataset
from tests._sample_data import make_sample_data

L.seed_everything(42)
estimator = MRGCN
sample_data = make_sample_data(n_samples=25, n_features=10, splits=(3, 5))


def test_deepmodule_not_installed():
    estimator(Xs=sample_data[0])
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.cluster.mrgcn as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(Xs=sample_data[0])
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    n_clusters = 3
    n_samples = len(Xs[0])
    train_data = MRGCNDataset(Xs=Xs)
    train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=True)
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    model = estimator(Xs=Xs, n_clusters=n_clusters, random_state=42)
    trainer.fit(model, train_dataloader)
    train_dataloader = DataLoader(dataset=train_data, batch_size=n_samples, shuffle=False)
    labels = trainer.predict(model, train_dataloader)[0]
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == model.n_clusters
    assert min(labels) == 0
    assert max(labels) == (model.n_clusters - 1)
    assert not np.isnan(labels).any()
    embedding_ = model._embedding(batch=next(iter(train_dataloader))).detach().cpu().numpy()
    assert len(embedding_) == n_samples


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters='invalid', Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid n_clusters."):
        estimator(n_clusters=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid Xs."):
        estimator(Xs="invalid")
    with pytest.raises(ValueError, match="Invalid k_num."):
        estimator(k_num="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid k_num."):
        estimator(k_num=0, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=0., Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(learning_rate=-1, Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid reg2."):
        estimator(reg2="invalid", Xs=sample_data[0])
    with pytest.raises(ValueError, match="Invalid reg3."):
        estimator(reg3="invalid", Xs=sample_data[0])


@pytest.mark.parametrize("Xs", sample_data)
def test_missing_values_handling(Xs):
    n_clusters = 2
    n_samples = len(Xs[0])
    amputer = Amputer(p= 0.3, random_state=42)
    imputer = SimpleImputer(strategy="constant", fill_value=0.0).set_output(transform="pandas")
    transformer = FunctionTransformer(lambda x: torch.from_numpy(x.values.astype(np.float32)))
    pipeline = make_pipeline(amputer, MMTransformer(imputer), MMTransformer(transformer))
    transformed_Xs = pipeline.fit_transform(Xs)
    train_data = MRGCNDataset(Xs=transformed_Xs)
    train_dataloader = DataLoader(dataset=train_data, batch_size=len(transformed_Xs[0]), shuffle=True)
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    model = estimator(Xs=transformed_Xs, n_clusters=n_clusters, random_state=42)
    trainer.fit(model, train_dataloader)
    train_dataloader = DataLoader(dataset=train_data, batch_size=len(transformed_Xs[0]), shuffle=False)
    labels = trainer.predict(model, train_dataloader)[0]
    assert labels is not None
    assert len(labels) == n_samples
    assert len(np.unique(labels)) == model.n_clusters
    assert min(labels) == 0
    assert max(labels) == (model.n_clusters - 1)
    assert not np.isnan(labels).any()
    embedding_ = model._embedding(batch=next(iter(train_dataloader))).detach().cpu().numpy()
    assert not np.isnan(embedding_).any().any()
    assert len(embedding_) == n_samples


if __name__ == "__main__":
    pytest.main()
