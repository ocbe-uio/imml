import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
L = pytest.importorskip("lightning")
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import importlib
import sys
from unittest.mock import patch

from imml.classify import MUSE
from imml.load import MUSEDataset

estimator = MUSE


@pytest.fixture
def sample_data():
    batch_size = 2
    n_modalities = 3
    Xs = [torch.rand((batch_size, 10)) for _ in range(n_modalities)]
    y = torch.tensor([0, 1], dtype=torch.float)
    observed_mod_indicator = torch.ones((batch_size, n_modalities), dtype=torch.bool)
    y_indicator = torch.ones((batch_size), dtype=torch.bool)
    return Xs, y, observed_mod_indicator, y_indicator


def test_deepmodule_not_installed():
    estimator(modalities=["text", "text"])
    with patch.dict(sys.modules, {"torch": None}):
        import imml.classify.muse as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(modalities=["tabular", "text"])
    importlib.reload(module_mock)


def test_default_params(sample_data):
    model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])
    assert hasattr(model, 'model')
    assert hasattr(model, 'learning_rate')
    assert hasattr(model, 'weight_decay')
    with torch.no_grad():
        loss = model.training_step(sample_data)
    assert isinstance(loss, torch.Tensor)


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator(modalities=["tabular", "tabular"], input_dim="not_a_list")
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(modalities=["tabular"], hidden_dim=None)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(modalities=["tabular"], hidden_dim=-1)
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=None)
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=[])
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=["invalid_modality"])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(modalities=["tabular"], learning_rate=None)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(modalities=["tabular"], learning_rate=-1.)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(modalities=["tabular"], weight_decay=None)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(modalities=["tabular"], weight_decay=-1.)
    with pytest.raises(ValueError, match="Invalid output_dim."):
        estimator(modalities=["tabular"], output_dim=None)
    with pytest.raises(ValueError, match="Invalid output_dim."):
        estimator(modalities=["tabular"], output_dim=-1)
    with pytest.raises(ValueError, match="Invalid extractors."):
        estimator(modalities=["tabular"], extractors="not_a_list")
    with pytest.raises(ValueError, match="Invalid gnn_layers."):
        estimator(modalities=["tabular"], gnn_layers=None)
    with pytest.raises(ValueError, match="Invalid gnn_layers."):
        estimator(modalities=["tabular"], gnn_layers=-1)
    with pytest.raises(ValueError, match="Invalid gnn_norm."):
        estimator(modalities=["tabular"], gnn_norm=123)
    with pytest.raises(ValueError, match="Invalid code_pretrained_embedding."):
        estimator(modalities=["tabular"], code_pretrained_embedding="not_a_bool")
    with pytest.raises(ValueError, match="Invalid bert_type."):
        estimator(modalities=["tabular"], bert_type=123)
    with pytest.raises(ValueError, match="Invalid dropout."):
        estimator(modalities=["tabular"], dropout=None)
    with pytest.raises(ValueError, match="Invalid dropout."):
        estimator(modalities=["tabular"], dropout=-1.)
    with pytest.raises(ValueError, match="Invalid dropout."):
        estimator(modalities=["tabular"], dropout=2.)


def test_lightning_methods(sample_data):
    with torch.no_grad():
        model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])
        loss = model.training_step(sample_data)
        assert isinstance(loss, torch.Tensor)
        loss = model.validation_step(sample_data)
        assert isinstance(loss, torch.Tensor)
        loss = model.test_step(sample_data)
        assert isinstance(loss, torch.Tensor)
        preds = model.predict_step(sample_data)
        assert not torch.isnan(preds).any()
        assert isinstance(preds, torch.Tensor)
        assert torch.ge(preds, 0).all() and torch.le(preds, 2).all()
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)


def test_missing_values_handling(sample_data):
    model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])
    sample_data[2][0, 0] = False
    sample_data[2][1, 1] = False
    with torch.no_grad():
        loss = model.training_step(sample_data)
    assert isinstance(loss, torch.Tensor)
    sample_data[3][0] = False
    with torch.no_grad():
        loss = model.training_step(sample_data)
    assert isinstance(loss, torch.Tensor)


def test_custom_loss_fn(sample_data):
    model = estimator(modalities=["tabular", "tabular", "tabular"],
                     input_dim=[10, 10, 10],
                     loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    with torch.no_grad():
        loss = model.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    preds = model.predict_step(sample_data)
    assert isinstance(preds, torch.Tensor)
    assert not torch.isnan(preds).any()
    assert preds.ndim == 1
    assert len(preds) == len(sample_data[1])
    assert torch.ge(preds, 0).all() and torch.le(preds, 2).all()

    model = estimator(modalities=["tabular", "tabular", "tabular"],
                     input_dim=[10, 10, 10], output_dim=2,
                     loss_fn=torch.nn.functional.cross_entropy)
    sample_data = (sample_data[0], sample_data[1].long(), sample_data[2], sample_data[3])
    with torch.no_grad():
        loss = model.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()
    preds = model.predict_step(sample_data)
    assert isinstance(preds, torch.Tensor)
    assert not torch.isnan(preds).any()
    assert preds.ndim == 2
    assert len(preds) == len(sample_data[1])
    assert preds.shape[1] == 2
    assert torch.ge(preds, 0).all() and torch.le(preds, 2).all()


def test_tab_text(sample_data):
    model = estimator(modalities=["tabular", "text"],
                     input_dim=[10])
    Xs = [
        pd.DataFrame(sample_data[0][0].numpy()),
        pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]),
    ]
    dataset = MUSEDataset(Xs=Xs, y=sample_data[1])
    sample_data = next(iter(DataLoader(dataset=dataset, batch_size=2)))
    with torch.no_grad():
        loss = model.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)


def test_incomplete_tab_text(sample_data):
    model = estimator(modalities=["tabular", "text"],
                     input_dim=[10])
    Xs = [
        pd.DataFrame(sample_data[0][0].numpy()),
        pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]),
    ]
    Xs[0].iloc[0,:] = np.nan
    Xs[1].iloc[1, 0] = np.nan
    dataset = MUSEDataset(Xs=Xs, y=sample_data[1])
    sample_data = next(iter(DataLoader(dataset=dataset, batch_size=2)))
    with torch.no_grad():
        loss = model.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="Error with MPS")
def test_example(sample_data):
    from lightning import Trainer
    import numpy as np
    import pandas as pd
    from torch.utils.data import DataLoader
    from imml.classify import MUSE
    from imml.load import MUSEDataset
    from imml.ampute import Amputer
    Xs = [pd.DataFrame(np.random.default_rng(42).random((2, 10)))]
    Xs.append(pd.DataFrame(np.random.default_rng(42).random((2, 15))))
    Xs.append(pd.DataFrame(["This is the graphical abstract of iMML.", "This is the logo of iMML."]))
    Xs = Amputer(p=0.2, random_state=42).fit_transform(Xs)  # this step is optional
    y = pd.Series(np.random.default_rng(42).integers(0, 2, len(Xs[0])), dtype=np.float32)
    train_data = MUSEDataset(Xs=Xs, y=y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    modalities = ["tabular", "tabular", "text"]
    estimator = MUSE(modalities=modalities, input_dim=[Xs[0].shape[1], Xs[1].shape[1]])
    trainer.fit(estimator, train_dataloader)
    trainer.predict(estimator, train_dataloader)


if __name__ == "__main__":
    pytest.main()