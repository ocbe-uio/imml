import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
L = pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch

from imml.classify import MUSE

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
            estimator(modalities=["image", "text"])
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
        assert isinstance(preds, torch.Tensor)
        optimizer = model.configure_optimizers()
        assert isinstance(optimizer, torch.optim.Optimizer)


def test_missing_values_handling(sample_data):
    with torch.no_grad():
        # Create model
        model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])

        # Create batch with missing values
        Xs, y, observed_mod_indicator, y_indicator = sample_data
        observed_mod_indicator[0, 0] = False  # First modality is missing for first sample
        observed_mod_indicator[1, 1] = False  # Second modality is missing for second sample

        # Test forward pass with missing values
        loss = model.training_step((Xs, y, observed_mod_indicator, y_indicator), 0)
        assert isinstance(loss, torch.Tensor)

        # Test with missing labels
        y_indicator[0] = False  # First label is missing
        loss = model.training_step((Xs, y, observed_mod_indicator, y_indicator), 0)
        assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main()