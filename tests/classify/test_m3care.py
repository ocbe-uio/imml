import pytest
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
L = pytest.importorskip("lightning")
import importlib
import sys
from unittest.mock import patch

from imml.classify import M3Care

estimator = M3Care


@pytest.fixture
def sample_data():
    batch_size = 2
    n_modalities = 3
    Xs = [torch.rand((batch_size, 10)) for _ in range(n_modalities)]
    y = torch.tensor([0, 1], dtype=torch.float)
    observed_mod_indicator = torch.ones((batch_size, n_modalities), dtype=torch.bool)
    return Xs, y, observed_mod_indicator


def test_deepmodule_not_installed():
    estimator(modalities=["image", "text"])
    with patch.dict(sys.modules, {"torch": None}):
        import imml.classify.m3care as module_mock
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(modalities=["image", "text"])
    importlib.reload(module_mock)


def test_default_params(sample_data):
    model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])
    assert hasattr(model, 'model')
    assert hasattr(model, 'learning_rate')
    assert hasattr(model, 'weight_decay')
    assert hasattr(model, 'loss_fn')

    with torch.no_grad():
        loss = model.training_step(sample_data)
    assert isinstance(loss, torch.Tensor)


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator(modalities=["tabular", "tabular"], input_dim="not_a_list")
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(modalities=["tabular", "tabular"], hidden_dim=None)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(modalities=["tabular", "tabular"], hidden_dim=-1)
    with pytest.raises(ValueError, match="Invalid embed_size."):
        estimator(modalities=["tabular", "tabular"], embed_size=None)
    with pytest.raises(ValueError, match="Invalid embed_size."):
        estimator(modalities=["tabular", "tabular"], embed_size=-1)
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=None)
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=[])
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=["invalid_modality"])
    with pytest.raises(ValueError, match="Invalid modalities."):
        estimator(modalities=["tabular"])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(modalities=["tabular", "tabular"], learning_rate=None)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(modalities=["tabular", "tabular"], learning_rate=-1.)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(modalities=["tabular", "tabular"], weight_decay=None)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(modalities=["tabular", "tabular"], weight_decay=-1.)
    with pytest.raises(ValueError, match="Invalid output_dim."):
        estimator(modalities=["tabular", "tabular"], output_dim=None)
    with pytest.raises(ValueError, match="Invalid output_dim."):
        estimator(modalities=["tabular", "tabular"], output_dim=-1)
    with pytest.raises(ValueError, match="Invalid loss_fn."):
        estimator(modalities=["tabular", "tabular"], loss_fn="not_callable")
    with pytest.raises(ValueError, match="Invalid keep_prob."):
        estimator(modalities=["tabular", "tabular"], keep_prob=None)
    with pytest.raises(ValueError, match="Invalid keep_prob."):
        estimator(modalities=["tabular", "tabular"], keep_prob=-1)
    with pytest.raises(ValueError, match="Invalid keep_prob."):
        estimator(modalities=["tabular", "tabular"], keep_prob=2.)
    with pytest.raises(ValueError, match="Invalid extractors."):
        estimator(modalities=["tabular", "tabular"], extractors="not_a_list")
    with pytest.raises(ValueError, match="Invalid vocab."):
        estimator(modalities=["tabular", "tabular"], vocab="not_a_list")


def test_lightning_methods(sample_data):
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
    model = estimator(modalities=["tabular", "tabular", "tabular"], input_dim=[10, 10, 10])
    Xs, y, observed_mod_indicator = sample_data
    observed_mod_indicator[0, 0] = False
    observed_mod_indicator[1, 1] = False
    with torch.no_grad():
        loss = model.training_step((Xs, y, observed_mod_indicator))
    assert isinstance(loss, torch.Tensor)


def test_custom_loss_fn(sample_data):
    model = estimator(modalities=["tabular", "tabular", "tabular"],
                     input_dim=[10, 10, 10],
                     loss_fn=torch.nn.functional.binary_cross_entropy_with_logits)
    with torch.no_grad():
        loss = model.training_step(sample_data, 0)
    assert isinstance(loss, torch.Tensor)


if __name__ == "__main__":
    pytest.main()