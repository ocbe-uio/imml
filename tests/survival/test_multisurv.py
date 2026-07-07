import pytest
torch = pytest.importorskip("torch")
import importlib
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from imml.fuse import ConcatFusion, MeanFusion
from imml.load import MultiSurvDataset
from imml.survival import MultiSurv as estimator


@pytest.fixture
def sample_data():
    batch_size = 4
    n_modalities = 3
    Xs = [torch.rand((batch_size, 10)) for _ in range(n_modalities)]
    y = torch.tensor(
        [[1., 0.5], [0., 1.5], [1., 2.5], [0., 3.5]],
        dtype=torch.float,
    )
    return Xs, y


def test_deepmodule_not_installed(sample_data):
    estimator(input_dim=[10, 10], hidden_dim=8, embed_size=8, t_binds=4)
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.survival.multisurv as module_mock

        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(input_dim=[10, 10], hidden_dim=8, embed_size=8, t_binds=4)
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


def test_default_params(sample_data):
    Xs, y = sample_data
    model = estimator(input_dim=[10, 10, 10], hidden_dim=8, embed_size=8, t_binds=4)

    assert hasattr(model, "model")
    assert hasattr(model, "learning_rate")
    assert hasattr(model, "weight_decay")
    assert hasattr(model, "loss_fn")
    assert torch.equal(model.time_points, torch.arange(0.0, 365.0 * 5, 365.0))

    with torch.no_grad():
        loss = model.training_step((Xs, y))
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss).any()


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator(input_dim="not_a_list")
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator()
    with pytest.raises(ValueError, match="Invalid t_binds."):
        estimator(input_dim=[10, 10], t_binds=1.5)
    with pytest.raises(ValueError, match="Invalid t_binds."):
        estimator(input_dim=[10, 10], t_binds=-1)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(input_dim=[10, 10], hidden_dim=None)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(input_dim=[10, 10], hidden_dim=-1)
    with pytest.raises(ValueError, match="Invalid embed_size."):
        estimator(input_dim=[10, 10], embed_size=None)
    with pytest.raises(ValueError, match="Invalid embed_size."):
        estimator(input_dim=[10, 10], embed_size=-1)
    with pytest.raises(ValueError, match="Invalid n_layers."):
        estimator(input_dim=[10, 10], n_layers=None)
    with pytest.raises(ValueError, match="Invalid n_layers."):
        estimator(input_dim=[10, 10], n_layers=-1)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(input_dim=[10, 10], learning_rate=None)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(input_dim=[10, 10], learning_rate=-1.0)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(input_dim=[10, 10], weight_decay=None)
    with pytest.raises(ValueError, match="Invalid weight_decay."):
        estimator(input_dim=[10, 10], weight_decay=-1.0)
    with pytest.raises(ValueError, match="Invalid fusion."):
        estimator(input_dim=[10, 10], fusion="max")
    with pytest.raises(ValueError, match="Invalid extractors."):
        estimator(input_dim=[10, 10], extractors="not_a_list")
    with pytest.raises(ValueError, match="Invalid extractors."):
        estimator(extractors=[torch.nn.Linear(10, 8), object()])
    with pytest.raises(ValueError, match="Invalid time_points."):
        estimator(input_dim=[10, 10], t_binds=4, time_points=[0, 365])


def test_lightning_methods(sample_data):
    Xs, y = sample_data
    model = estimator(input_dim=[10, 10, 10], hidden_dim=8, embed_size=8, t_binds=4)

    with torch.no_grad():
        loss = model.training_step((Xs, y), 0)
        assert isinstance(loss, torch.Tensor)
        loss = model.validation_step((Xs, y), 0)
        assert isinstance(loss, torch.Tensor)
        loss = model.test_step((Xs, y), 0)
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss).any()
        preds = model.predict_step((Xs, y), 0)
        assert isinstance(preds, torch.Tensor)
        assert preds.shape == (5, len(y))
        assert torch.allclose(preds[0], torch.ones(len(y)))
        assert torch.ge(preds, 0).all() and torch.le(preds, 1).all()

    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)


def test_forward_matches_original_output_contract(sample_data):
    Xs, _ = sample_data
    model = estimator(input_dim=[10, 10, 10], hidden_dim=8, embed_size=8, t_binds=4)

    with torch.no_grad():
        features, risk = model.model(Xs)

    assert len(features) == len(Xs)
    assert all(feature.shape == (len(Xs[0]), 8) for feature in features)
    assert risk.shape == (len(Xs[0]), 4)
    assert torch.ge(risk, 0).all() and torch.le(risk, 1).all()


def test_dict_input_and_custom_fusion(sample_data):
    Xs, _ = sample_data
    model = estimator(
        input_dim=[10, 10],
        fusion=MeanFusion(),
        hidden_dim=8,
        embed_size=8,
        t_binds=4,
    )

    with torch.no_grad():
        features, risk = model.model(Xs[:2])

    assert len(features) == 2
    assert all(feature.shape[1] == 8 for feature in features)
    assert risk.shape == (len(Xs[0]), 4)


def test_custom_extractors(sample_data):
    Xs, y = sample_data
    extractors = [torch.nn.Linear(10, 8), torch.nn.Linear(10, 8), torch.nn.Linear(10, 8)]
    model = estimator(
        extractors=extractors,
        hidden_dim=8,
        embed_size=8,
        t_binds=4,
    )

    with torch.no_grad():
        loss = model.training_step((Xs, y), 0)
        preds = model.predict_step((Xs, y), 0)

    assert isinstance(loss, torch.Tensor)
    assert preds.shape == (5, len(y))


def test_tensor_time_points(sample_data):
    Xs, y = sample_data
    time_points = torch.tensor([0.0, 365.0, 730.0, 1095.0, 1460.0])
    model = estimator(
        input_dim=[10, 10, 10],
        hidden_dim=8,
        embed_size=8,
        t_binds=4,
        time_points=time_points,
    )

    with torch.no_grad():
        loss = model.training_step((Xs, y), 0)

    assert model.time_points.dtype == torch.float32
    assert isinstance(loss, torch.Tensor)


def test_missing_values_handling(sample_data):
    Xs, y = sample_data
    Xs[0][0, :] = 0.
    model = estimator(
        extractors=[torch.nn.Identity(), torch.nn.Identity()],
        fusion=MeanFusion(),
        hidden_dim=10,
        embed_size=8,
        t_binds=4,
    )

    with torch.no_grad():
        features, risk = model.model(Xs)
        loss = model.training_step((Xs, y))
        preds = model.predict_step((Xs, y))

    assert features[0].shape == (3, 10)
    assert features[1].shape == (4, 10)
    assert risk.shape == (4, 4)
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert preds.shape == (5, len(y))


def test_predictions_to_pycox_adds_initial_survival(sample_data):
    model = estimator(input_dim=[10, 10], hidden_dim=8, embed_size=8, t_binds=3)
    preds = torch.tensor([[0.5, 0.5, 0.25], [0.8, 0.5, 0.5]])

    y_pred = model._predictions_to_pycox(preds)

    expected = torch.tensor(
        [
            [1.0, 1.0],
            [0.5, 0.8],
            [0.25, 0.4],
            [0.0625, 0.2],
        ]
    )
    assert torch.allclose(y_pred, expected)


def test_example():
    from lightning import Trainer
    from imml.load import MultiSurvDataset
    from imml.survival import MultiSurv

    Xs = [pd.DataFrame(np.random.default_rng(42).random((8, 10))) for _ in range(2)]
    y = pd.DataFrame(
        {
            "event": np.random.default_rng(42).integers(0, 2, 8),
            "time": np.random.default_rng(42).uniform(0.5, 5, 8),
        }
    )
    train_data = MultiSurvDataset(Xs=Xs, y=y)
    train_dataloader = DataLoader(dataset=train_data, batch_size=4, shuffle=True)
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False, enable_model_summary=False)
    model = MultiSurv(input_dim=[10, 10], hidden_dim=8, embed_size=8, t_binds=4)
    trainer.fit(model, train_dataloader)
    predictions = trainer.predict(model, train_dataloader)
    assert predictions[0].shape[0] == 5


if __name__ == "__main__":
    pytest.main()
