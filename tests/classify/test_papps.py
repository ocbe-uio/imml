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
from imml.classify import PAPPS
from imml.load import PAPPSDataset
from tests._sample_data import make_sample_data

L.seed_everything(42)
estimator = PAPPS
sample_data = make_sample_data(n_samples=25, n_features=10, splits=(3, 5))


def _make_labels(n_samples, missing=False):
    y = np.array([0] * (n_samples * 3 // 4) + [1] * (n_samples - n_samples * 3 // 4),
                 dtype=float)
    if missing:
        y[::5] = np.nan
    return y


def _fit_model(Xs, y, max_epochs=2):
    dataset = PAPPSDataset(Xs=Xs, y=y)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    model = estimator(
        input_dim=[X.shape[1] for X in Xs],
        num_classes=2,
        latent_dim=8,
        hidden_dim=16,
        batch_size=len(dataset),
        max_epochs=max_epochs,
        darp_epoch=0,
    )
    trainer = Trainer(max_epochs=max_epochs, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)
    return model, trainer, dataloader


def test_deepmodule_not_installed():
    estimator(input_dim=[2, 2])
    with patch.dict(sys.modules, {"torch": None}):
        import imml as imml_mock
        import imml.classify.papps as module_mock
        importlib.reload(imml_mock)
        importlib.reload(module_mock)
        with pytest.raises(ImportError, match="Module 'deep' needs to be installed."):
            estimator(input_dim=[2, 2])
    importlib.reload(imml_mock)
    importlib.reload(module_mock)


@pytest.mark.parametrize("Xs", sample_data)
def test_default_params(Xs):
    y = _make_labels(len(Xs[0]))
    model, trainer, dataloader = _fit_model(Xs, y)
    predictions = trainer.predict(model, dataloader)[0]
    batch = next(iter(dataloader))

    assert predictions.shape == (len(y), 2)
    assert torch.isfinite(predictions).all()
    assert torch.all((predictions >= 0) & (predictions <= 1))
    assert torch.isfinite(model.validation_step(batch))
    assert torch.isfinite(model.test_step(batch))
    assert model._last_pseudo_labels.shape == (len(y),)
    assert isinstance(model.configure_optimizers(), torch.optim.Optimizer)


def test_invalid_params():
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator(input_dim=[])
    with pytest.raises(ValueError, match="Invalid input_dim."):
        estimator(input_dim="invalid")
    with pytest.raises(ValueError, match="Invalid num_classes."):
        estimator(input_dim=[2, 3], num_classes=1)
    with pytest.raises(ValueError, match="Invalid latent_dim."):
        estimator(input_dim=[2, 3], latent_dim=-1)
    with pytest.raises(ValueError, match="Invalid hidden_dim."):
        estimator(input_dim=[2, 3], hidden_dim=None)
    with pytest.raises(ValueError, match="Invalid batch_size."):
        estimator(input_dim=[2, 3], batch_size=0)
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(input_dim=[2, 3], learning_rate=-1.0)
    with pytest.raises(ValueError, match="Invalid class_distribution."):
        estimator(input_dim=[2, 3], num_classes=2, class_distribution=[1, -1])
    with pytest.raises(ValueError, match="Invalid class_distribution."):
        estimator(input_dim=[2, 3], num_classes=2, class_distribution=[1, 1.5])
    with pytest.raises(ValueError, match="Invalid class_distribution."):
        estimator(input_dim=[2, 3], num_classes=2, class_distribution=[0, 0])
    with pytest.raises(ValueError, match="Invalid learning_rate."):
        estimator(input_dim=[2, 3], learning_rate=0.0)
    with pytest.raises(ValueError, match="Invalid darp_epoch."):
        estimator(input_dim=[2, 3], darp_epoch=-1)
    for name in ("latent_dim", "hidden_dim", "batch_size", "max_epochs"):
        with pytest.raises(ValueError, match=f"Invalid {name}."):
            estimator(input_dim=[2, 3], **{name: 0})
    for name in ("beta", "rho", "wcl", "alpha"):
        with pytest.raises(ValueError, match=f"Invalid {name}."):
            estimator(input_dim=[2, 3], **{name: -1.0})


def test_reference_components():
    from imml.classify.papps import (
        _category_contrastive_loss,
        _estimate_pseudo,
        _make_progressive_mask,
        _make_progressive_ratios,
        _opt_solver,
        _weighted_mse,
    )

    ratios = _make_progressive_ratios(torch.tensor([2.0, 8.0, 4.0]))
    assert ratios.tolist() == pytest.approx([1.0, 0.25, 0.5])

    reconstructed = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.zeros_like(reconstructed)
    observed = torch.tensor([1.0, 0.0])
    expected = torch.tensor([1.0, 2.0, 0.0, 0.0]).square().mean()
    assert torch.equal(_weighted_mse(reconstructed, target, observed), expected)

    probabilities = torch.tensor([[0.8, 0.2], [0.3, 0.7], [0.6, 0.4]])
    distribution = torch.tensor([2.0, 1.0])
    pseudo_targets, weights = _estimate_pseudo(distribution, probabilities, 2, 0.5)
    assert pseudo_targets.shape == probabilities.shape
    assert weights.shape == (2,)
    refined = _opt_solver(probabilities, distribution, num_iter=2)
    assert refined.shape == probabilities.shape
    assert torch.isfinite(refined).all()
    assert _category_contrastive_loss(torch.empty(0, 4), torch.empty(0, dtype=torch.long), 2) == 0

    with pytest.raises(ValueError, match="positive count"):
        _make_progressive_ratios(torch.zeros(2))

    random_values = iter([0.1, 0.9])
    with patch("imml.classify.papps.random.random", side_effect=lambda: next(random_values)):
        selected = _make_progressive_mask(np.array([0.5, 1.0]), torch.tensor([0, 1]), 2, 0)
    assert selected.tolist() == [True, True]
    random_values = iter([0.9, 0.9])
    with patch("imml.classify.papps.random.random", side_effect=lambda: next(random_values)):
        selected = _make_progressive_mask(np.array([0.5, 1.0]), torch.tensor([0, 1]), 2, 0)
    assert selected.tolist() == [False, True]


def test_darp_solver_retains_previous_iterate(monkeypatch):
    from scipy import optimize
    from imml.classify.papps import _opt_solver

    def fail_newton(*args, **kwargs):
        raise RuntimeError("non-convergent")

    monkeypatch.setattr(optimize, "newton", fail_newton)
    probabilities = torch.zeros(2, 2)
    refined = _opt_solver(probabilities, torch.tensor([1.0, 1.0]), num_iter=1)
    assert torch.isfinite(refined).all()


def test_batch_validation_and_forward():
    model = estimator(
        input_dim=[4, 5],
        num_classes=2,
        class_distribution=[2, 2],
        latent_dim=4,
        hidden_dim=8,
        batch_size=4,
        max_epochs=2,
    )
    Xs = [torch.rand(4, 4), torch.rand(4, 5)]
    y = torch.tensor([0.0, 0.0, 1.0, 1.0])
    distribution = torch.tensor([2.0, 2.0])
    missing = torch.zeros(4, 2, dtype=torch.bool)
    labeled = torch.ones(4, dtype=torch.bool)
    standard_batch = (Xs, y, distribution)

    unpacked = model._unpack_batch(standard_batch)
    assert unpacked[2].shape == (4, 2)
    legacy_four_batch = (Xs, y, missing, labeled)
    assert torch.equal(model._unpack_batch(legacy_four_batch, True)[4], distribution)
    legacy_five_batch = (Xs, y, missing, labeled, distribution)
    assert torch.equal(model._unpack_batch(legacy_five_batch, True)[4], distribution)

    with torch.no_grad():
        outputs = model(Xs, missing)
    assert outputs[1].shape == (4, 2)

    invalid_batches = [
        ((Xs[:1], y, distribution), "Expected 2 views"),
        (([torch.rand(4, 4), torch.rand(4, 6)], y, distribution), "configured number"),
        (([torch.rand(3, 4), Xs[1]], y[:3], distribution), "same number"),
        ((Xs, y, torch.zeros(3, 2), labeled, distribution), "indicator shape"),
        ((Xs, y, torch.ones(4, 2), labeled, distribution), "at least one observed"),
        ((Xs, torch.tensor([0.0, 1.0, 2.0, 1.0]), distribution), "encoded"),
        ((Xs, torch.full((4,), float("nan")), distribution), "at least one observed label"),
        ((Xs, y, torch.tensor([1.0, 1.0])), "sum to the dataset size"),
        ((Xs, y, missing, labeled, torch.tensor([1.0, 3.0])), "does not match"),
        ((Xs, y, torch.tensor([-1.0, 5.0])), "finite, non-negative"),
    ]
    for invalid_batch, message in invalid_batches:
        with pytest.raises(ValueError, match=message):
            model._unpack_batch(invalid_batch, require_global_counts=True)

    with pytest.raises(ValueError, match="must contain"):
        model._unpack_batch((Xs, y))
    with pytest.raises(ValueError, match="requires global class counts"):
        estimator(input_dim=[4, 5], num_classes=2)._unpack_batch(
            (Xs, y, missing, labeled), require_global_counts=True
        )
    with pytest.raises(ValueError, match="one count per class"):
        model._unpack_batch((Xs, y, torch.ones(3)), require_global_counts=True)


def test_missing_prototypes_and_unlabeled_validation():
    model = estimator(input_dim=[4, 5], num_classes=2, latent_dim=4, hidden_dim=8)
    Xs = [torch.rand(4, 4), torch.rand(4, 5)]
    labels = torch.tensor([0, 0, 0, 0])
    labeled = torch.ones(4, dtype=torch.bool)
    missing = torch.zeros(4, 2, dtype=torch.bool)
    prototypes, valid_classes = model._get_prototypes(
        Xs, labels, labeled, missing, None, None, 2
    )
    assert valid_classes == [True, False]
    assert prototypes[0][1] is None

    unlabeled_batch = (
        Xs,
        torch.full((4,), float("nan")),
        torch.tensor([4.0, 0.0]),
    )
    assert model.validation_step(unlabeled_batch) == 0


@pytest.mark.parametrize("valid_classes", ([False, False], [True, True]))
def test_training_when_prototypes_are_unavailable(valid_classes, monkeypatch):
    Xs = [torch.rand(8, 4), torch.rand(8, 5)]
    Xs[0][0] = torch.nan
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    dataset = PAPPSDataset(Xs=Xs, y=y)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    model = estimator(
        input_dim=[4, 5],
        num_classes=2,
        latent_dim=4,
        hidden_dim=8,
        batch_size=8,
        max_epochs=1,
    )
    model._get_prototypes = lambda views, *args: (
        [[None, None] for _ in views],
        valid_classes,
    )
    trainer = Trainer(max_epochs=1, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)


@pytest.mark.parametrize("Xs", sample_data)
def test_missing_values_handling(Xs):
    Xs = Amputer(p=0.3, random_state=42).fit_transform(Xs)
    y = _make_labels(len(Xs[0]), missing=True)
    model, trainer, dataloader = _fit_model(Xs, y)

    predictions = trainer.predict(model, dataloader)[0]
    dataset = dataloader.dataset
    assert predictions.shape == (len(y), 2)
    assert torch.isfinite(predictions).all()
    assert dataset.missing_mod_indicator.any()
    assert (~dataset.y_indicator).any()


def test_example():
    Xs = [
        torch.from_numpy(np.random.default_rng(42).random((20, 10)).astype(np.float32))
        for _ in range(3)
    ]
    y = np.random.default_rng(42).integers(0, 3, len(Xs[0]))
    dataset = PAPPSDataset(Xs=Xs, y=y)
    dataloader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False)
    model = estimator(
        input_dim=[X.shape[1] for X in Xs],
        num_classes=3,
        latent_dim=8,
        hidden_dim=16,
        batch_size=len(dataset),
        max_epochs=2,
    )
    trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
    trainer.fit(model, dataloader)
    predictions = trainer.predict(model, dataloader)[0]

    assert predictions.shape == (len(y), 3)
    assert torch.isfinite(predictions).all()


if __name__ == "__main__":
    pytest.main()
