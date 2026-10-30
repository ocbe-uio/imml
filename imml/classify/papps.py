# License: BSD-3-Clause

import math
import random
import numpy as np

from .. import deepmodule_installed, deepmodule_error, LightningModule, Module

if deepmodule_installed:
    import torch
    from torch import nn, optim
    import torch.nn.functional as F


def _category_contrastive_loss(representations, labels, num_classes):
    if labels.numel() == 0:
        return representations.new_zeros(())

    similarities = representations @ representations.t()
    similarities = similarities - torch.diag_embed(torch.diagonal(similarities))
    one_hot = F.one_hot(labels, num_classes).to(representations.dtype)
    counts = one_hot.sum(dim=0, keepdim=True)
    class_means = similarities @ one_hot
    denominators = (counts - one_hot).expand_as(class_means).clone()
    denominators[denominators == 0] = 1
    class_means = class_means / denominators
    predicted_classes = class_means.argmax(dim=1)
    max_similarities = class_means.max(dim=1).values
    positive_similarities = (class_means * one_hot).sum(dim=1)
    same_class = (labels == predicted_classes).to(representations.dtype)
    return F.relu(same_class + max_similarities - positive_similarities).sum()


def _make_progressive_ratios(class_distribution):
    distribution = class_distribution.detach().cpu().numpy().copy()
    indexed_counts = sorted(enumerate(distribution), key=lambda pair: pair[1])
    max_count = distribution.max()
    if max_count <= 0:
        raise ValueError("class_distribution must contain at least one positive count.")
    ratios = np.asarray([count / max_count for _, count in indexed_counts[::-1]])
    distribution[[index for index, _ in indexed_counts]] = ratios
    return distribution


def _make_progressive_mask(ratios, pseudo_labels, total_epochs, epoch):
    exponent = (total_epochs - epoch) / total_epochs
    selected = [random.random() < ratios[label] ** exponent for label in pseudo_labels.detach().cpu().tolist()]
    return torch.tensor(selected, dtype=torch.bool)


def _estimate_pseudo(class_distribution, probabilities, num_classes, alpha):
    pseudo_targets = torch.zeros_like(probabilities)
    selected_probability = torch.zeros(num_classes)
    for class_index in range(num_classes - 1, -1, -1):
        count = int(alpha * class_distribution[class_index])
        sorted_probabilities, sample_indices = probabilities[:, class_index].sort(descending=True)
        pseudo_targets[sample_indices[:count], class_index] = 1
        selected_probability[class_index] = sorted_probabilities[:count].sum()
    weights = (class_distribution + 1e-6) / (selected_probability + 1e-6)
    return pseudo_targets, weights


def _dual_objective(value, class_probabilities, sample_scales, entropy_weights, target_count):
    return np.sum(class_probabilities * sample_scales * np.exp(-value / entropy_weights)) - target_count


def _opt_solver(probabilities, target_distribution, num_iter=10, tolerance=0.1, newton_iter=30):
    from scipy import optimize

    entropy = -(probabilities * torch.log(probabilities + 1e-6)).sum(dim=1)
    entropy_weights = 1 / (entropy + 1e-10)
    probabilities_np = probabilities.detach().cpu().numpy()
    weights_np = entropy_weights.detach().cpu().numpy()
    target_np = target_distribution.detach().cpu().numpy()
    num_samples, num_classes = probabilities.shape
    scaled_probabilities = probabilities_np / math.e
    sample_scales = np.ones(num_samples)
    dual_variables = np.zeros(num_classes)
    class_scales = np.exp(-dual_variables[None, :] / weights_np[:, None])

    for _ in range(num_iter):
        denominator = np.sum(scaled_probabilities * class_scales, axis=1)
        denominator[denominator == 0] = 1e-10
        sample_scales = 1 / denominator
        next_dual = np.zeros(num_classes)
        for class_index in range(num_classes):
            try:
                next_dual[class_index] = optimize.newton(
                    _dual_objective,
                    dual_variables[class_index],
                    maxiter=newton_iter,
                    args=(scaled_probabilities[:, class_index], sample_scales,
                          weights_np, target_np[class_index]),
                    tol=tolerance,
                )
            except (RuntimeError, FloatingPointError):
                next_dual[class_index] = dual_variables[class_index]
        dual_variables = next_dual
        class_scales = np.exp(-dual_variables[None, :] / weights_np[:, None])

    denominator = np.sum(scaled_probabilities * class_scales, axis=1)
    denominator[denominator == 0] = 1e-10
    sample_scales = 1 / denominator
    return torch.as_tensor(
        scaled_probabilities * sample_scales[:, None] * class_scales,
        dtype=probabilities.dtype,
    )


def _weighted_mse(reconstruction, target, observed):
    weighted_error = observed[:, None] * (target - reconstruction)
    return weighted_error.square().mean()


class _PAPPSViewEncoder(Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        first_dim = round(input_dim * 0.8)
        self.enc_1 = nn.Linear(input_dim, first_dim)
        self.enc_2 = nn.Linear(first_dim, first_dim)
        self.enc_3 = nn.Linear(first_dim, hidden_dim)
        self.z_layer = nn.Linear(hidden_dim, latent_dim)
        self.z_b0 = nn.BatchNorm1d(latent_dim)

    def forward(self, x):
        x = F.relu(self.enc_1(x))
        x = F.relu(self.enc_2(x))
        x = F.relu(self.enc_3(x))
        return self.z_b0(self.z_layer(x))


class _PAPPSViewDecoder(Module):
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        first_dim = round(input_dim * 0.8)
        self.dec_0 = nn.Linear(latent_dim, latent_dim)
        self.dec_1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_2 = nn.Linear(hidden_dim, first_dim)
        self.dec_3 = nn.Linear(first_dim, first_dim)
        self.x_bar_layer = nn.Linear(first_dim, input_dim)

    def forward(self, z):
        x = F.relu(self.dec_0(z))
        x = F.relu(self.dec_1(x))
        x = F.relu(self.dec_2(x))
        x = F.relu(self.dec_3(x))
        return self.x_bar_layer(x)


class _PAPPSAutoencoder(Module):
    def __init__(self, input_dim, latent_dim, num_classes, hidden_dim):
        super().__init__()
        self.encoder_list = nn.ModuleList(
            [_PAPPSViewEncoder(size, latent_dim, hidden_dim) for size in input_dim]
        )
        self.decoder_list = nn.ModuleList(
            [_PAPPSViewDecoder(size, latent_dim, hidden_dim) for size in input_dim]
        )
        self.regression = nn.Linear(latent_dim, num_classes)
        self.act = nn.Sigmoid()

    def forward(self, Xs, observed):
        representations = []
        individual_probabilities = []
        fused = 0
        for view_index, (encoder, x) in enumerate(zip(self.encoder_list, Xs)):
            representation = encoder(x)
            representations.append(representation)
            individual_probabilities.append(
                self.act(self.regression(F.relu(representation)))
            )
            fused = fused + observed[:, view_index:view_index + 1] * representation

        fused = fused / observed.sum(dim=1, keepdim=True)
        reconstructions = [decoder(fused) for decoder in self.decoder_list]
        probabilities = self.act(self.regression(F.relu(fused)))
        return reconstructions, probabilities, fused, representations, individual_probabilities


class _PAPPSNetwork(Module):
    def __init__(self, input_dim, latent_dim, num_classes, hidden_dim):
        super().__init__()
        self.ae = _PAPPSAutoencoder(input_dim, latent_dim, num_classes, hidden_dim)

    def forward(self, Xs, observed):
        return self.ae(Xs, observed)


class PAPPS(LightningModule):
    r"""
    Prototype Alignment and Progressive Pseudo-label Sampling (PA-PPS). [#papps_paper]_ [#papps_code]_

    PA-PPS is a class-imbalance-aware method for classification with incomplete,
    dense numeric modalities and partially observed labels. It combines a shared
    multi-modal autoencoder, per-modality category contrastive loss, class-prototype
    imputation, progressive pseudo-label selection, and DARP pseudo-label
    refinement.

    Each training epoch needs predictions and prototypes over the *entire*
    training set. Supply a dataloader that returns one full-dataset batch
    (``batch_size=len(dataset)``, ``shuffle=False``); PA-PPS splits that batch
    into its own optimization mini-batches. The dataset batch must include the
    global class counts, as provided by :class:`~imml.load.PAPPSDataset`.

    This class provides training, validation, testing, and prediction logic compatible with the
    `Lightning Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

    Parameters
    ----------
    input_dim : list of int
        Number of features in each numeric modality.
    num_classes : int, default=2
        Number of classes, encoded as integers from 0 to ``num_classes - 1``.
    class_distribution : array-like, default=None
        Global sample count for each class. When omitted, it is read from the
        full-dataset batch. These counts are used for progressive sampling and
        DARP, as in the reference implementation.
    latent_dim : int, default=256
        Shared latent representation size.
    hidden_dim : int, default=1500
        Final hidden size in each modality encoder and decoder.
    batch_size : int, default=32
        Internal optimization mini-batch size.
    learning_rate : float, default=0.1
        SGD learning rate.
    beta : float, default=0.1
        Reconstruction-loss coefficient.
    rho : float, default=1e-5
        Category-contrastive-loss coefficient.
    wcl : float, default=0.01
        Unlabeled classification-loss coefficient.
    alpha : float, default=1.5
        DARP pseudo-label selection multiplier.
    darp_epoch : int, default=800
        Start DARP refinement after this zero-based epoch.
    max_epochs : int, default=1000
        Number of training epochs, used for progressive pseudo-label selection.

    Example
    -------
    >>> from lightning import Trainer
    >>> from torch.utils.data import DataLoader
    >>> from imml.classify import PAPPS
    >>> from imml.load import PAPPSDataset
    >>> Xs = [torch.from_numpy(np.random.default_rng(42).random((20, 10))) for i in range(3)]
    >>> dataset = PAPPSDataset(Xs, y)
    >>> loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    >>> model = PAPPS(input_dim=[X.shape[1] for X in Xs], num_classes=3, max_epochs=2)
    >>> Trainer(max_epochs=2).fit(model, loader)

    References
    ----------
    .. [#papps_paper] Prototype alignment and progressive pseudo-label sampling for
                      class-imbalanced incomplete multi-view classification.
    .. [#papps_code] https://github.com/liy1998/PA-PPS

    See Also
    --------
    :class:`~imml.load.PAPPSDataset`
    """

    automatic_optimization = False

    def __init__(self, input_dim, num_classes=2, class_distribution=None, latent_dim=256,
                 hidden_dim=1500, batch_size=32, learning_rate=0.1, beta=0.1,
                 rho=1e-5, wcl=0.01, alpha=1.5, darp_epoch=800, max_epochs=1000):
        if not deepmodule_installed:
            raise ImportError(deepmodule_error)
        if not isinstance(input_dim, list) or not input_dim or any(
            not isinstance(size, int) or size <= 0 for size in input_dim
        ):
            raise ValueError("Invalid input_dim. It must be a non-empty list of positive integers.")
        if not isinstance(num_classes, int) or num_classes < 2:
            raise ValueError("Invalid num_classes. It must be an integer of at least 2.")
        for name, value in (("latent_dim", latent_dim), ("hidden_dim", hidden_dim),
                            ("batch_size", batch_size), ("max_epochs", max_epochs)):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"Invalid {name}. It must be a positive integer.")
        for name, value in (("learning_rate", learning_rate), ("beta", beta), ("rho", rho),
                            ("wcl", wcl), ("alpha", alpha)):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}. It must be a finite non-negative number.")
        if learning_rate == 0:
            raise ValueError("Invalid learning_rate. It must be positive.")
        if not isinstance(darp_epoch, int) or darp_epoch < 0:
            raise ValueError("Invalid darp_epoch. It must be a non-negative integer.")
        if class_distribution is not None:
            distribution = torch.as_tensor(class_distribution, dtype=torch.float32)
            if distribution.shape != (num_classes,) or not torch.isfinite(distribution).all() \
                    or (distribution < 0).any() or (distribution != distribution.round()).any() \
                    or distribution.sum() <= 0:
                raise ValueError("Invalid class_distribution. It must contain one non-negative count per class.")
        else:
            distribution = None

        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.class_distribution = distribution
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.beta = beta
        self.rho = rho
        self.wcl = wcl
        self.alpha = alpha
        self.darp_epoch = darp_epoch
        self.max_epochs = max_epochs
        self.model = _PAPPSNetwork(input_dim, latent_dim, num_classes, hidden_dim)
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Module):
                for child in module.modules():
                    if isinstance(child, nn.Linear):
                        nn.init.xavier_uniform_(child.weight)
                        nn.init.constant_(child.bias, 0.0)
        self._last_pseudo_labels = None
        self._last_selected_mask = None

    def forward(self, Xs, missing_mod_indicator):
        observed = (~missing_mod_indicator.bool()).to(dtype=Xs[0].dtype)
        return self.model(Xs, observed)

    def _unpack_batch(self, batch, require_global_counts=False):
        if len(batch) == 3:
            Xs, y, batch_distribution = batch
        elif len(batch) == 5:
            Xs, y, missing, labeled, batch_distribution = batch
        elif len(batch) == 4:
            Xs, y, missing, labeled = batch
            batch_distribution = self.class_distribution
        else:
            raise ValueError(
                "PA-PPS batches must contain (Xs, y, class_distribution)."
            )
        if len(Xs) != len(self.input_dim):
            raise ValueError(f"Expected {len(self.input_dim)} views, got {len(Xs)}.")
        if any(x.ndim != 2 or x.shape[1] != size for x, size in zip(Xs, self.input_dim)):
            raise ValueError("Each modality must be a 2D tensor with the configured number of features.")
        num_samples = Xs[0].shape[0]
        if any(x.shape[0] != num_samples for x in Xs):
            raise ValueError("All views must contain the same number of samples.")
        if len(batch) == 3:
            missing = torch.stack([torch.isnan(x).all(dim=1) for x in Xs], dim=1)
            labeled = torch.isfinite(y)
        if missing.shape != (num_samples, len(Xs)) or labeled.shape != (num_samples,):
            raise ValueError("Invalid inferred missing-modality or observed-label indicator shape.")
        if missing.bool().all(dim=1).any():
            raise ValueError("Every sample must have at least one observed modality.")
        Xs = [torch.nan_to_num(x.float(), nan=0.0) for x in Xs]
        y = torch.nan_to_num(y.float(), nan=0.0)
        distribution = None
        if batch_distribution is not None:
            batch_distribution = torch.as_tensor(batch_distribution, dtype=torch.float32)
            if batch_distribution.ndim == 2:
                batch_distribution = batch_distribution[0]
            distribution = batch_distribution
        if require_global_counts:
            if distribution is None:
                raise ValueError(
                    "Training requires global class counts; provide class_distribution "
                    "in the batch or on the estimator."
                )
            if distribution.shape != (self.num_classes,):
                raise ValueError("class_distribution must contain one count per class.")
            if not torch.isfinite(distribution).all() or (distribution < 0).any() \
                    or not torch.isclose(
                        distribution.sum(), distribution.new_tensor(float(num_samples))
                    ):
                raise ValueError(
                    "class_distribution must be finite, non-negative, and sum to the dataset size."
                )
            if self.class_distribution is not None and len(batch) != 3 and not torch.equal(
                distribution.cpu(), self.class_distribution.cpu()
            ):
                raise ValueError("Batch class_distribution does not match the estimator's class_distribution.")
        if labeled.any() and ((y[labeled] < 0).any() or (y[labeled] >= self.num_classes).any()):
            raise ValueError("Observed labels must be encoded from 0 to num_classes - 1.")
        if require_global_counts and not labeled.any():
            raise ValueError("PA-PPS training requires at least one observed label.")
        return Xs, y.long(), missing.bool(), labeled.bool(), distribution

    def training_step(self, batch, batch_idx=None):
        Xs, y, missing, labeled, distribution = self._unpack_batch(
            batch, require_global_counts=True
        )
        num_samples = len(y)
        device = Xs[0].device
        Xs = [x.float() for x in Xs]
        observed = (~missing).to(dtype=Xs[0].dtype)
        class_ratios = _make_progressive_ratios(distribution)
        if self.class_distribution is None:
            self.class_distribution = distribution.detach().cpu()

        prototypes, valid_classes = self._get_prototypes(
            Xs, y, labeled, missing, self._last_pseudo_labels, self._last_selected_mask,
            self.num_classes,
        )
        full_pseudo = torch.full(
            (num_samples, self.num_classes), 1 / self.num_classes, dtype=torch.float32
        )
        labeled_cpu = labeled.detach().cpu()
        if labeled_cpu.any():
            full_pseudo[labeled_cpu] = F.one_hot(y[labeled].detach().cpu(), self.num_classes).float()

        optimizer = self.optimizers()
        index_array = np.arange(num_samples)
        np.random.shuffle(index_array)
        epoch_pseudo_labels = torch.zeros(num_samples, dtype=torch.long, device=device)
        epoch_selected_mask = torch.zeros(num_samples, dtype=torch.bool, device=device)
        total_loss = Xs[0].new_zeros(())

        self.model.train()
        for start in range(0, num_samples, self.batch_size):
            indices = index_array[start:start + self.batch_size]
            idx = torch.as_tensor(indices, dtype=torch.long, device=device)
            batch_missing = missing[idx]
            batch_observed = observed[idx]
            X_batch = [x[idx] for x in Xs]
            local_labeled = labeled[idx]
            labeled_positions = torch.nonzero(local_labeled, as_tuple=False).flatten()
            unlabeled_positions = torch.nonzero(~local_labeled, as_tuple=False).flatten()
            unlabeled_indices = idx[unlabeled_positions]

            with torch.no_grad():
                _, pseudo_probabilities, _, _, _ = self.model(
                    X_batch, batch_observed
                )
                pseudo_original = torch.softmax(
                    pseudo_probabilities, dim=1
                )[unlabeled_positions]
                full_pseudo[unlabeled_indices.cpu()] = pseudo_original.detach().cpu()
                if self.current_epoch > self.darp_epoch:
                    pseudo_backup = full_pseudo.clone()
                    pseudo_targets, pseudo_weights = _estimate_pseudo(
                        distribution.cpu(), full_pseudo, self.num_classes, self.alpha
                    )
                    scaled_targets = pseudo_targets * pseudo_weights.reshape(1, -1)
                    reweighted = full_pseudo * scaled_targets + 1e-6
                    reweighted = reweighted / reweighted.sum(dim=1, keepdim=True)
                    refined = _opt_solver(reweighted, distribution.cpu())[unlabeled_indices.cpu()]
                    full_pseudo = pseudo_backup
                else:
                    refined = pseudo_original

                _, pseudo_labels = refined.max(dim=1)
                selected = _make_progressive_mask(
                    class_ratios, pseudo_labels, self.max_epochs, self.current_epoch
                ).to(device)
                current_labels = torch.zeros(len(idx), dtype=torch.long, device=device)
                current_labels[labeled_positions] = y[idx[labeled_positions]]
                current_labels[unlabeled_positions] = pseudo_labels
                epoch_pseudo_labels[unlabeled_indices] = pseudo_labels
                epoch_selected_mask[unlabeled_indices] = selected
                pseudo_one_hot = F.one_hot(pseudo_labels, self.num_classes).to(Xs[0].dtype)

            X_filled = [x.clone() for x in X_batch]
            for sample_position in range(len(idx)):
                class_index = int(current_labels[sample_position])
                if not valid_classes[class_index]:
                    continue
                for view_index in range(len(Xs)):
                    if batch_missing[sample_position, view_index] \
                            and prototypes[view_index][class_index] is not None:
                        X_filled[view_index][sample_position] = prototypes[view_index][class_index]

            reconstructions, probabilities, _, representations, _ = self.model(
                X_filled, torch.ones_like(batch_observed)
            )
            target = F.one_hot(y[idx[labeled_positions]], self.num_classes).to(probabilities.dtype)
            full_target = torch.zeros_like(probabilities)
            full_target[labeled_positions] = target
            unobserved_target = 1 - full_target
            supervised_loss = (
                full_target * torch.log(probabilities + 1e-10)
                + unobserved_target * torch.log(1 - probabilities + 1e-10)
            ).abs().mean()

            if len(unlabeled_positions):
                unlabeled_loss = -(
                    F.log_softmax(probabilities[unlabeled_positions], dim=1)
                    * pseudo_one_hot
                ).sum(dim=1).mul(selected).mean()
            else:
                unlabeled_loss = probabilities.new_zeros(())

            labeled_category_loss = probabilities.new_zeros(())
            for view_index, representation in enumerate(representations):
                visible_labeled = ~batch_missing[labeled_positions, view_index]
                selected_representations = representation[labeled_positions] * visible_labeled[:, None]
                labeled_category_loss = labeled_category_loss + _category_contrastive_loss(
                    selected_representations, y[idx[labeled_positions]], self.num_classes
                )

            pseudo_positions = unlabeled_positions[selected]
            pseudo_category_loss = probabilities.new_zeros(())
            for view_index, representation in enumerate(representations):
                visible_pseudo = ~batch_missing[pseudo_positions, view_index]
                selected_representations = representation[pseudo_positions] * visible_pseudo[:, None]
                pseudo_category_loss = pseudo_category_loss + _category_contrastive_loss(
                    selected_representations, current_labels[pseudo_positions], self.num_classes
                )

            reconstruction_loss = sum(
                _weighted_mse(reconstruction, original, batch_observed[:, view_index])
                for view_index, (reconstruction, original) in enumerate(zip(reconstructions, X_batch))
            )
            if self.current_epoch > self.darp_epoch:
                loss = (
                    supervised_loss
                    + self.beta * reconstruction_loss
                    + self.rho * (labeled_category_loss + 0.01 * self.wcl * pseudo_category_loss)
                    + self.wcl * unlabeled_loss
                )
            else:
                loss = supervised_loss + self.beta * reconstruction_loss + self.rho * labeled_category_loss

            optimizer.zero_grad()
            self.manual_backward(loss)
            optimizer.step()
            total_loss = total_loss + loss.detach()

        self._last_pseudo_labels = epoch_pseudo_labels.detach()
        self._last_selected_mask = epoch_selected_mask.detach()
        self.log("train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True,
                 batch_size=num_samples)
        return total_loss

    def _get_prototypes(self, Xs, y, labeled, missing, pseudo_labels, selected_mask, num_classes):
        valid_classes = [False] * num_classes
        prototypes = [[None] * num_classes for _ in Xs]
        for class_index in range(num_classes):
            members = labeled & (y == class_index)
            if pseudo_labels is not None and selected_mask is not None:
                members = members | (selected_mask & (pseudo_labels == class_index))
            if not members.any():
                continue
            valid_classes[class_index] = True
            for view_index, features in enumerate(Xs):
                available_members = members & ~missing[:, view_index]
                if available_members.any():
                    prototypes[view_index][class_index] = features[available_members].mean(dim=0)
        return prototypes, valid_classes

    def _inference(self, batch):
        Xs, _, missing, _, _ = self._unpack_batch(batch)
        Xs = [x.float() for x in Xs]
        _, probabilities, _, _, _ = self.model(Xs, (~missing).to(Xs[0].dtype))
        return probabilities

    def validation_step(self, batch, batch_idx=None):
        Xs, y, missing, labeled, _ = self._unpack_batch(batch)
        if not labeled.any():
            return Xs[0].new_zeros(())
        probabilities = self._inference(batch)
        targets = F.one_hot(y[labeled].long(), self.num_classes).to(probabilities.dtype)
        return (
            targets * torch.log(probabilities[labeled] + 1e-10)
            + (1 - targets) * torch.log(1 - probabilities[labeled] + 1e-10)
        ).abs().mean()

    def test_step(self, batch, batch_idx=None):
        return self.validation_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx=None):
        return self._inference(batch)

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)


__all__ = ["PAPPS"]
