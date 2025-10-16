"""
=============================================================
Statistics and interaction structure of a multi-modal dataset
=============================================================

A multi-modal dataset can be characterized beyond basic shape information. With `iMML` you can:

- Summarize core properties of each modality (samples, features, completeness).
- Quantify how modalities relate to a target via PID (Partial Information Decomposition):
  Redundancy (shared info), Uniqueness (modality-specific info), and Synergy (info emerging only when modalities are combined).

What you will learn:

- How to describe per‑modality completeness and cross‑modality overlap with ``get_summary``.
- How to compute redundancy, uniqueness, and synergy (PID) with respect to a target using ``pid``.
- How to visualize and interpret PID results.
- How PID generalizes when you have more than two modalities.

This tutorial is fully reproducible and uses a small synthetic dataset. You can easily
replace the data‑loading section with your own data following the same structure.
"""

# sphinx_gallery_thumbnail_number = 2

# License: BSD 3-Clause License

###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import copy
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from imml.statistics import pid
from imml.explore import get_summary
from imml.visualize import plot_pid

###################################
# Step 2: Create or load a multi-modal dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For reproducibility, we generate a small synthetic classification dataset and split the features into two
# modalities (Xs[0], Xs[1]).
#
# Using your own data:
#
# - Represent your dataset as a Python list Xs, one entry per modality.
# - Each Xs[i] should be a 2D array-like (pandas DataFrame or NumPy array) of shape (n_samples, n_features_i).
# - All modalities must refer to the same samples and be aligned by row.

random_state = 42
X, y = make_classification(n_samples=50, random_state=random_state)
# Two modalities: first 10 features and last 10 features
Xs = [X[:, :10], X[:, 10:]]
print("Samples:", len(Xs[0]), "\t", "Modalities:", len(Xs), "\t", "Features:", [X.shape[1] for X in Xs])


###################################################
# Step 3: Summarize the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The ``get_summary`` function provides a compact overview of the multi‑modal dataset. Below we first
# make the dataset a bit more complex by introducing some incomplete samples, then show two views:
# 1) a dataframe aggregated across modalities (one_row=True) and 2) per‑modality counts (one_row=False).

inc_Xs = copy.deepcopy(Xs)
# Introduce block-wise missingness in a few regions for illustration
inc_Xs[0][:20, :] = np.nan
inc_Xs[0][25, 1] = np.nan
inc_Xs[1][18:22, :] = np.nan
inc_Xs[1][-15:, 3] = np.nan

summary = get_summary(Xs=inc_Xs, one_row=True, compute_pct=True, return_df=True)
summary

###################################################
# Per‑modality view:
summary = get_summary(Xs=inc_Xs, modalities=["Modality A", "Modality B"], one_row=False, compute_pct=True, return_df=True)
summary

###################################################
# For quick inspection, we can also plot the per‑modality counts. Here we create a bar chart.

summary.index = summary.index.str.replace(" samples", "")
_ = summary[[c for c in summary.columns if not c.startswith('%')]].plot(
    kind="bar", xlabel="Samples", ylabel="Count", rot=0,
    title="Summary of the multi-modal dataset")


###################################################
# Step 4: Compute PID statistics (Redundancy, Uniqueness, Synergy)
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using ``pid``, we quantify the degree of redundancy, uniqueness, and synergy relating input modalities to the target.
# With two input modalities, ``pid`` returns a dictionary with keys: "Redundancy", "Uniqueness1", "Uniqueness2",
# and "Synergy".

rus = pid(Xs=Xs, y=y, random_state=random_state, normalize=True)
rus  # a dict with keys: Redundancy, Uniqueness1, Uniqueness2, Synergy


###############################################################################
# Step 5: Visualize the PID as a Venn-like diagram
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# You can directly pass the rus dict returned by ``pid`` to ``plot_pid``. Alternatively, ``plot_pid`` can also compute
# PID internally if you pass Xs and y, which is convenient when you want a one‑liner.

rus = {"Redundancy": 0.2, "Synergy": 0.1, "Uniqueness1": 0.45, "Uniqueness2": 0.25}
fig, ax = plot_pid(rus=rus, modalities=["Modality A", "Modality B"], abb=False)

###################################################
# Step 6: Interpreting PID results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# - Redundancy: Information about the target available in both modalities. High values suggest overlap.
# - Uniqueness1/2: Modality‑specific information about the target. High values suggest complementarity.
# - Synergy: Information that emerges only when modalities are combined. High synergy often indicates interactions.
#
# If redundancy is high while uniqueness and synergy are low, this may suggest that the dataset could be more
# appropriately analyzed using classical unimodal modeling.

###################################################
# Step 7: Working with more than two modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# If you have more than two modalities, PID statistics are computed pairwise; ``pid`` returns a list of
# dictionaries (one per pair). For example, adding a third modality yields three pairwise results.
rus = pid(Xs=Xs + [Xs[0]], y=y, random_state=random_state, normalize=True)
rus


###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In this tutorial, we:
#
# - Summarized key per‑modality statistics for a multi‑modal dataset.
# - Quantified redundancy, uniqueness, and synergy with respect to a target using PID.
# - Visualized and interpreted PID, including the multi‑modality (>2) case.
#
# These insights help you understand complementarity and interactions across modalities, informing model design and
# feature engineering for downstream multi‑modal learning.
