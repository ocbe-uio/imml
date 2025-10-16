"""
==================================================
Modality-wise missing data simulation (Amputation)
==================================================

Evaluation and benchmarking of new algorithms or models under diverse conditions is essential to ensure their
robustness, added value and generalizability. `iMML` simplifies this process by simulating incomplete multi-modal
datasets with modality-wise missing data. This so-called data amputation process allows for controlled testing of
methods by introducing missing data from various mechanisms that reflect real-world scenarios where different
modalities may be partially observed or entirely missing.

What you will learn:

- The four amputation (missingness) mechanisms supported by `iMML` (PM, MCAR, MNAR, MEM).
- How to generate modality-wise incomplete multi-modal datasets with ``Amputer``.
- How to visualize missingness patterns across modalities with ``plot_missing_modality``.
- How missingness mechanisms and rates affect per-modality data availability.

Missingness mechanisms:

- Partial missing (PM): some modalities are fully observed for all samples, while others are partially missing at random.
- Missing completely at random (MCAR): missing modalities occur randomly across samples.
- Missing not at random (MNAR): certain samples are missing in specific modalities due to factors that influence missingness.
- Mutually exclusive missing (MEM): incomplete samples have only one observed modality (an extreme case of MNAR).

This tutorial is fully reproducible and uses a small synthetic dataset. You can easily
replace the data-loading section with your own data following the same structure.
"""

# sphinx_gallery_thumbnail_number = 1

# License: BSD 3-Clause License

###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.core.display_functions import display

from imml.ampute import Amputer
from imml.explore import get_summary
from imml.visualize import plot_missing_modality

##########################
# Step 2: Load the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^
# For illustration, we create a small random multi-modal dataset with 10 samples and 4 modalities.
#
# Using your own data:
#
# - Represent your dataset as a Python list Xs, one entry per modality.
# - Each Xs[i] should be a 2D array-like (pandas DataFrame or NumPy array) of shape (n_samples, n_features_i).
# - All modalities must refer to the same samples and be aligned by row.

random_state = 7
n_mods = 4
n_samples = 10
rng = np.random.default_rng(random_state)
Xs = [pd.DataFrame(rng.random((n_samples, 10))) for i in range(n_mods)]


###################################################
# Step 3: Simulate missing data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Using ``Amputer``, we introduce missing data to simulate a scenario where some modalities are missing. Here,
# 80% of the samples will be incomplete following a mutually exclusive missing (MEM) pattern.

mechanism = "mem"
p=0.8
amputer = Amputer(mechanism=mechanism, p=p, random_state=random_state)
transformed_Xs = amputer.fit_transform(Xs)


###################################
# We can visualize which modalities are missing using a binary color map (black = observed, white = missing).
# Each row is a sample; each column is a modality.
_ = plot_missing_modality(Xs=transformed_Xs, figsize= (3.18,2.2))


###################################
# Step 4: Compare amputation mechanisms
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We now illustrate the four different amputation patterns: mutually exclusive missing (MEM),
# partial missing (PM), missing completely at random (MCAR), and missing not at random (MNAR).

mechanism_dict = {"mem": "Mutually exclusive missing",
                  "pm": "Partial missing",
                  "mcar": "Missing completely at random",
                  "mnar": "Missing not at random",
                  }


samples_dict = {}
fig,axs = plt.subplots(1,4, figsize= (12.5,2.5))
for idx, (mechanism, title) in enumerate(mechanism_dict.items()):
    ax = axs[idx]
    transformed_Xs = Amputer(mechanism=mechanism, p=0.8, random_state=random_state).fit_transform(Xs)
    _, ax = plot_missing_modality(Xs=transformed_Xs, ax=ax)
    ax.set_title(title)
    if idx != 0:
        ax.get_yaxis().set_visible(False)
    samples_dict[mechanism_dict[mechanism]] = get_summary(Xs=transformed_Xs, one_row=True)
plt.tight_layout()


###################################################
# As shown in the table below, all cases have the same numbers of complete and incomplete samples overall.
# However, the number of observed samples in each modality varies with the missingness pattern.

pd.DataFrame.from_dict(samples_dict, orient= "index")

###################################
# Step 5: Vary the missingness rate
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Next, we explore how patterns behave as we increase the percentage of incomplete samples. We amputate a
# random multi-modal dataset under each mechanism at different missingness rates.

n_mods = 5
n_samples = 100
Xs = [pd.DataFrame(rng.random((n_samples, 10))) for i in range(n_mods)]
for p in np.arange(0.1, 1., 0.1):
    samples_dict = {}
    fig,axs = plt.subplots(1,4, figsize= (12,2.5))
    for idx, (ax, mechanism) in enumerate(zip(axs, list(mechanism_dict.keys()))):
        transformed_Xs = Amputer(mechanism=mechanism, p=p, random_state=random_state+1).fit_transform(Xs)
        _, ax = plot_missing_modality(Xs=transformed_Xs, ax=ax)
        if p == 0.1:
            ax.set_title(mechanism_dict[mechanism])
        if idx != 0:
            ax.get_yaxis().set_visible(False)
        samples_dict[mechanism] = get_summary(Xs=transformed_Xs, one_row=True)
    plt.tight_layout()

    display(pd.DataFrame.from_dict(samples_dict, orient= "index"))


###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# While all the cases have the same number of complete and incomplete samples, each pattern represents a unique
# distribution of missing data across different modalities, helping researchers to assess the robustness of
# machine learning models in the presence of incomplete data.
