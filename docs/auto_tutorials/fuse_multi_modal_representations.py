# tutorials/fuse_multi_modal_representations.py
"""
====================================================
Fusing incomplete multi-modal representations
====================================================

The ``imml.fuse`` module contains PyTorch modules for combining modality representations, even when not all
modalities are available, after each modality has been encoded into a shared feature space. This tutorial
shows common fusion strategies.

What you will learn:

- How aggregate fusion modules combine list of tensors with shape ``(batch_size, n_features)``.
- How ``ConcatFusion`` differs from aggregate fusion modules.
- How ``AttentionFusion`` learns feature-wise modality weights.
- How ``EmbraceNet`` samples available modalities when some representations are missing.

"""

# sphinx_gallery_thumbnail_number = 1

# License: BSD 3-Clause License

###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import pandas as pd
import torch
import matplotlib.pyplot as plt
import lightning as L

from imml.fuse import AttentionFusion, ConcatFusion, EmbraceNet, MaxFusion, MeanFusion, SumFusion

#############################################
# Step 2: Create modality representations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# In a deep multi-modal model, these tensors would usually be produced by modality-specific encoders. For this
# tutorial, we create small synthetic embeddings directly. A row of NaNs denotes a missing modality for that sample.

random_state = 42
L.seed_everything(random_state)
n_modalities = 3
batch_size = 6
n_features = 4

Xs = [torch.randn(batch_size, n_features) for _ in range(n_modalities)]
Xs[0][2] = torch.nan
Xs[1][4] = torch.nan
Xs[2][1] = torch.nan

###################################
# Step 3: Aggregate fusion
# ^^^^^^^^^^^^^^^^^^^^^^^^
# ``MeanFusion``, ``SumFusion``, and ``MaxFusion`` return one fused representation per sample and preserve the shared
# feature dimension.

aggregate_fusions = {
    "mean": MeanFusion(),
    "sum": SumFusion(),
    "max": MaxFusion(),
}
aggregate_outputs = {name: fusion(Xs) for name, fusion in aggregate_fusions.items()}

for name, output in aggregate_outputs.items():
    print(name, output.shape)

###################################
# Step 4: Concatenation fusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``ConcatFusion`` receives a list of tensors, each with shape ``(batch_size, n_features)``, and concatenates them
# along the feature axis. The output feature dimension is therefore ``n_modalities * n_features``.

concat_fusion = ConcatFusion()
concat_output = concat_fusion(Xs)
print("concat", concat_output.shape)

###################################
# Step 5: Attention fusion
# ^^^^^^^^^^^^^^^^^^^^^^^^
# ``AttentionFusion`` computes feature-wise scores for each modality and applies a softmax across modalities. During
# training, the linear layer can learn which modalities should contribute more strongly to each feature.

attention_fusion = AttentionFusion(n_features=n_features)
attention_output = attention_fusion(Xs)

print("attention", attention_output.shape)

###################################
# Step 6: EmbraceNet fusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``EmbraceNet`` assigns zero selection probability to missing modality rows when ``missing_values=0``. The output is
# stochastic because each feature can be sampled from a different available modality.

embracenet = EmbraceNet()
embrace_output = embracenet(Xs)
print("embracenet", embrace_output.shape)

###################################
# Step 7: Compare fused representations
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# The fused embeddings have different scales and dimensions depending on the chosen strategy. The plot below compares
# the first two fused features for the aggregate methods and EmbraceNet.

fig, ax = plt.subplots(figsize=(6, 4))
plot_dict = {**aggregate_outputs, "attention": attention_output, "embracenet": embrace_output}
for name, output in plot_dict.items():
    values = output.detach().numpy()
    ax.scatter(values[:, 0], values[:, 1], label=name, alpha=0.8)
ax.set_xlabel("Fused feature 0")
ax.set_ylabel("Fused feature 1")
ax.legend()
fig.tight_layout()

###################################
# Conclusion
# ^^^^^^^^^^
# Fusion modules provide interchangeable ways to combine modality representations. Aggregate modules keep the shared
# feature size, concatenation preserves all modality-specific features, attention learns feature-wise weights, and
# EmbraceNet can sample from available modalities when some inputs are missing.

