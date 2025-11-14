"""
===========================================================================================================
Classify an incomplete tabular–language dataset (Synthetic Employee Dataset)
===========================================================================================================

This tutorial illustrates how to classify samples from an incomplete tabular-language dataset using the `iMML` library.
We will use the ``MUSE`` algorithm from the `iMML` classify module on the `Synthetic Employee Dataset
<https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k>`__ and evaluate its performance.

What you will learn:

- How to load a public tabular-language dataset `Synthetic Employee Dataset
  <https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k>`__
  via `Hugging Face Datasets <https://huggingface.co/datasets>`__).
- How to adapt this workflow to your own tabular-language data.
- How to train the ``MUSE`` classifier when tabular or text modalities might be missing.
- How to evaluate the results using the Mathews correlation coefficient (MCC) and a confusion matrix.

"""

# sphinx_gallery_thumbnail_number = 1

# License: BSD 3-Clause License

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial, install the extras for deep learning:
#   pip install imml[deep]
# We also use the Hugging Face Datasets library to load the Synthetic Employee Dataset:
#   pip install datasets


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning import Trainer
from sklearn.metrics import matthews_corrcoef, accuracy_score,  ConfusionMatrixDisplay, RocCurveDisplay
from datasets import load_dataset

from imml.classify import MUSE
from imml.ampute import Amputer
from imml.load import MUSEDataset


################################
# Step 2: Prepare the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will use a synthetic employee dataset (available at `Hugging Face Datasets
# <https://huggingface.co/datasets>`__ as `Synthetic Employee Dataset
# <https://huggingface.co/datasets/BrotherTony/employee-burnout-turnover-prediction-800k>`__

random_state = 42
L.seed_everything(random_state) # Set the seed

ds = load_dataset("BrotherTony/employee-burnout-turnover-prediction-800k",
                  split="train[:1000]") # Retrieve the first 1000 records
df = ds.to_pandas()

df.head()

###################################
# As it can be seen, the dataset contains multiple attributes per record (more than 30). For the illustrative purpose
# of this tutorial, we will use only the numeric attributes, and the `recent_feedback` column, which includes
# employee comments about the working conditions, and which will be our text modality.
# Our response variable is the `left_company` column, which includes the label about whether the employee left the
# company or not. Let us retrieve the data of interest and create the training (80%) and testing (20%) partition:

df = pd.concat([
    df.select_dtypes(include='number'),
    df['recent_feedback'],
    df['left_company'].astype(int),
], axis=1
)
train_df, test_df = train_test_split(df, test_size=0.2, shuffle=True,
                                     stratify=df["left_company"], random_state=random_state)
print("train_df", train_df.shape)
print("test_df", test_df.shape)

###################################################
# Train data
# Independent variables
Xs_train = [
    train_df.iloc[:,0:-2], # Numeric modality
    train_df[['recent_feedback']] # Text modality
]
# Dependent variable
y_train = train_df['left_company'].astype(np.float32) # Response variable

# Test data
# Independent variables
Xs_test = [
    test_df.iloc[:,0:-2], # Numeric modality
    test_df[['recent_feedback']] # Text modality
]
# Dependent variable
y_test = test_df['left_company'].astype(np.float32) # Response variable


###################################################
# Step 3: Simulate missing modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To exemplify the use of ``MUSE`` in a common scenario in which the dataset contains missing modalities, we will
# randomly introduce missing data using ``Amputer``. Using this function we will transform the training and test
# datasets so 10% of samples will have either numeric or text modalities missing.
transformer = Amputer(p=0.1, random_state=random_state)
Xs_train = transformer.fit_transform(Xs_train)
Xs_test = transformer.fit_transform(Xs_test)

########################################################
# Step 4: Training the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We will start by transforming the numeric variables using a `StandardScaler`:
scaler = StandardScaler().set_output(transform="pandas")
Xs_train[0] = scaler.fit_transform(Xs_train[0])
Xs_test[0] = scaler.transform(Xs_test[0])
###################################################
# Now, we will create the loaders.
train_data = MUSEDataset(Xs=Xs_train, y=y_train)
train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
test_data = MUSEDataset(Xs=Xs_test, y=y_test)
test_dataloader = DataLoader(dataset=test_data, batch_size=10, shuffle=False)

########################################################
# We will train the ``MUSE`` model with only 10 epochs for speed, using the
# `Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ library.

trainer = Trainer(max_epochs=10, logger=False, enable_checkpointing=False)
estimator = MUSE(modalities= ["tabular", "text"], # Specify the two types of modalities
                 input_dim=[Xs_train[0].shape[1]])
trainer.fit(estimator, train_dataloader)

########################################################
# Step 5: Evaluation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# After the training, we can evaluate the predictions. Given the small dataset and short training, the model does
# not perform very well, and the resulting probabilities are distributed in just a few of concrete values.
# Therefore, we will do some evaluation to choose the most appropriate probability threshold to assign the classes.

train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=False)
preds_train = trainer.predict(estimator, train_dataloader)
y_pred_train = torch.cat(preds_train)

plt.hist(y_pred_train, bins=15)
plt.show()

########################################################
# As it can be seen, the probabilities are distributed in just three repeated values. We will stablish a threshold
# based on this to assign the classes.

tuned_threshold = 0.31
y_pred_train_labels = (y_pred_train > tuned_threshold).int()
y_train_true = torch.from_numpy(y_train.values).int()

ConfusionMatrixDisplay.from_predictions(y_true=y_train_true, y_pred=y_pred_train_labels)
plt.title("Training Set Evaluation")

print("MCC:", round(matthews_corrcoef(y_true=y_train_true, y_pred=y_pred_train_labels), 2))
print("Accuracy:", round(accuracy_score(y_true=y_train_true, y_pred=y_pred_train_labels), 2))

########################################################
# It is not the best performance, but given the small data, the missing values, and short training, it is expected
# that the model does not offer the best solution.
# 
# Let us evaluate now the performance with the test set:

preds_test = trainer.predict(estimator, test_dataloader)
y_pred_test = torch.cat(preds_test)
y_pred_test_labels = y_pred_test > tuned_threshold
y_test_true = torch.from_numpy(y_test.values).bool()

# Finally, let us plot and print some of the performance metrics:
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Confusion matrix
ConfusionMatrixDisplay.from_predictions(y_true=y_test_true, y_pred=y_pred_test_labels,
                                        ax=axes[0])
axes[0].set_title("Testing Set Evaluation")

# ROC Curve
RocCurveDisplay.from_predictions(y_true=y_test_true, y_pred=y_pred_test_labels,
                                 ax=axes[1])
axes[1].set_title("Testing Set ROC Curve")

# Adjust layout
plt.tight_layout()
plt.show()

print("MCC:", matthews_corrcoef(y_true=y_test_true, y_pred=y_pred_test_labels))
print("Accuracy:", accuracy_score(y_true=y_test_true, y_pred=y_pred_test_labels))

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This tutorial illustrated how to load and prepare a tabular-text dataset for classification using `iMML`. We trained
# a model using the ``MUSE`` algorithm available in `iMML` with 10% of randomly missing text and tabular modalities.
# It is not strange that the classification performance obtained by the model is rather poor, given the small
# subsample of the dataset and the little training of the model, but it serves as an example on how to use `iMML`.
# For a better performance and more reliable results, the full dataset and longer training should be used.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables the use of state-of-the-art algorithms for classification with
# modality incompleteness in tabular-language datasets.

