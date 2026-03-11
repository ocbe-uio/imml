# tutorials/classify_incomplete_vision_language.py
"""
===========================================================================================================
Classify an incomplete vision–language dataset (Oxford‑IIIT Pets) with deep learning
===========================================================================================================

This tutorial demonstrates how to classify samples from an incomplete vision–language dataset using the `iMML`
library. `iMML` supports robust classification even when some modalities (e.g., text or image) are missing, making it
suitable for real‑world multi‑modal data where missingness is common.

We will use the ``RAGPT`` algorithm from the `iMML` classify module on the `Oxford‑IIIT Pets
<https://huggingface.co/datasets/visual-layer/oxford-iiit-pet-vl-enriched?library=datasets>`__  dataset
and evaluate its performance.

What you will learn:

- How to load a public vision–language dataset
  (`Oxford‑IIIT Pets <https://huggingface.co/datasets/visual-layer/oxford-iiit-pet-vl-enriched?library=datasets>`_
  via `Hugging Face Datasets <https://huggingface.co/datasets>`__).
- How to adapt this workflow to your own vision–language data.
- How to build a retrieval‑augmented memory bank and prompts with ``MCR``.
- How to train the ``RAGPT`` classifier when image or text may be missing.
- How to track metrics during training and evaluate with MCC and a confusion matrix.
"""

# sphinx_gallery_thumbnail_number = 1

# License: BSD 3-Clause License

###################################
# Step 0: Prerequisites
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To run this tutorial, install the extras for deep learning:
#   pip install imml[deep]
# We also use the Hugging Face Datasets library to load Oxford‑IIIT Pets:
#   pip install datasets


###################################
# Step 1: Import required libraries
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

import shutil
from PIL import Image
from lightning import Trainer
import lightning as L
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
import torch
import os
import pandas as pd
from sklearn.metrics import matthews_corrcoef, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from datasets import load_dataset

from imml.ampute import Amputer
from imml.classify import RAGPT
from imml.load import RAGPTDataset, RAGPTCollator
from imml.model_selection import train_test_mm_split
from imml.retrieve import MCR

################################
# Step 2: Prepare the dataset
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We use the oxford-iiit-pet-vl-enriched dataset, a public vision–language dataset with images and captions
# available on `Hugging Face Datasets
# <https://huggingface.co/datasets>`__ as visual-layer/oxford-iiit-pet-vl-enriched. For retrieval, we will use
# the ``MCR`` class from the retrieve module.

random_state = 42
L.seed_everything(random_state)

# Local working directory (images will be saved here so MCR can read paths)
data_folder = "oxford_iiit_pet"
folder_images = os.path.join(data_folder, "imgs")
os.makedirs(folder_images, exist_ok=True)

# Load the dataset
ds = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched", split="train[:40]")

# Build a DataFrame with image paths and captions. We persist images to disk because
# the retriever expects paths.
n_total = len(ds)
rows = []
for i in range(n_total):
    ex = ds[i]
    img = ex.get("image", None)
    caption = ex.get("caption_enriched", None)
    label = ex.get("label_cat_dog", None)
    img_path = os.path.join(folder_images, f"{i:06d}.jpg")
    try:
        img.save(img_path)
    except Exception:
        img.convert("RGB").save(img_path)
    rows.append({"img": img_path, "text": caption, "label": label})

df = pd.DataFrame(rows)
le = LabelEncoder()
df["class"] = le.fit_transform(df["label"])
df["class"].value_counts()

###################################
# Split into 40% bank memory, 40% train and 20% test sets
Xs = [df[["img"]],df[["text"]]]
y = df["class"]
Xs_train, Xs_test, y_train, y_test = train_test_mm_split(Xs, y, test_size=0.2,
                                                         shuffle=True, stratify=y,
                                                         random_state=random_state)
Xs_train, Xs_bank, y_train, y_bank = train_test_mm_split(Xs_train, y_train, test_size=0.5,
                                                         shuffle=True, stratify=y_train,
                                                         random_state=random_state)
print("Xs_train", Xs_train[0].shape)
print("Xs_test", Xs_test[0].shape)
print("Xs_bank", Xs_bank[0].shape)

###################################################
# Step 3: Simulate missing modalities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# To reflect realistic scenarios, we randomly introduce missing data using ``Amputer``. In this case, 60% of training
# and test samples will have either text or image missing. You can change this parameter for more or less
# amount of incompleteness.
amputer = Amputer(p=0.6, random_state=random_state)
Xs_train = amputer.fit_transform(Xs_train)
Xs_test = amputer.transform(Xs_test)


########################################################
# Step 4: Generate the prompts using a retriever
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``RAGPT`` needs prompts, which are created from a memory bank with a retriever.
# We use ``MCR`` (Multi-Channel Retriever) to construct a memory bank and generate prompts.

modalities = ["image", "text"]
batch_size = 64
estimator = MCR(batch_size=batch_size, modalities=modalities, save_memory_bank=True,
                prompt_path=data_folder, n_neighbors=2, generate_cap=True)
estimator.fit(Xs=Xs_bank, y=y_bank)
memory_bank = estimator.memory_bank_
print("memory_bank", memory_bank.shape)
memory_bank.info()

########################################################
# Load generated training and testing prompts.
train_db = estimator.transform(Xs=Xs_train, y=y_train)
print("train_db", train_db.shape)
train_db.head()

test_db = estimator.transform(Xs=Xs_test, y=y_test)
print("test_db", test_db.shape)


########################################################
# Step 5: Training the model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Create the loaders.
train_data = RAGPTDataset(database=train_db)
train_dataloader = DataLoader(dataset= train_data, batch_size=batch_size,
                              collate_fn= RAGPTCollator(), shuffle=True)

test_data = RAGPTDataset(database=test_db)
test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size,
                             collate_fn=RAGPTCollator(), shuffle=False)

########################################################
# Train the ``RAGPT`` model using the generated prompts. For speed in this demo we train for only 2 epochs using
# the `Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ library.
trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
estimator = RAGPT()
trainer.fit(estimator, train_dataloader)

########################################################
# Step 6: Advanced Usage: Track Metrics During Training
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# As any other model in `Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_, we can
# modify the internal functions. For instance, we can track loss and compute evaluation metrics during training.

trainer = Trainer(max_epochs=2, logger=False, enable_checkpointing=False)
estimator = RAGPT()
estimator.loss_list = []
estimator.agg_loss_list = []
validation_step = estimator.validation_step

def compute_metric(*args):
    loss = validation_step(*args)
    estimator.loss_list.append(loss)
    return loss
estimator.validation_step = compute_metric

def agg_metric(*args):
    estimator.agg_loss_list.append(torch.stack(estimator.loss_list).mean())
    estimator.loss_list = []
estimator.on_validation_epoch_end = agg_metric

trainer.fit(estimator, train_dataloader, test_dataloader)

########################################################
# Step 7: Evaluation
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# After training, we can evaluate predictions and visualize the results.
preds = trainer.predict(estimator, test_dataloader)
preds = [batch.softmax(dim=1) for batch in preds]
preds = [pred for batch in preds for pred in batch]
preds = torch.stack(preds).argmax(1).cpu()
losses = [i.item() for i in estimator.agg_loss_list]

nrows, ncols = 2,3
test_df = pd.concat(Xs_test, axis=1)
test_df = pd.concat([test_df, y_test.to_frame("label")], axis=1)
test_df = test_df.reset_index(drop=True)
preds = preds[test_df.index]
fig, axes = plt.subplots(nrows, ncols, constrained_layout=True)
for i, (i_row, row) in enumerate(test_df.sample(n=nrows*ncols, random_state=random_state).iterrows()):
    pred = preds[i_row]
    image_to_show = row["img"]
    caption = row["text"]
    real_class = le.classes_[row["label"]]
    ax = axes[i//ncols, i%ncols]
    ax.axis("off")
    if isinstance(image_to_show, str):
        image_to_show = Image.open(image_to_show).resize((512, 512), Image.Resampling.LANCZOS)
        ax.imshow(image_to_show)
    else:
        ax.plot(0.5, 0.5, 'rx', markersize=100, markeredgewidth=10)
    pred_class = le.classes_[pred]
    c = "green" if pred_class == real_class else "red"
    ax.set_title(f"Pred:{pred_class}; Real:{real_class}", **{"color":c})
    if isinstance(caption, str):
        caption = caption.split(" ")
        if len(caption) >=6:
            caption = caption[:len(caption)//2] + ["\n"] + caption[len(caption)//2:]
            caption = " ".join(caption)
        ax.annotate(caption, xy=(0.5, -0.08), xycoords='axes fraction', ha='center', va='top')
    else:
        ax.annotate("X", xy=(0.5, -0.08), xycoords='axes fraction', ha='center', va='top', color="red", fontsize=30)

shutil.rmtree(data_folder, ignore_errors=True)

#######################################################

ConfusionMatrixDisplay.from_predictions(y_true=y_test, y_pred=preds)
print("Testing metric:", round(matthews_corrcoef(y_true=y_test, y_pred=preds), 2))

#######################################################
# Despite using only 40 instances and minimal training, the performance was excellent thanks to the pretrained models.

###################################
# Summary of results
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We first built a memory bank with 40% independent vision-language samples using the `iMML` retrieve module to
# generate retrieval-augmented prompts with a multi-channel retriever (``MCR``). Subsequently, we trained a model
# using the ``RAGPT`` algorithm available in `iMML` under 25% randomly missing text and image modalities. The model
# demonstrated strong robustness on the test set.
#
# This example is intentionally simplified, using only 40 instances for demonstration.
# For stronger performance and more reliable results, the full dataset and longer training should be used.

###################################
# Conclusion
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This example illustrates how `iMML` enables state-of-the-art performance in classification, even in the presence
# of significant modality incompleteness in vision-language datasets.

