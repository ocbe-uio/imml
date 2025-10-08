Contributing to `iMML`
=========================

(adapted from `Scikit-learn
<https://scikit-learn.org/stable/>`__ and `mvlearn
<https://mvlearn.github.io/>`__)

Submitting a bug report or a feature request
--------------------------------------------

We track bugs and feature requests using GitHub issues. If you encounter a problem or have an idea for a new feature,
feel free to open an issue.

If you run into any trouble while using this package, we encourage you to submit an issue through our Issue
Tracker `Bug Tracker <https://github.com/ocbe-uio/imml/issues>`_. Suggestions for enhancements or pull requests are also welcome.

Before posting, ensure your submission aligns with these guidelines:

-  Check for duplicates: Look for similar `issues <https://github.com/ocbe-uio/imml/issues>`_ or `pull requests <https://github.com/ocbe-uio/imml/pulls>`_ already submitted.
-  Bug reports: Follow our tips in :ref:`filing_bugs` to provide all necessary details.
-  Adhere to guidelines: Ensure your code complies with the general :ref:`guidelines` and matches
   the :ref:`api_of_imml_objects`.

.. _filing_bugs:

How to make a good bug report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When reporting an issue on GitHub `Github <https://github.com/ocbe-uio/imml/issues>`_, please include the following details to help us assist you effectively:

-  Minimal code example: Provide a concise code snippet to replicate the issue (see
   `this <https://stackoverflow.com/help/mcve>`_ for more details). If it is longer than around 50 lines,
   use a gist `gist <https://gist.github.com>`_ or a public repository.
-  Key details: If providing code is not practical, specify the methods or modules involved and the data shapes
   you are working with.
-  Errors and tracebacks: Include the full error traceback if applicable.
-  Environment information: Include your operating system, Python version, and the version of this package. Run
   this snippet to collect the details:

   .. code-block:: python

      import platform; print(platform.platform());
      import sys; print("Python", sys.version);
      import imml; print("imml", imml.version)

-  Formatting: Use appropriate code blocks for examples and errors. Refer to the guide on
   `Creating and highlighting code blocks <https://help.github.com/articles/creating-and-highlighting-code-blocks>`_.

Contributing code
-----------------

We recommend the following workflow for contributing code:

1. Use the ‘Fork’ button in the GitHub interface to copy the project into your account. This
   creates a copy of the code under your GitHub user account. For more details on how to fork a repository see
   `this guide <https://help.github.com/articles/fork-a-repo/>`_.

2. Clone your fork locally:

    .. code:: bash

        git clone git@github.com:YourLogin/imml.git
        cd imml


3. Create a ``feature`` branch for your changes:

    .. code:: bash

        git checkout -b my-feature

   Avoid working on the ``main`` branch directly.


4. Develop the feature on your feature branch and commit your changes:

    .. code:: bash

        git add modified_files
        git commit

   Then push the changes to your GitHub account with:

    .. code:: bash

        git push -u origin my-feature

Pull Request Checklist
~~~~~~~~~~~~~~~~~~~~~~

Before submitting a pull request, ensure:

-  Follow the `coding-guidelines <#guidelines>`__.

-  Descriptive title: Use a meaningful title summarizing your contribution.

-  Documentation: Add informative docstrings, including examples when necessary.

-  At least one paragraph of narrative documentation with links to references in the literature and the example.

-  Tests: Provide unit tests to validate functionality and type correctness.

-  Local Testing: Ensure all tests pass locally using ``pytest``. Install dependencies:

    .. code:: bash

        pip install imml[tests]

   then run

    .. code:: bash

        pytest

   or you can run pytest on a single test file by

    .. code:: bash

        pytest path/to/test.py

.. _guidelines:

Guidelines
----------

Coding Guidelines
~~~~~~~~~~~~~~~~~

Consistently formatted code improves readability and maintainability.

Docstring Guidelines
~~~~~~~~~~~~~~~~~~~~

Properly formatted docstrings are essential for documentation generation. Follow the conventions outlined in
`numpydoc <https://numpydoc.readthedocs.io/en/latest/format.html#overview>`__. Refer to the
`example.py <https://numpydoc.readthedocs.io/en/latest/example.html#example>`__ provided by numpydoc.

.. _api_of_imml_objects:

API of `iMML` Objects
----------------------

Estimators
~~~~~~~~~~

The core components of `iMML` are the estimators, designed to train on datasets. These objects follow the conventions
established by `Scikit-learn
<https://scikit-learn.org/stable/>`__ and `mvlearn
<https://mvlearn.github.io/>`__. Estimators inherit from ``sklearn.base.BaseEstimator`` and adhere to its
established guidelines.

To ensure compatibility, developers should align with `Scikit-learn
<https://scikit-learn.org/stable/>`__'s standards whenever possible, including using
validation checks like ``check_Xs`` in ``imml.utils``, to confirm the suitability of the input data.

Instantiation
^^^^^^^^^^^^^

An estimator's ``__init__`` method defines its configuration by accepting constants that influence the behavior
of its methods. These constants should not include actual data or any values derived from it, as data handling is
left to the ``fit`` method. Key points for implementing the ``__init__`` method:

-  All parameters must be keyword arguments with default values.
-  Each parameter should be assigned as an instance attribute.
-  Input validation should not occur during initialization.
-  Randomness control. For stochastic estimators, include a random_state parameter to ensure reproducibility. The
   same seed (random_state) should always yield identical outputs for the same data.The random_state parameter can
   accept:
   - An int, to produce consistent results across runs.
   - None, for non-deterministic results.

A correct implementation of ``__init__`` looks like:

.. code:: python

    def __init__(self, param1=1, param2=2, random_state=None):
        self.param1 = param1
        self.param2 = param2
        self.random_state = random_state

Fitting
^^^^^^^

Estimators must provide a ``fit(Xs, y=None)`` method to process data. This method is invoked as:

.. code:: python

        estimator.fit(Xs, y)

or

.. code:: python

        estimator.fit(Xs)

Parameters:

-  Xs: A list of (pd.DataFrame or np.ndarray) data matrices, with each matrix representing a different modality.
    - Xs length: n_mods
    - Xs[i] shape: (n_samples, n_features_i)

-  y: Array of labels, shape (n_samples,).

-  kwargs: Optional parameters.

The samples across modalities in Xs and y are matched. Note that data matrices in Xs must have the same number of
samples (rows) but the number of features (columns) may differ. If a value (feature or modality) is missing, it should
be represented as np.nan.

The ``fit`` method should return the instance itself (self) to support chaining operations.

Transformers
^^^^^^^^^^^^

A ``transformer`` modify data using the transform method. An estimator may also be a transformer that learns the
transformation parameters. The transformer object implements the ``transform`` method, i.e.

.. code:: python

    Xs_transformed = transformer.transform(Xs)

This is typically called after fitting the transformer. Alternatively, the ``transform`` method combines both steps:

.. code:: python

    Xs_transformed = transformer.fit_transform(Xs, y)

Transformers in `iMML` should be designed to work seamlessly with lists of both ``pandas.DataFrame`` and
``numpy.ndarray``. The input type should dictate the output type. For instance:

-  If the input is a list of pandas.DataFrame, the transformer should return a list of ``pandas.DataFrame`` or a
   single ``pandas.DataFrame``.
-  If the input is a list of numpy.ndarray, the transformer should return a list of ``numpy.ndarray`` or a
   single ``numpy.ndarray``.

Predictors
^^^^^^^^^^
A ``predictor`` generate predictions from the input data via the ``predict`` method:

.. code:: python

    y_predicted = predictor.predict(Xs)

Like transformers, predictors can combine fitting and prediction using the ``fit_predict`` method:

.. code:: python

    y_predicted = predictor.fit_predict(Xs, y)

Deep Learning
~~~~~~~~~~~~~

Currently, repositories offering deep learning methods lack a unified convention. To address this, `iMML` adopts
the `Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ library, which provides a
structured and flexible framework for implementing deep learning models. By standardizing deep learning methods in
`iMML` using theLightning library, we ensure that all implementations are robust, reproducible, and easy to extend.

Dataset defition
^^^^^^^^^^^^^^^^

In this framework, datasets are defined by creating a class that inherits from
`torch.utils.data.Dataset <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_ . This class
should accept input parameters such as Xs (multi-modal dataset) and transform (optional data preprocessing
transformations). Below is a basic example:

.. code:: python

    import torch

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, X, transform=None):
            self.X = X
            self.transform = transform

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            sample = self.X[idx]
            if self.transform:
                sample = self.transform(sample)
            return sample

Estimator defition
^^^^^^^^^^^^^^^^^^^

Estimators are defined by creating a class that inherits from
`lightning.LightningModule <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_. In this class,
you must override specific methods to customize training, validation, and testing logic. For detailed guidance,
refer to the official documentation
`Lightning Module <https://lightning.ai/docs/pytorch/stable/common/lightning_module.html>`_. Here is an example of
an estimator class:

.. code:: python

    import torch

    class CustomEstimator(LightningModule):
        def __init__(self, model, optimizer, loss_fn):
            super().__init__()
            self.model = model
            self.optimizer = optimizer
            self.loss_fn = loss_fn

        def forward(self, X):
            return self.model(X)

        def training_step(self, batch, batch_idx):
            X, y = batch
            y_pred = self(X)
            loss = self.loss_fn(y_pred, y)
            self.log("train_loss", loss)
            return loss

        def configure_optimizers(self):
            return self.optimizer

Training
^^^^^^^^^^^^^^^^^^^

The defined LightningModule is used as an argument to the
`Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ class provided by theLightning library.
The Trainer handles the training process, including logging, checkpointing, and scaling across devices. Refer to the
`Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_ documentation for further details. Here is
an example of how to train the model:

.. code:: python

    from lightning import Trainer

    # Instantiate dataset, dataloaders, model, optimizer, and loss function
    dataset = CustomDataset(X, transform=your_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    model = CustomModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Define the estimator
    estimator = CustomEstimator(model, optimizer, loss_fn)

    # Train the estimator using the Trainer
    trainer = Trainer(max_epochs=10)
    trainer.fit(estimator, dataloader)



