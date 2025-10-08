Overview
====================

`iMML` is a Python package that provides a **robust tool-set for integrating, processing, and analyzing incomplete
multi-modal datasets** to support a wide range of machine learning tasks. Starting with a dataset containing N samples
with K modalities, `iMML` effectively handles missing data for **classification, clustering, data retrieval,
imputation and amputation, feature selection, feature extraction and data exploration**, hence enabling efficient
analysis of partially observed samples.

.. image:: /figures/graph.png
  :width: 700
  :alt: Overview of iMML for multi-modal learning with incomplete data.
.. centered::
    **Overview of iMML for multi-modal learning with incomplete data**.

Background
----------

Multi-modal learning, where diverse data types are integrated and analyzed together, has emerged as a critical field
in artificial intelligence. Multi-modal machine learning models that effectively integrate multiple data modalities
generally outperform their uni-modal counterparts by leveraging more comprehensive and complementary information.
However, **most algorithms in this field assume fully observed data**, an assumption that is often
unrealistic in real-world scenarios.

Motivation
----------

Learning from incomplete multi-modal data has seen an important growth last years.
Despite this progress, several limitations still persist.
The landscape of available methods is fragmented, largely due to the diversity of use cases and data modalities,
which complicates both their application and benchmarking.
Systematic use and comparison of the current methods are often hindered by practical challenges, such as
incompatible input data formats and conflicting software dependencies.
As a result, researchers and practitioners frequently face challenges in choosing a practical method and invest
considerable efforts into reconciling codebases, rather than addressing the core scientific questions.
This suggests that **the community currently lacks robust and standardized tools to effectively handle
incomplete multi-modal data**.

Key features
------------

To address this gap, we have developed `iMML`, a Python package designed for multi-modal learning with incomplete data.
The key features of this package are:

-   **Comprehensive toolkit**: `iMML` offers a broad set of tools for integrating, processing, and analyzing
    incomplete multi-modal datasets implemented as a single, user-friendly interface to facilitate adoption by
    a wide community of users.
    The package includes extensive technical testing to ensure robustness, and thorough documentation enables
    end-users to apply its functionality effectively.
-   **Accessible**: `iMML` makes the tools readily available to the Python community, simplifying their usage,
    comparison, and benchmarking, and thereby addresses the current lack of resources and standardized methods
    for handling incomplete multi-modal data.
-   **Extensible**: `iMML` provides a common framework where researchers can contribute and
    integrate new approaches, serving as a community platform for hosting new algorithms and methods.

Installation
-------------

Run the following command to install the most recent release of `iMML` using ``pip``:

.. code:: bash

    pip install imml

Or if you prefer ``uv``, use:

.. code:: bash

    uv pip install imml

Some features of *iMML* rely on optional dependencies. To enable these additional features, ensure you install
the required packages as described in our documentation: https://imml.readthedocs.io/stable/main/installation.html.

Usage
--------

This package provides a user-friendly interface to apply these algorithms to user-provided data.
`iMML` was designed to be compatible with widely-used machine learning and data analysis tools, such as
`Pandas <https://pandas.pydata.org/>`__, `Numpy <https://numpy.org/>`__, `Scikit-learn
<https://scikit-learn.org/stable/>`__, and
`Lightning <https://lightning.ai/>`__ , hence allowing researchers to **apply machine learning models with
minimal programming effort**. Moreover, it can be easily integrated into Scikit-learn pipelines for data
preprocessing and modeling.

For this demonstration, we will generate a random dataset, that we have called `Xs`, as a multi-modal dataset
to simulate a multi-modal scenario:

.. code:: python

    import numpy as np
    Xs = [np.random.random((10,5)) for i in range(3)] # or your multi-modal dataset

You can use any other complete or incomplete multi-modal dataset. Once you have your dataset ready, you can
leverage the `iMML` library for a wide range of machine learning tasks, such as:

- Decompose a multi-modal dataset using ``MOFA`` to capture joint information.

.. code:: python

    from imml.decomposition import MOFA
    transformed_Xs = MOFA().fit_transform(Xs)

- Cluster samples from a multi-modal dataset using ``NEMO`` to find hidden groups.

.. code:: python

    from imml.cluster import NEMO
    labels = NEMO().fit_predict(Xs)

- Simulate incomplete multi-modal datasets for evaluation and testing purposes using ``Amputer``.

.. code:: python

    from imml.ampute import Amputer
    transformed_Xs = Amputer(p=0.8).fit_transform(Xs)

Free software
-------------

`iMML` is free software; you can redistribute it and/or modify it under the terms of the `BSD 3-Clause License`.

Contribute
------------

**We welcome practitioners, researchers, and the open-source community** to contribute to the `iMML` project,
and in doing so, helping us extend and refine the library for the community. Such a community-wide effort will
make `iMML` more versatile, sustainable, powerful, and accessible to the machine learning community across
many domains.

Project roadmap
---------------

Our vision is to establish `iMML` as a leading and reliable library for multi-modal learning across research and
applied settings. Therefore, our priorities include to broaden algorithmic coverage, improve performance and
scalability, strengthen interoperability, and grow a healthy contributor community.


