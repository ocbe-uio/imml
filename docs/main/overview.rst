Overview
====================

Multi-modal learning, where diverse data types are integrated and analyzed together, has emerged as a critical
field in artificial intelligence.
However, most algorithms assume fully observed data, an assumption that is often unrealistic in real-world scenarios.
To address this gap, we have developed `iMML`, a Python package designed for multi-modal learning with incomplete data.

.. image:: /figures/graph.png
  :width: 700
  :alt: Overview of iMML for multi-modal learning with incomplete data.
.. centered::
    **Overview of iMML for multi-modal learning with incomplete data**.


Key features
------------

The key features of this package are:

-   **Coverage**: More than 25 methods for integrating, processing, and analyzing incomplete multi-modal
    datasets implemented as a single, user-friendly interface.
-   **Comprehensive**: Designed to be compatible with widely-used machine learning and data analysis tools, allowing
    use with minimal programming effort.
-   **Extensible**: A unified framework where researchers can contribute and integrate new approaches, serving
    as a community platform for hosting new methods.

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

For this example, we will generate a random multi-modal dataset, that we have called `Xs`:

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

Our vision is to establish `iMML` as a leading and reliable library for multi-modal learning across research and
applied settings. Our priorities include to broaden algorithmic coverage, improve performance and
scalability, strengthen interoperability, and grow a healthy contributor community. Therefore, we welcome
practitioners, researchers, and the open-source community to contribute to the `iMML` project, and in doing so,
helping us extend and refine the library for the community. Such a community-wide effort will make `iMML` more
versatile, sustainable, powerful, and accessible to the machine learning community across many domains.

For the full contributing guide, please see:

- In-repo: https://github.com/ocbe-uio/imml/tree/main?tab=contributing-ov-file
- Documentation: https://imml.readthedocs.io/stable/development/contributing.html

Help us grow
------------

How you can help `iMML` grow:

- 🔥 Try it out and share your feedback.
- 🤝 Contribute if you are interested in building with us.
- 🗣️ Share this project with colleagues who deal with multi-modal data.
- 🌟 And of course… give the repo a star to support the project!

