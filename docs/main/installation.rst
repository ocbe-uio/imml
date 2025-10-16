Installation
============

Follow the instructions below to install `iMML` on your system. Some features of `iMML` rely on optional dependencies.
To enable these additional features, ensure you install the required packages as described in the
:ref:`optional_dependencies` section below.

.. _pipAnchor:

Instructions
-----------------------------

Using pip
^^^^^^^^^

To install `iMML` using ``pip``, ensure Python 3 and ``pip`` (Python's package manager) are properly set up on
your system. If ``pip`` is not already installed or needs an update, refer to the official documentation at
https://pip.pypa.io.

Run the following command to install the most recent release of `iMML`:

.. code:: bash

    pip install imml

To upgrade to a newer version of `iMML`, use the `--upgrade` flag:

.. code:: bash

    pip install --upgrade imml

If you do not have permission, install `iMML` in your user directory by adding the `--user` flag:

.. code:: bash

    pip install --user imml

Using uv
^^^^^^^^^

To install `iMML` using ``uv``, ensure Python 3 and ``uv`` (Python's package manager) are properly set up on
your system. If ``uv`` is not already installed or needs an update, refer to the official documentation at
https://docs.astral.sh/uv/.

Run the following command to install the most recent release of `iMML`:

.. code:: bash

    uv pip install imml

Manually
^^^^^^^^^^^^

Alternatively, you can download `iMML` directly from its source repository on GitHub. This method is useful if you
want to install the latest development version:

1. Clone the repository:

.. code:: bash

    git clone

2. Navigate to the project directory:

.. code:: bash

    cd imml

Install the package in editable mode:

.. code:: bash

    pip install -e .

This will install `iMML` and the required dependencies (see below).

Dependencies
^^^^^^^^^^^^^^^^^^^^^

`iMML` requires the following packages:

-  scikit-learn>=1.4.1
-  pandas>=2.2.2
-  networkx>=2.5
-  h5py>=3.9.0
-  snfpy>=0
-  control>=0.10.1
-  cvxpy>=1.7.1
-  numba>=0.58.0

`iMML` is supported for Python 3.10 and above.

.. _optional_dependencies:

Optional dependencies
-----------------------------

`iMML` supports additional features that require optional dependencies. You can install these dependencies by
specifying extras during installation. For example:

- [matlab]: Some algorithms were originally developed in Matlab. If you want to use the original implementation,
  use this module.
- [r]: Some algorithms were originally developed in R. If you want to use the original implementation, use this module.
- [deep]: Deep learning methods are included in this module.

To include these dependencies, execute in the terminal:

.. code:: bash

    pip install imml[keyword]

where 'keyword' is from the list above. For example:

.. code:: bash

    pip install imml[deep] # to install "deep" module

    pip install imml[matlab] # to install "matlab" module

    pip install imml[r] # to install "r" module

To install all possible dependencies:

.. code:: bash

    pip install imml[all]


OS Requirements
---------------
This package is supported for Linux, macOS and Windows machines.

Testing
-------

To test the package, install the testing dependencies using:

.. code:: bash

    pip install imml[tests]

This will install pytest and pytest-cov.

Documenting
------------

To include new documentation, install the documenting dependencies using:

.. code:: bash

    pip install imml[docs]
