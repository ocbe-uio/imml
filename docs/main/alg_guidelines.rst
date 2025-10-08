Algorithm selection guide
==============================

This page provides a quick overview of the available algorithms in `iMML` and the
supported input modalities. Use this table to choose an appropriate method for your
task and check whether additional modules or dependencies are required.

.. list-table::
   :header-rows: 1
   :widths: 10 5 70 5 10
   :align: center

   * - Task
     - Algorithm
     - Input modalities
     - Module
     - Extra dependencies
   * - Classification
     - M3Care
     - Numeric | Image | Text
     - deep
     -
   * - Classification
     - MUSE
     - Numeric | Text | Series
     - deep
     -
   * - Classification
     - RAGPT
     - Image & Text
     - deep
     -
   * - Clustering
     - DAIMC
     - Numeric
     -
     -
   * - Clustering
     - EEIMVC
     - Numeric
     -
     -
   * - Clustering
     - IMSCAGL
     - Numeric
     - matlab
     - octave
   * - Clustering
     - IMSR
     - Numeric
     -
     -
   * - Clustering
     - IntegrAO
     - Numeric
     - deep
     -
   * - Clustering
     - LFIMVC
     - Numeric
     -
     -
   * - Clustering
     - MKKMIK
     - Numeric
     - matlab
     - octave
   * - Clustering
     - MONET
     - Numeric
     -
     -
   * - Clustering
     - MRGCN
     - Numeric
     - deep
     -
   * - Clustering
     - NEMO
     - Numeric
     -
     -
   * - Clustering
     - OMVC
     - Numeric
     - matlab
     - octave
   * - Clustering
     - OPIMC
     - Numeric
     - matlab
     - octave
   * - Clustering
     - OSLFIMVC
     - Numeric
     - matlab
     - octave, octave-statistics
   * - Clustering
     - PIMVC
     - Numeric
     - matlab
     - octave
   * - Clustering
     - SIMCADC
     - Numeric
     -
     -
   * - Clustering
     - SUMO
     - Numeric
     -
     -
   * - Decomposition
     - DFMF
     - Numeric
     -
     -
   * - Decomposition
     - MOFA
     - Numeric
     -
     -
   * - Decomposition
     - JNMF
     - Numeric
     - r
     - R, nnTensor
   * - Feature selection
     - JNMFFeatureSelection
     - Numeric
     - r
     - R, nnTensor
   * - Impute
     - DFMFImputer
     - Numeric
     -
     -
   * - Impute
     - MOFAImputer
     - Numeric
     -
     -
   * - Impute
     - JNMFImputer
     - Numeric
     - r
     - R, nnTensor
   * - Retrieve
     - MCR
     - Image & Text
     - deep
     -
   * - Statistics
     - pid
     - Numeric
     -
     -

How to install an additional module
----------------------------------------------------------

See our `page <https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies>`__ on
how to install a module.

How to install extra dependencies
----------------------------------------------------------

Extra dependencies when using "matlab" module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use 'matlab' as an engine, you will need to have `Octave` (`MATLAB`) in your machine. In linux, you can
install it using the following commands:

.. code:: bash

    sudo apt install octave

For other platforms, please refer to the `official guides <https://octave.org/download>`__.

Additionally, to install extra dependencies, execute the following commands in a terminal:

.. code:: bash

    sudo apt install octave-control
    sudo apt install octave-statistics

Extra dependencies when using "r" module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use 'r' as an engine, you will need to have R in your machine. In linux, you can install it using the
following commands:

.. code:: bash

    sudo apt install r-base r-base-dev -y

For other platforms, please refer to the `official guides <https://cran.r-project.org/doc/manuals/r-patched/R-admin.html>`__.

Additionally, to install extra dependencies, execute the following commands in R:

.. code:: R

    install.packages("nnTensor")
