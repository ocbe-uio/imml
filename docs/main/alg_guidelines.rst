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
     - :class:`~imml.classify.M3Care`
     - Numeric | Image | Text
     - deep
     -
   * - Classification
     - :class:`~imml.classify.MUSE`
     - Numeric | Text | Time series
     - deep
     -
   * - Classification
     - :class:`~imml.classify.RAGPT`
     - Image & Text
     - deep
     -
   * - Clustering
     - :class:`~imml.cluster.DAIMC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.EEIMVC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.IMSCAGL`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.IMSR`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.IntegrAO`
     - Numeric
     - deep
     -
   * - Clustering
     - :class:`~imml.cluster.LFIMVC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.MKKMIK`
     - Numeric
     - octave
     - octave
   * - Clustering
     - :class:`~imml.cluster.MLMF`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.MONET`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.MRGCN`
     - Numeric
     - deep
     -
   * - Clustering
     - :class:`~imml.cluster.NEMO`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.OMVC`
     - Numeric
     - octave
     - octave
   * - Clustering
     - :class:`~imml.cluster.OPIMC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.OSLFIMVC`
     - Numeric
     - octave
     - octave, octave-statistics
   * - Clustering
     - :class:`~imml.cluster.PIMVC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.SIMCADC`
     - Numeric
     -
     -
   * - Clustering
     - :class:`~imml.cluster.SUMO`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.DFMF`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.MOFA`
     - Numeric
     -
     -
   * - Decomposition
     - :class:`~imml.decomposition.JNMF`
     - Numeric
     -
     -
   * - Feature selection
     - :class:`~imml.feature_selection.JNMFFeatureSelector`
     - Numeric
     -
     -
   * - Attention fusion
     - :class:`~imml.fuse.AttentionFusion`
     - Numeric
     - deep
     -
   * - Concat fusion
     - :class:`~imml.fuse.ConcatFusion`
     - Numeric
     - deep
     -
   * - EmbraceNet
     - :class:`~imml.fuse.EmbraceNet`
     - Numeric
     - deep
     -
   * - Max fusion
     - :class:`~imml.fuse.MaxFusion`
     - Numeric
     - deep
     -
   * - Mean fusion
     - :class:`~imml.fuse.MeanFusion`
     - Numeric
     - deep
     -
   * - Sum fusion
     - :class:`~imml.fuse.SumFusion`
     - Numeric
     - deep
     -
   * - Impute
     - :class:`~imml.impute.DFMFImputer`
     - Numeric
     -
     -
   * - Impute
     - :class:`~imml.impute.MOFAImputer`
     - Numeric
     -
     -
   * - Impute
     - :class:`~imml.impute.JNMFImputer`
     - Numeric
     -
     -
   * - Statistics
     - :class:`~imml.statistics.pid`
     - Numeric
     -
     -
   * - Survival
     - :class:`~imml.survival.MultiSurv`
     - Numeric
     - deep
     -

How to install an additional module
----------------------------------------------------------

See our `page <https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies>`__ on
how to install a module.

How to install extra dependencies
----------------------------------------------------------

Extra dependencies when using "octave" module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to use 'octave' as an engine, you will need to have `Octave` in your machine. In linux, you can
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
