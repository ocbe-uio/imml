Changelog
=========

.. role:: raw-html(raw)
   :format: html

.. role:: raw-latex(raw)
   :format: latex

.. |MajorFeature| replace:: :raw-html:`<font color="green">[Major Feature]</font>`
.. |Feature| replace:: :raw-html:`<font color="green">[Feature]</font>`
.. |Efficiency| replace:: :raw-html:`<font color="blue">[Efficiency]</font>`
.. |Enhancement| replace:: :raw-html:`<font color="blue">[Enhancement]</font>`
.. |Fix| replace:: :raw-html:`<font color="red">[Fix]</font>`
.. |API| replace:: :raw-html:`<font color="DarkOrange">[API]</font>`

Change tags (adopted from `Scikit-learn
<https://scikit-learn.org/stable/>`__ and `mvlearn
<https://mvlearn.github.io/>`__):

- |MajorFeature| : something big that you couldn’t do before.

- |Feature| : something that you couldn’t do before.

- |Efficiency| : an existing feature now may not require as much computation or memory.

- |Enhancement| : a miscellaneous minor improvement.

- |Fix| : something that previously didn’t work as documentated – or according to reasonable expectations – should now work.

- |API| : you will need to change your code to have the same effect in the future; or a feature will be removed in the future.


Version 0.6.0
-------------
July 10, 2026

Updates in this release:

:mod:`imml.preprocessing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Feature| :class:`~imml.preprocessing.UMTransformer` has been implemented.


Version 0.5.1
-------------
June 13, 2026

Updates in this release:

- |Fix| Sphinx versions have been restricted to <9 for creating the documentation, as new versions do not work well.
- |Efficiency| : R dependency has been removed for building the documentation.

:mod:`imml.cluster`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Enhancement| Python engine has been implemented in :class:`~imml.cluster.PIMVC`.

:mod:`imml.decomposition`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Enhancement| Python engine has been implemented in :class:`~imml.decomposition.JNMF`.

:mod:`imml.preprocessing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| :class:`~imml.preprocessing.select_complete_samples` and
  :class:`~imml.preprocessing.select_incomplete_samples` can now also return the y variable.

:mod:`imml.visualize`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Feature| mod_names argument was added to :class:`~imml.visualize.plot_combinations`.


Version 0.4.0
-------------
June 5, 2026

Updates in this release:

:mod:`imml.classify`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| Updated :class:`~imml.classify.RAGPT` API for beter alignment with the framework.
- |Fix| Better alignment with the original implementation of :class:`~imml.classify.MUSE`.

:mod:`imml.load`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| Updated :class:`~imml.load.RAGPTDataset` API for beter alignment with the framework.

:mod:`imml.retrieve`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| This module has been removed due to none method meet the selection criteria :mod:`~imml.retrieve`.


Version 0.3.1
-------------
May 28, 2026

Updates in this release:

:mod:`imml.classify`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixed device when processing images in :class:`~imml.classify.M3Care`.
- |Fix| Better alignment with the original implementation of :class:`~imml.classify.RAGPT`.

:mod:`imml.cluster`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixed eigensolver to use :code:`eigsh` instead of :code:`eigs` for symmetric kernel matrices
  in :class:`~imml.cluster.EEIMVC`, ensuring equivalence with the original MATLAB implementation.
- |Fix| Fixed error in original Octave implementation :class:`~imml.cluster.SIMCADC`.
- |Fix| Revised Python translation in :class:`~imml.cluster.LFIMVC`.
- |Fix| Revised Python translation in :class:`~imml.cluster.DAIMC`.
- |Fix| Revised Python translation in :class:`~imml.cluster.NEMO`.
- |API| Block_size parameter changed to batch_size in :class:`~imml.cluster.OPIMC`.

:mod:`imml.model_selection`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixed default return_type parameter in :class:`~imml.model_selection.MMSplitter`.

:mod:`imml.preprocessing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixed :class:`~imml.preprocessing.select_complete_samples` and
  :class:`~imml.preprocessing.select_incomplete_samples` when working with string and object dtypes.
- |Fix| Removed redundant mask calculation in :class:`~imml.preprocessing.select_incomplete_samples`.


Version 0.3.0
-------------
March 17, 2026

Updates in this release:

- |MajorFeature| We have created a new module :mod:`imml.model_selection`.
- |Enhancement| We have added a "See also" section to related classes.
- |Enhancement| We have added a section to link tutorials with classes.
- |Enhancement| `iMML` supports now Python 3.14.
- |API| Matlab module and arguments were replaced by Octave to better reflect their usage.
- |Efficiency| Imports of optional modules have been centralized.
- |API| SimpleModImputer and simple_mod_imputer was removed. You can use  MMTransformer(transformer = SimpleImputer())
  instead.
- |Efficiency| snfpy package was removed from the requirements.

:mod:`imml.impute`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.impute.SimpleModImputer` and :class:`~imml.impute.simple_mod_imputer` was removed. You can
  use MMTransformer(transformer = SimpleImputer()) instead.

:mod:`imml.model_selection`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Feature| :class:`~imml.model_selection.MMSplitter` was added.
- |Feature| :class:`~imml.model_selection.train_test_mm_split` was added.

:mod:`imml.preprocessing`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.preprocessing.Multi_Mod_Transformer` was renamed to :class:`~imml.preprocessing.MMTransformer`.

:mod:`imml.statistics`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.statistics.pid` now returns also the total information.

:mod:`imml.utils`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.utils.check_Xs` was converted to :class:`~imml.utils.check_Xs_y`.

:mod:`imml.visualize`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Feature| :class:`~imml.visualize.plot_summary` was added.
- |Feature| :class:`~imml.visualize.plot_combinations` was added.


Version 0.2.0
-------------
November 3, 2025

Updates in this release:

- |Fix| Corrected inheritance hierarchy in clustering algorithms by replacing ``ClassifierMixin`` with the
  appropriate ``ClusterMixin`` base class from `Scikit-learn <https://scikit-learn.org/stable/>`__.
- |Enhancement| Improved code readability by updating references to
  `Lightning <https://lightning.ai/docs/pytorch/stable/starter/introduction.html>`_ package base classes to use their
  explicit class names instead of generic references.
- |Enhancement| Enhanced navigation in the
  `algorithm selection guide <https://imml.readthedocs.io/stable/main/alg_guidelines.html>`_ by adding direct
  hyperlinks from each algorithm to its corresponding detailed documentation page, making it easier for users
  to explore specific implementations.
- |Efficiency| numba package was removed from the requirements.

:mod:`imml.ampute`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Enhancement| :class:`~imml.ampute.Amputer` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.
- |Enhancement| :class:`~imml.ampute.RemoveMods` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.

:mod:`imml.classify`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| :class:`~imml.classify.MUSE` Fixed text extractor load when using text modality.
- |Fix| :class:`~imml.classify.M3Care` Fixed error when working with multiple data modalities.


:mod:`imml.impute`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Enhancement| :class:`~imml.ampute.MissingModIndicator` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.
- |Enhancement| :class:`~imml.ampute.get_missing_mod_indicator` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.
- |Enhancement| :class:`~imml.ampute.ObservedModIndicator` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.
- |Enhancement| :class:`~imml.ampute.get_observed_mod_indicator` Now support lists and
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.

:mod:`imml.load`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.load.M3CareDataset` observed_mod_indicator argument was removed.
- |API| :class:`~imml.load.MUSEDataset` observed_mod_indicator and y_indicator arguments were removed.

:mod:`imml.utils`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Enhancement| :class:`~imml.utils.check_Xs` Now support
  `pytorch tensors <https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor>`_.


Version 0.1.1
-------------
October 17, 2025

Updates in this release:

- |Enhancement| Improving documentation for several methods.
- |Enhancement| Improved documentation for installation and extra dependencies.
- |Enhancement| Adding `guidelines <https://imml.readthedocs.io/stable/main/alg_guidelines.html>`_ on how to choose
  an algorithm.
- |Enhancement| Added license headers to all files.
- |Fix| Fixed iPython dependency issue. Oct2Py depends on iPython but returned an error when importing ipython>=9.0.0.

`.github/workflows/ci_test.yml <https://github.com/ocbe-uio/imml/blob/main/.github/workflows/ci_test.yml>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixing actions/missing-workflow-permissions security.


Version 0.1.0
-------------
October 03, 2025

We are happy to announce the first major public version of `iMML`!
