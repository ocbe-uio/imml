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


Version 0.3.0
-------------
November 3, 2025

Updates in this release:

- |MajorFeature| We have created a new module :mod:`imml.model_selection`.
- |Enhancement| We have added a "See also" section to link tutorials with classes.
- |Enhancement| `iMML` supports now Python 3.14.
- |Enhancement| Added a tutorial for dealing with incomplete tabular-text data.
  By `RafRomB <https://github.com/RafRomB>`__. `#14 <https://github.com/ocbe-uio/imml/pull/14>`__

:mod:`imml.model_selection`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Feature| :class:`~imml.model_selection.multi_train_test_split` was added.

:mod:`imml.utils`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |API| :class:`~imml.utils.check_Xs` was converted to :class:`~imml.utils.check_Xs_y`

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
- |Fix| :class:`~imml.classify.MUSE` Fixed error when working with multiple data modalities.
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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixing actions/missing-workflow-permissions security.


Version 0.1.0
-------------
October 03, 2025

We are happy to announce the first major public version of `iMML`!
