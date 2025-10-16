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


Version 0.1.1
-------------

Updates in this release:

- |Enhancement| Improving documentation for several methods.
- |Enhancement| Improved documentation for installation and extra dependencies.
- |Enhancement| Adding guidelines on how to choose an algorithm.
- |Enhancement| Added license headers to all files.
- |Fix| Fixed iPython dependency issue. Oct2Py depends on iPython but returned an error when importing ipython>=9.0.0.

`.github/workflows/ci_test.yml <https://github.com/ocbe-uio/imml/blob/main/.github/workflows/ci_test.yml>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- |Fix| Fixing actions/missing-workflow-permissions security.


Version 0.1.0
-------------

We are happy to announce the first major public version of `iMML`!
