:orphan:

Tutorials
====================

This folder collects hands‑on, runnable examples that demonstrate how to use `iMML` for common multi‑modal learning
workflows, including exploring datasets, simulating missing modalities, classifying and clustering with incomplete data,
among others. Each script is self‑contained and designed to be easy to adapt to your own data.
You can find the tutorials in:

- Online documentation: https://imml.readthedocs.io/stable/auto_tutorials/index.html.
  The online gallery renders these scripts as rich, formatted pages with figures.
- In-repo: https://github.com/ocbe-uio/imml/tree/main/tutorials. As Python scripts, run any file directly. Most scripts will print intermediate results and pop up figures.

Questions or feedback?

- Open an issue: https://github.com/ocbe-uio/imml/issues
- Contributions are welcome: https://imml.readthedocs.io/stable/development/contributing.html.


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A multi-modal dataset can be characterized beyond basic shape information. With iMML you can:">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_multil_modal_data_statistics_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_multil_modal_data_statistics.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Statistics and interaction structure of a multi-modal dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Evaluation and benchmarking of new algorithms or models under diverse conditions is essential to ensure their robustness, added value and generalizability. iMML simplifies this process by simulating incomplete multi-modal datasets with modality-wise missing data. This so-called data amputation process allows for controlled testing of methods by introducing missing data from various mechanisms that reflect real-world scenarios where different modalities may be partially observed or entirely missing.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_generate_missing_modalities_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_generate_missing_modalities.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modality-wise missing data simulation (Amputation)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Clustering involves grouping samples into distinct groups. In this tutorial, we show how to use iMML to perform clustering on a multi-modal dataset. We also demonstrate how to work with incomplete multi-modal data, where some samples are missing one or more modalities, and how to benchmark the impact of missingness.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_cluster_incomplete_mmd_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_cluster_incomplete_mmd.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Clustering a multi-modal dataset</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to retrieve samples from an incomplete vision–language dataset using iMML. We will use the MCR retriever to find similar items across modalities (image/text) even when one modality is missing. The example uses the public nlphuji/flickr30k dataset from Hugging Face Datasets, so you don&#x27;t need to prepare files manually.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_retrieve_incomplete_vision_language_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_retrieve_incomplete_vision_language.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Retrieval on a vision–language dataset (flickr30k)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When the learning algorithms cannot directly handle missing data, imputation methods become essential to allow their application. Thus, iMML has a module designed for filling missing data, which can be particularly useful when using external methods that are unable to handle missing values directly.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_impute_multi_modal_data_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_impute_multi_modal_data.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Impute incomplete modality- and feature-wise multi-modal data</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This tutorial demonstrates how to classify samples from an incomplete vision–language dataset using the iMML library. iMML supports robust classification even when some modalities (e.g., text or image) are missing, making it suitable for real‑world multi‑modal data where missingness is common.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_classify_incomplete_vision_language_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_classify_incomplete_vision_language.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classify an incomplete vision–language dataset (Oxford‑IIIT Pets) with deep learning</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-dimensional datasets can severely impact machine learning projects, by increasing computational demands, data-adquisition costs and reducing model interpretability. It can also degrade performance due to the curse of dimensionality, as well as the presence of correlated, noisy, or irrelevant features. Consequently, reducing the number of features is often critical. Dimensionality reduction addresses these challenges by enhancing computational efficiency, highlighting key features, reducing noise, and enabling better data visualization.">

.. only:: html

  .. image:: /auto_tutorials/images/thumb/sphx_glr_select_and_extract_features_thumb.png
    :alt:

  :ref:`sphx_glr_auto_tutorials_select_and_extract_features.py`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dimensionality reduction: Feature extraction and feature selection</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. toctree::
   :hidden:

   /auto_tutorials/multil_modal_data_statistics
   /auto_tutorials/generate_missing_modalities
   /auto_tutorials/cluster_incomplete_mmd
   /auto_tutorials/retrieve_incomplete_vision_language
   /auto_tutorials/impute_multi_modal_data
   /auto_tutorials/classify_incomplete_vision_language
   /auto_tutorials/select_and_extract_features


.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-gallery

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download all examples in Python source code: auto_tutorials_python.zip </auto_tutorials/auto_tutorials_python.zip>`

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download all examples in Jupyter notebooks: auto_tutorials_jupyter.zip </auto_tutorials/auto_tutorials_jupyter.zip>`


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
