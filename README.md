---
<p align="center"><b>A Python package for integrating, processing, and analyzing incomplete multi-modal datasets</b></p>

[![PyPI - Version](https://img.shields.io/pypi/v/imml)](https://pypi.org/project/imml/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/imml)
[![Read the Docs](https://img.shields.io/readthedocs/imml)](https://imml.readthedocs.io)
[![CI Tests](https://github.com/ocbe-uio/imml/actions/workflows/ci_test.yml/badge.svg)](https://github.com/ocbe-uio/imml/actions/workflows/ci_test.yml)
![Codecov](https://codecov.io/github/ocbe-uio/imml/graph/bundle/badge.svg)
[![CodeQL](https://github.com/ocbe-uio/imml/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ocbe-uio/imml/actions/workflows/github-code-scanning/codeql)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/ocbe-uio/imml/pulls)
![PyPI - Status](https://img.shields.io/pypi/status/imml)
![GitHub repo size](https://img.shields.io/github/repo-size/ocbe-uio/imml)
[![GitHub License](https://img.shields.io/github/license/ocbe-uio/imml)](https://github.com/ocbe-uio/imml/blob/main/LICENSE)
[![Cite](https://img.shields.io/badge/Cite-BibTeX-informational.svg)](https://github.com/ocbe-uio/imml#Citation)

<p align="center">
  <img alt="iMML Logo" src="https://raw.githubusercontent.com/ocbe-uio/imml/refs/heads/main/docs/figures/logo_imml.png" width="150">
</p>

<p align="center">
  <a href="https://doi.org/10.1038/s41467-026-77212-w"><b>Paper</b></a> &nbsp;·&nbsp;
  <a href="#Overview"><b>Overview</b></a> &nbsp;·&nbsp;
  <a href="#Key-features"><b>Key features</b></a> &nbsp;·&nbsp;
  <a href="#Installation"><b>Installation</b></a> &nbsp;·&nbsp;
  <a href="#Usage"><b>Usage</b></a> &nbsp;·&nbsp;
  <a href="#Free-software"><b>Free software</b></a> &nbsp;·&nbsp;
  <a href="#Contribute"><b>Contribute</b></a> &nbsp;·&nbsp;
  <a href="#Help-us-grow"><b>Help us</b></a> &nbsp;·&nbsp;
  <a href="#Citation"><b>Citation</b></a>
</p>

---

Overview
====================

Multi-modal learning, where diverse data types are integrated and analyzed together, has emerged as a critical 
field in artificial intelligence.
However, most algorithms assume fully observed data, an assumption that is often unrealistic in real-world scenarios.
To address this gap, we have developed *iMML*, a Python package designed for multi-modal learning with incomplete data.

![Overview of iMML for multi-modal learning with incomplete data](https://raw.githubusercontent.com/ocbe-uio/imml/refs/heads/main/docs/figures/graph.png)
<p align="center"><strong>Overview of iMML for multi-modal learning with incomplete data.</strong></p>

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
--------------

Run the following command to install the most recent release of *iMML* using *pip*:

```bash
pip install imml
```

Or if you prefer *uv*, use:

```bash
uv pip install imml
```

Some features of *iMML* rely on optional dependencies. To enable these additional features, ensure you install 
the required packages as described in our documentation: https://imml.readthedocs.io/stable/main/installation.html.


Usage
--------

For this example, we will generate a random multi-modal dataset, that we have called ``Xs``:

```python
import numpy as np
Xs = [np.random.random((10,5)) for i in range(3)] # or your multi-modal dataset
```

You can use any other complete or incomplete multi-modal dataset. Once you have your dataset ready, you can
leverage the *iMML* library for a wide range of machine learning tasks, such as:

- Decompose a multi-modal dataset using ``MOFA`` to capture joint information.

```python
from imml.decomposition import MOFA
transformed_Xs = MOFA().fit_transform(Xs)
```

- Cluster samples from a multi-modal dataset using ``NEMO`` to find hidden groups.

```python
from imml.cluster import NEMO
labels = NEMO().fit_predict(Xs)
```

- Simulate incomplete multi-modal datasets for evaluation and testing purposes using ``Amputer``.

```python
from imml.ampute import Amputer
transformed_Xs = Amputer(p=0.8).fit_transform(Xs)
```

Free software
-------------

*iMML* is free software; you can redistribute it and/or modify it under the terms of the `BSD 3-Clause License`.

Contribute
====================

Our vision is to establish *iMML* as a leading and reliable library for multi-modal learning across research and 
applied settings. Our priorities include to broaden algorithmic coverage, improve performance and 
scalability, strengthen interoperability, and grow a healthy contributor community. Therefore, we welcome 
practitioners, researchers, and the open-source community to contribute to the *iMML* project, and in doing so, 
helping us extend and refine the library for the community. Such a community-wide effort will make *iMML* more 
versatile, sustainable, powerful, and accessible to the machine learning community across many domains.

For the full contributing guide, please see:

- In-repo: https://github.com/ocbe-uio/imml/tree/main?tab=contributing-ov-file
- Documentation: https://imml.readthedocs.io/stable/development/contributing.html

Help us grow
====================

How you can help *iMML* grow:

- 🔥 Try it out and share your feedback.
- 🤝 Contribute if you are interested in building with us.
- 🗣️ Share this project with colleagues who deal with multi-modal data.
- 🌟 And of course… give the repo a star to support the project!

Citation
====================

[![Google scholar](https://img.shields.io/badge/Google%20Scholar-4285F4?style=for-the-badge&logo=googlescholar&logoColor=white)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=jxo3FisAAAAJ&sortby=pubdate&citation_for_view=jxo3FisAAAAJ:XAp-VaTZjjwC)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-00CCBB?style=for-the-badge&logo=researchgate&logoColor=white)](https://www.researchgate.net/publication/413783367_Multi-modal_learning_with_incomplete_data)
[![DOI](https://img.shields.io/badge/DOI-0085CA?style=for-the-badge&logo=doi&logoColor=white)](https://doi.org/10.1038/s41467-026-77212-w)

If you use our package, please cite the following paper:

> López, A., Zobolas, J., Dumontier, T. et al. Multi-modal learning with incomplete data. Nat Commun (2026). https://doi.org/10.1038/s41467-026-77212-w

```
@article{lopez2026multi,
  title={Multi-modal learning with incomplete data},
  author={López, Alberto and Zobolas, John and Dumontier, Tanguy and Aittokallio, Tero},
  journal={Nature Communications},
  year={2026},
  publisher={Nature Publishing Group UK London},
  DOI={https://doi.org/10.1038/s41467-026-77212-w}
}
```
