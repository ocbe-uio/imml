# License: BSD-3-Clause

__version__ = "0.3.1"

try:
    import torch
    from torch import nn
    from torchvision import models as models
    import lightning as L
    import torchvision.transforms as transforms
    from torch import Tensor
    LightningModule = L.LightningModule
    Module = nn.Module
    Dataset = torch.utils.data.Dataset
    deepmodule_installed = True
    deepmodule_error = None
except ImportError:
    LightningModule = object
    Module = object
    Dataset = object
    Tensor = str
    deepmodule_installed = False
    deepmodule_error = "Module 'deep' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"


try:
    import oct2py
    octavemodule_installed = True
    oct2py_module_error = None
except ImportError:
    octavemodule_installed = False
    oct2py_module_error = "Module 'octave' needs to be installed. See https://imml.readthedocs.io/stable/main/installation.html#optional-dependencies"
