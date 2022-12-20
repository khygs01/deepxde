"""Package for pytorch NN modules."""

__all__ = [
    "DeepONetCartesianProd",
    "FNN",
    "MIONetCartesianProd",
    "NN",
    "PFNN",
    "FSFNN",
    "PODDeepONet",
    "PODMIONet",
]

from .deeponet import DeepONetCartesianProd, PODDeepONet
from .mionet import MIONetCartesianProd, PODMIONet
from .fnn import FNN, PFNN, FSFNN
from .nn import NN
