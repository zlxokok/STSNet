import warnings

from .distance import PairwiseDistance
from .module import Module
from .. import functional as F
from .. import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional