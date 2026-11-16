"""Type aliases shared by the package."""
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float64]    #: array of floats: compositions, flows, residuals, ...
