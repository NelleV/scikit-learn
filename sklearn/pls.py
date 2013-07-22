import warnings

from CCA.pls import *
from CCA.cca import CCA

warnings.warn("The pls module was renamed to CCA and will be removed "
              "in 0.15", DeprecationWarning)
