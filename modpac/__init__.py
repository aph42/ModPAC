### Main import hooks for modpac

from .configuration import *
from .state import *

from .modpac import ModPAC

modpac_root = os.path.dirname(__path__[0])

__version__ = "0.5.0 unstable"
