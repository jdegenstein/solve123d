"""solve123d import definitions"""

from .constraint_solver import *
from ._version import version as __version__

__all__ = ["Variable", "var", "solve", "make_wrapper", "magic"]
