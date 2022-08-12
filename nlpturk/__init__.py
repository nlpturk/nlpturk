import sys

from .pkg import __version__
from ._base import _M

sys.modules[__name__].__class__ = _M
