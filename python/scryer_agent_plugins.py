import importlib as _importlib
import sys as _sys

_MODULE = _importlib.import_module("scryer_agent.plugins")
_sys.modules[__name__] = _MODULE
