import importlib as _importlib
import sys as _sys

_MODULE = _importlib.import_module("scryer_agent.tools")
_sys.modules[__name__] = _MODULE
