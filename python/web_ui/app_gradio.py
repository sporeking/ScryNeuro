import importlib as _importlib
import sys as _sys

_MODULE = _importlib.import_module("scryer_agent.web_ui.app_gradio")
_sys.modules[__name__] = _MODULE

build_demo = _MODULE.build_demo
launch = _MODULE.launch


if __name__ == "__main__":
    launch()
