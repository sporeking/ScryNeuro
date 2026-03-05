"""
ScryNeuro Neural Network Runtime
=================================

Neural network model management plugin for the ScryNeuro Prolog-Python bridge.
Provides model loading, inference, and registry for PyTorch, ONNX, and custom models.

This module is imported by scryer_nn.pl via py_import("scryer_nn_runtime", ...).

Usage from Prolog (via prolog/scryer_nn.pl):
    ?- nn_load(mnist, "models/mnist.pt", [model_type=pytorch]).
    ?- nn_predict(mnist, InputHandle, OutputHandle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from scryer_py_runtime import get_device

logger = logging.getLogger("scryneuro.nn")


# ============================================================================
# Model Registry
# ============================================================================


@dataclass
class ModelEntry:
    """A registered neural network model."""

    name: str
    model: Any  # The actual model object (PyTorch Module, etc.)
    model_type: str  # "pytorch", "tensorflow", "onnx", "custom"
    device: str
    metadata: dict = field(default_factory=dict)


class ModelRegistry:
    """Global registry for neural network models.

    Supports PyTorch, ONNX, and custom model loaders.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelEntry] = {}

    def load(
        self,
        name: str,
        path: str,
        model_type: str = "pytorch",
        device: str = "auto",
        **kwargs: Any,
    ) -> ModelEntry:
        """Load a model from disk and register it."""
        if name in self._models:
            raise ValueError(f"Model '{name}' is already loaded")

        device = get_device(device)

        if model_type == "pytorch":
            model = self._load_pytorch(path, device, **kwargs)
        elif model_type == "onnx":
            model = self._load_onnx(path, **kwargs)
        elif model_type == "custom":
            model = self._load_custom(path, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        entry = ModelEntry(
            name=name,
            model=model,
            model_type=model_type,
            device=device,
            metadata=kwargs,
        )
        self._models[name] = entry
        logger.info(f"Loaded model '{name}' ({model_type}) on {device}")
        return entry

    def get(self, name: str) -> ModelEntry:
        """Get a registered model by name."""
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found. Available: {list(self._models)}")
        return self._models[name]

    def predict(self, name: str, input_data: Any, **kwargs: Any) -> Any:
        """Run inference on a registered model."""
        entry = self.get(name)

        if entry.model_type == "pytorch":
            return self._predict_pytorch(entry, input_data, **kwargs)
        elif entry.model_type == "onnx":
            return self._predict_onnx(entry, input_data, **kwargs)
        else:
            # Custom models should be callable
            return entry.model(input_data, **kwargs)

    def unload(self, name: str) -> None:
        """Unload a model and free resources."""
        if name in self._models:
            del self._models[name]
            logger.info(f"Unloaded model '{name}'")

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return list(self._models.keys())

    # --- Loaders ---

    @staticmethod
    def _load_pytorch(path: str, device: str, **kwargs: Any) -> Any:
        import torch

        model = torch.load(
            path, map_location=device, weights_only=kwargs.get("weights_only", False)
        )
        if hasattr(model, "eval"):
            model.eval()
        return model

    @staticmethod
    def _load_onnx(path: str, **kwargs: Any) -> Any:
        import onnxruntime as ort

        providers = kwargs.get("providers", ["CPUExecutionProvider"])
        return ort.InferenceSession(path, providers=providers)

    @staticmethod
    def _load_custom(path: str, **kwargs: Any) -> Any:
        """Load a custom model via a loader function."""
        loader_spec = kwargs.get("loader")
        if not loader_spec:
            raise ValueError(
                "Custom models require 'loader' kwarg (e.g., 'mymodule:load_func')"
            )
        module_name, func_name = loader_spec.rsplit(":", 1)
        import importlib

        mod = importlib.import_module(module_name)
        loader_fn = getattr(mod, func_name)
        return loader_fn(path, **kwargs)

    # --- Predictors ---

    @staticmethod
    def _predict_pytorch(entry: ModelEntry, input_data: Any, **kwargs: Any) -> Any:
        import torch

        with torch.no_grad():
            if isinstance(input_data, list):
                input_data = torch.tensor(input_data, device=entry.device)
            elif not isinstance(input_data, torch.Tensor):
                input_data = torch.tensor(input_data, device=entry.device)
            else:
                input_data = input_data.to(entry.device)
            output = entry.model(input_data)
            return output

    @staticmethod
    def _predict_onnx(entry: ModelEntry, input_data: Any, **kwargs: Any) -> Any:
        import numpy as np

        session = entry.model
        input_name = session.get_inputs()[0].name
        if isinstance(input_data, list):
            input_data = np.array(input_data, dtype=np.float32)
        result = session.run(None, {input_name: input_data})
        return result[0]


# ============================================================================
# Global Instance
# ============================================================================

_model_registry = ModelRegistry()


# ============================================================================
# Module-level API (called from Rust FFI via spy_call)
# ============================================================================


def load_model(path: str, kwargs: Optional[dict] = None) -> Any:
    """Load a neural network model.

    Called from Prolog: py_call(Runtime, "load_model", PathHandle, KwargsHandle, Result).
    """
    if kwargs is None:
        kwargs = {}
    name = kwargs.pop("name", Path(path).stem)
    model_type = kwargs.pop("model_type", "pytorch")
    device = kwargs.pop("device", "auto")
    entry = _model_registry.load(
        name, path, model_type=model_type, device=device, **kwargs
    )
    return entry.model


def predict(model: Any, input_data: Any, kwargs: Optional[dict] = None) -> Any:
    """Run inference on a model.

    Called from Prolog: py_call(Runtime, "predict", ModelHandle, InputHandle, Result).
    """
    if kwargs is None:
        kwargs = {}

    # If model is callable, call directly
    if callable(model):
        try:
            import torch

            with torch.no_grad():
                return model(input_data)
        except ImportError:
            return model(input_data)

    raise TypeError(f"Model of type {type(model)} is not callable")
