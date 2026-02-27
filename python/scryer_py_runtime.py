"""
ScryNeuro Python Runtime
========================

Python-side support module for the ScryNeuro Prolog-Python bridge.
Provides high-level interfaces for:

- Neural network model management (load, predict, batch)
- LLM integration (load, generate, chat)
- Tensor utilities (create, convert, inspect)
- Device management (CPU/GPU)

This module is imported by the Rust bridge (via PyO3) and called
through the FFI layer from Prolog.

Usage from Prolog (via scryer_py.pl):
    ?- py_import("scryer_py_runtime", RT).
    ?- py_call(RT, "load_model", PathHandle, KwargsHandle, ModelHandle).
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger("scryneuro")

# ============================================================================
# Device Management
# ============================================================================


def get_device(preferred: str = "auto") -> str:
    """Get the best available compute device.

    Args:
        preferred: "auto", "cpu", "cuda", "cuda:0", "mps", etc.

    Returns:
        Device string suitable for PyTorch/JAX.
    """
    if preferred != "auto":
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


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
        """Load a model from disk and register it.

        Args:
            name: Unique identifier for the model.
            path: Path to the model file.
            model_type: One of "pytorch", "onnx", "custom".
            device: Compute device ("auto", "cpu", "cuda", etc.).
            **kwargs: Additional loader arguments.

        Returns:
            The registered ModelEntry.
        """
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
        """Run inference on a registered model.

        Args:
            name: Model identifier.
            input_data: Input tensor/data.
            **kwargs: Additional inference options.

        Returns:
            Model output (tensor or structured data).
        """
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
        """Load a custom model via a loader function.

        kwargs must contain 'loader' key pointing to a callable:
            loader = "mymodule:load_func"
        """
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
# LLM Manager
# ============================================================================


@dataclass
class LLMEntry:
    """A registered LLM instance."""

    name: str
    model_id: str
    provider: str  # "openai", "anthropic", "huggingface", "ollama", "custom"
    client: Any  # Provider-specific client object
    metadata: dict = field(default_factory=dict)


class LLMManager:
    """Manager for Large Language Models.

    Supports multiple providers with a unified interface.
    """

    def __init__(self) -> None:
        self._llms: dict[str, LLMEntry] = {}

    def load(
        self,
        name: str,
        model_id: str,
        provider: str = "openai",
        **kwargs: Any,
    ) -> LLMEntry:
        """Load/configure an LLM.

        Args:
            name: Unique identifier.
            model_id: Model identifier (e.g., "gpt-4", "llama-3-8b").
            provider: LLM provider.
            **kwargs: Provider-specific configuration.

        Returns:
            The registered LLMEntry.
        """
        if name in self._llms:
            raise ValueError(f"LLM '{name}' is already loaded")

        if provider == "openai":
            client = self._init_openai(model_id, **kwargs)
        elif provider == "anthropic":
            client = self._init_anthropic(model_id, **kwargs)
        elif provider == "huggingface":
            client = self._init_huggingface(model_id, **kwargs)
        elif provider == "ollama":
            client = self._init_ollama(model_id, **kwargs)
        elif provider == "custom":
            client = kwargs.get("client")
            if client is None:
                raise ValueError("Custom provider requires 'client' kwarg")
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")

        entry = LLMEntry(
            name=name,
            model_id=model_id,
            provider=provider,
            client=client,
            metadata=kwargs,
        )
        self._llms[name] = entry
        logger.info(f"Loaded LLM '{name}' ({provider}:{model_id})")
        return entry

    def get(self, name: str) -> LLMEntry:
        """Get a registered LLM by name."""
        if name not in self._llms:
            raise KeyError(f"LLM '{name}' not found. Available: {list(self._llms)}")
        return self._llms[name]

    def generate(
        self,
        name: str,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Generate text from an LLM.

        Args:
            name: LLM identifier.
            prompt: Input prompt.
            **kwargs: Generation parameters (temperature, max_tokens, etc.).

        Returns:
            Generated text string.
        """
        entry = self.get(name)

        if entry.provider == "openai":
            return self._generate_openai(entry, prompt, **kwargs)
        elif entry.provider == "anthropic":
            return self._generate_anthropic(entry, prompt, **kwargs)
        elif entry.provider == "huggingface":
            return self._generate_huggingface(entry, prompt, **kwargs)
        elif entry.provider == "ollama":
            return self._generate_ollama(entry, prompt, **kwargs)
        elif entry.provider == "custom":
            return entry.client(prompt, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {entry.provider}")

    def unload(self, name: str) -> None:
        """Unload an LLM."""
        if name in self._llms:
            del self._llms[name]
            logger.info(f"Unloaded LLM '{name}'")

    # --- Provider Initializers ---

    @staticmethod
    def _init_openai(model_id: str, **kwargs: Any) -> Any:
        from openai import OpenAI

        api_key = kwargs.get("api_key")
        base_url = kwargs.get("base_url")
        return OpenAI(api_key=api_key, base_url=base_url)

    @staticmethod
    def _init_anthropic(model_id: str, **kwargs: Any) -> Any:
        from anthropic import Anthropic

        api_key = kwargs.get("api_key")
        return Anthropic(api_key=api_key)

    @staticmethod
    def _init_huggingface(model_id: str, **kwargs: Any) -> Any:
        from transformers import pipeline

        task = kwargs.get("task", "text-generation")
        device = get_device(kwargs.get("device", "auto"))
        device_arg = -1 if device == "cpu" else 0
        return pipeline(task, model=model_id, device=device_arg)

    @staticmethod
    def _init_ollama(model_id: str, **kwargs: Any) -> Any:
        """Return a dict with config; Ollama uses HTTP API."""
        host = kwargs.get("host", "http://localhost:11434")
        return {"model_id": model_id, "host": host}

    # --- Generators ---

    @staticmethod
    def _generate_openai(entry: LLMEntry, prompt: str, **kwargs: Any) -> str:
        response = entry.client.chat.completions.create(
            model=entry.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", 0.7),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return response.choices[0].message.content

    @staticmethod
    def _generate_anthropic(entry: LLMEntry, prompt: str, **kwargs: Any) -> str:
        response = entry.client.messages.create(
            model=entry.model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=kwargs.get("max_tokens", 1024),
            temperature=kwargs.get("temperature", 0.7),
        )
        return response.content[0].text

    @staticmethod
    def _generate_huggingface(entry: LLMEntry, prompt: str, **kwargs: Any) -> str:
        result = entry.client(
            prompt,
            max_new_tokens=kwargs.get("max_tokens", 256),
            temperature=kwargs.get("temperature", 0.7),
            do_sample=True,
        )
        return result[0]["generated_text"]

    @staticmethod
    def _generate_ollama(entry: LLMEntry, prompt: str, **kwargs: Any) -> str:
        import requests

        host = entry.client["host"]
        response = requests.post(
            f"{host}/api/generate",
            json={
                "model": entry.model_id,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 256),
                },
            },
        )
        response.raise_for_status()
        return response.json()["response"]


# ============================================================================
# Tensor Utilities
# ============================================================================


class TensorUtils:
    """Utility functions for tensor operations.

    Provides framework-agnostic tensor creation and conversion.
    """

    @staticmethod
    def create(data: Any, dtype: str = "float32", device: str = "cpu") -> Any:
        """Create a tensor from Python data.

        Args:
            data: List, nested list, or scalar.
            dtype: Data type string ("float32", "float64", "int64", etc.).
            device: Target device.

        Returns:
            PyTorch tensor (preferred) or NumPy array.
        """
        try:
            import torch

            dtype_map = {
                "float32": torch.float32,
                "float64": torch.float64,
                "int32": torch.int32,
                "int64": torch.int64,
                "bool": torch.bool,
            }
            torch_dtype = dtype_map.get(dtype, torch.float32)
            return torch.tensor(data, dtype=torch_dtype, device=device)
        except ImportError:
            import numpy as np

            return np.array(data, dtype=dtype)

    @staticmethod
    def to_list(tensor: Any) -> list:
        """Convert a tensor to a nested Python list."""
        if hasattr(tensor, "tolist"):
            return tensor.tolist()
        if hasattr(tensor, "numpy"):
            return tensor.numpy().tolist()
        return list(tensor)

    @staticmethod
    def shape(tensor: Any) -> tuple:
        """Get tensor shape."""
        if hasattr(tensor, "shape"):
            return tuple(tensor.shape)
        raise TypeError(f"Cannot get shape of {type(tensor)}")

    @staticmethod
    def dtype(tensor: Any) -> str:
        """Get tensor dtype as string."""
        if hasattr(tensor, "dtype"):
            return str(tensor.dtype)
        raise TypeError(f"Cannot get dtype of {type(tensor)}")

    @staticmethod
    def device(tensor: Any) -> str:
        """Get tensor device as string."""
        if hasattr(tensor, "device"):
            return str(tensor.device)
        return "cpu"


# ============================================================================
# Global Instances (used by the Rust bridge)
# ============================================================================

_model_registry = ModelRegistry()
_llm_manager = LLMManager()
_tensor_utils = TensorUtils()


# ============================================================================
# Module-level API (called from Rust FFI via spy_call)
# ============================================================================


def load_model(path: str, kwargs: Optional[dict] = None) -> Any:
    """Load a neural network model.

    Called from Prolog: py_call(Runtime, "load_model", PathHandle, KwargsHandle, Result).

    Args:
        path: Path to the model file.
        kwargs: Dict with options: name, model_type, device, etc.

    Returns:
        The loaded model object (stored as a handle on the Prolog side).
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

    Args:
        model: The model object (handle from load_model).
        input_data: Input tensor/data.
        kwargs: Additional inference options.

    Returns:
        Model output.
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


def load_llm(model_id: str, kwargs: Optional[dict] = None) -> Any:
    """Load an LLM.

    Args:
        model_id: Model identifier (e.g., "gpt-4").
        kwargs: Dict with options: name, provider, api_key, etc.

    Returns:
        LLMEntry object (stored as handle on Prolog side).
    """
    if kwargs is None:
        kwargs = {}
    name = kwargs.pop("name", model_id)
    provider = kwargs.pop("provider", "openai")
    entry = _llm_manager.load(name, model_id, provider=provider, **kwargs)
    return entry


def generate(llm_entry: Any, prompt: str, kwargs: Optional[dict] = None) -> str:
    """Generate text with an LLM.

    Args:
        llm_entry: LLMEntry object from load_llm.
        prompt: Input prompt string.
        kwargs: Generation parameters.

    Returns:
        Generated text.
    """
    if kwargs is None:
        kwargs = {}

    if isinstance(llm_entry, LLMEntry):
        return _llm_manager.generate(llm_entry.name, prompt, **kwargs)

    raise TypeError(f"Expected LLMEntry, got {type(llm_entry)}")


def create_tensor(data: Any, dtype: str = "float32", device: str = "cpu") -> Any:
    """Create a tensor from Python data."""
    return _tensor_utils.create(data, dtype=dtype, device=device)


def tensor_to_list(tensor: Any) -> list:
    """Convert tensor to nested Python list."""
    return _tensor_utils.to_list(tensor)


def tensor_shape(tensor: Any) -> tuple:
    """Get tensor shape."""
    return _tensor_utils.shape(tensor)


def tensor_dtype(tensor: Any) -> str:
    """Get tensor dtype."""
    return _tensor_utils.dtype(tensor)


# ============================================================================
# Initialization
# ============================================================================


def _setup_logging(level: str = "INFO") -> None:
    """Configure logging for ScryNeuro."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="[ScryNeuro] %(levelname)s: %(message)s",
    )


# Auto-setup on import
_setup_logging()
logger.debug("ScryNeuro Python runtime loaded")
