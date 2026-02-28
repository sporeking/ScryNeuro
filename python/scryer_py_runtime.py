"""
ScryNeuro Python Runtime (Core)
================================

Python-side core module for the ScryNeuro Prolog-Python bridge.
Provides:

- Device management (get_device)
- Tensor utilities (create, convert, inspect)

Neural network, LLM, and RL functionality have been moved to separate plugin modules:
  - scryer_nn_runtime.py  (imported by prolog/scryer_nn.pl)
  - scryer_llm_runtime.py (imported by prolog/scryer_llm.pl)
  - scryer_rl_runtime.py  (imported by prolog/scryer_rl.pl)

This module is imported by the Rust bridge (via PyO3) and called
through the FFI layer from Prolog.

Usage from Prolog (via scryer_py.pl):
    ?- py_import("scryer_py_runtime", RT).
    ?- py_call(RT, "create_tensor", DataHandle, TensorHandle).
"""

from __future__ import annotations

import logging
import sys
from typing import Any

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
# Global Instance (used by the Rust bridge)
# ============================================================================

_tensor_utils = TensorUtils()


# ============================================================================
# Module-level API (called from Rust FFI via spy_call)
# ============================================================================


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
