"""
ScryNeuro LLM Runtime
=====================

Large Language Model runtime module for the ScryNeuro Prolog-Python bridge.
Provides a unified interface for multiple LLM providers: OpenAI, Anthropic,
HuggingFace, Ollama, and custom.

This module is imported by scryer_llm.pl via py_import("scryer_llm_runtime", ...).

Usage from Prolog (via scryer_llm.pl):
    ?- llm_load(gpt, "gpt-4", [provider=openai, api_key="sk-..."]).
    ?- llm_generate(gpt, "What is 2+2?", Response).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from scryer_py_runtime import get_device

logger = logging.getLogger("scryneuro.llm")


# ============================================================================
# LLM Entry
# ============================================================================


@dataclass
class LLMEntry:
    """A registered LLM instance."""

    name: str
    model_id: str
    provider: str  # "openai", "anthropic", "huggingface", "ollama", "custom"
    client: Any  # Provider-specific client object
    metadata: dict = field(default_factory=dict)


# ============================================================================
# LLM Manager
# ============================================================================


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
# Global Instance (used by the Rust bridge via scryer_llm.pl)
# ============================================================================

_llm_manager = LLMManager()


# ============================================================================
# Module-level API (called from Rust FFI via spy_call)
# ============================================================================


def load_llm(model_id: str, kwargs: Optional[dict] = None) -> Any:
    """Load an LLM.

    Called from Prolog: py_call(Runtime, "load_llm", ModelIdHandle, KwargsHandle, Result).

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
