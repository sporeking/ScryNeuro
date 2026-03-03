//! ScryNeuro: Scryer Prolog ↔ Python bridge for neuro-symbolic AI.
//!
//! This crate builds a `cdylib` (.so) loaded by Scryer Prolog via `use_foreign_module/2`.
//! It embeds a Python runtime using PyO3 and exposes C ABI functions for:
//!
//! - Python object lifecycle (eval, import, call, drop)
//! - Type conversion (int, float, bool, string, JSON)
//! - Collection operations (list, dict)
//! - Error handling (thread-local last-error pattern)
//!
//! The Prolog layer (`prolog/scryer_py.pl`) wraps these into ergonomic predicates.
//! The Python layer (`python/scryer_py_runtime.py`) provides model/LLM/tensor management.

pub mod convert;
pub mod error;
pub mod ffi;
pub mod registry;
