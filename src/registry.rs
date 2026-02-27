//! Handle registry: maps integer handles to Python objects.
//!
//! Handles are monotonically increasing `isize` values starting from 1.
//! Handle 0 is reserved as the null/error sentinel.
//!
//! All operations are `Mutex`-protected for thread safety.
//! Python objects (`Py<PyAny>`) are GIL-independent and can be stored
//! without holding the GIL, but the GIL must be held when dropping them.

use std::collections::HashMap;
use std::sync::Mutex;

use pyo3::prelude::*;

/// Global handle registry, initialized by `spy_init()`.
static REGISTRY: Mutex<Option<HandleRegistry>> = Mutex::new(None);

struct HandleRegistry {
    objects: HashMap<isize, Py<PyAny>>,
    next_id: isize,
}

impl HandleRegistry {
    fn new() -> Self {
        Self {
            objects: HashMap::new(),
            next_id: 1,
        }
    }
}

// ==================== Public API ====================

/// Initialize the registry. Called once from `spy_init()`.
pub fn init_registry() {
    let mut guard = REGISTRY.lock().expect("Registry mutex poisoned");
    *guard = Some(HandleRegistry::new());
}

/// Destroy the registry, dropping all Python objects.
/// Must be called with the GIL held so Py<PyAny> destructors can run.
pub fn destroy_registry() {
    let mut guard = REGISTRY.lock().expect("Registry mutex poisoned");
    *guard = None;
}

/// Insert a Python object and return its handle.
pub fn insert(obj: Py<PyAny>) -> Result<isize, String> {
    let mut guard = REGISTRY.lock().map_err(|e| format!("Registry lock: {e}"))?;
    let reg = guard
        .as_mut()
        .ok_or("ScryNeuro not initialized. Call spy_init() first.")?;
    let id = reg.next_id;
    reg.next_id = reg
        .next_id
        .checked_add(1)
        .ok_or("Handle counter overflow")?;
    reg.objects.insert(id, obj);
    Ok(id)
}

/// Clone the `Py<PyAny>` for a given handle (the handle remains valid).
/// Requires the GIL to be held because `Py<PyAny>::clone_ref` needs it.
pub fn get(py: Python<'_>, handle: isize) -> Result<Py<PyAny>, String> {
    let guard = REGISTRY.lock().map_err(|e| format!("Registry lock: {e}"))?;
    let reg = guard
        .as_ref()
        .ok_or("ScryNeuro not initialized. Call spy_init() first.")?;
    reg.objects
        .get(&handle)
        .map(|obj| obj.clone_ref(py))
        .ok_or_else(|| format!("Invalid handle: {handle}"))
}

/// Remove a handle and return the owned `Py<PyAny>`.
/// The caller should drop it with the GIL held.
pub fn remove(handle: isize) -> Result<Py<PyAny>, String> {
    let mut guard = REGISTRY.lock().map_err(|e| format!("Registry lock: {e}"))?;
    let reg = guard
        .as_mut()
        .ok_or("ScryNeuro not initialized. Call spy_init() first.")?;
    reg.objects
        .remove(&handle)
        .ok_or_else(|| format!("Invalid handle: {handle}"))
}

/// Execute a closure with a `Bound<'py, PyAny>` for the given handle.
/// The GIL must already be held (`py` token required).
///
/// Clones the object reference and releases the registry lock before
/// calling the closure, so the closure can safely call `registry::insert()`
/// or other registry operations without deadlocking.
pub fn with_object<'py, F, R>(py: Python<'py>, handle: isize, f: F) -> Result<R, String>
where
    F: FnOnce(&Bound<'py, PyAny>) -> Result<R, String>,
{
    // Clone the Py<PyAny> while holding the lock, then release it.
    let cloned = {
        let guard = REGISTRY.lock().map_err(|e| format!("Registry lock: {e}"))?;
        let reg = guard
            .as_ref()
            .ok_or("ScryNeuro not initialized. Call spy_init() first.")?;
        reg.objects
            .get(&handle)
            .ok_or_else(|| format!("Invalid handle: {handle}"))?
            .clone_ref(py)
    }; // guard dropped here — lock released
    f(cloned.bind(py))
}

/// Return the number of live handles (for diagnostics).
pub fn len() -> Result<usize, String> {
    let guard = REGISTRY.lock().map_err(|e| format!("Registry lock: {e}"))?;
    let reg = guard
        .as_ref()
        .ok_or("ScryNeuro not initialized. Call spy_init() first.")?;
    Ok(reg.objects.len())
}
