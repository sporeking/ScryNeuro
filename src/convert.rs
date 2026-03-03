//! Type conversion helpers and TLS string buffer for FFI returns.
//!
//! String-returning FFI functions use a thread-local buffer to avoid
//! memory leaks. Scryer copies the `cstr` return value into a Prolog atom
//! before the predicate succeeds, so the buffer can be safely reused.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use pyo3::prelude::*;

// ==================== TLS String Buffer ====================

thread_local! {
    /// Reusable buffer for returning C strings from FFI functions.
    /// Valid until the next `set_return_str` call.
    static RETURN_BUF: RefCell<CString> = RefCell::new(CString::new("").unwrap());
}

/// Store a string in the TLS buffer and return a pointer to it.
///
/// The pointer is valid until the next call to `set_return_str`.
/// Scryer copies the `cstr` into a Prolog term immediately, so this is safe.
pub fn set_return_str(s: String) -> *const c_char {
    RETURN_BUF.with(|cell| {
        let cstr = CString::new(s)
            .unwrap_or_else(|_| CString::new("<string contains null byte>").unwrap());
        *cell.borrow_mut() = cstr;
        cell.borrow().as_ptr()
    })
}

// ==================== C String Helpers ====================

/// Safely convert a `*const c_char` to `&str`.
///
/// # Safety
/// The pointer must be valid, null-terminated, and point to valid UTF-8.
pub unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("Null string pointer".into());
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|e| format!("Invalid UTF-8: {e}"))
}

// ==================== Python Error Formatting ====================

/// Convert a `PyErr` into a human-readable error string,
/// including traceback if available.
pub fn pyerr_to_string(py: Python<'_>, err: PyErr) -> String {
    let msg = format!("{err}");
    if let Some(tb) = err.traceback(py) {
        if let Ok(formatted) = tb.format() {
            return format!("{formatted}{msg}");
        }
    }
    msg
}

// ==================== JSON Bridge ====================

/// Convert a Python object to a JSON string using `json.dumps()`.
pub fn py_to_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    let json_mod = py.import("json")?;
    let result = json_mod.call_method1("dumps", (obj,))?;
    result.extract::<String>()
}

/// Parse a JSON string into a Python object using `json.loads()`.
pub fn json_to_py<'py>(py: Python<'py>, json_str: &str) -> PyResult<Bound<'py, PyAny>> {
    let json_mod = py.import("json")?;
    json_mod.call_method1("loads", (json_str,))
}
