//! Thread-local error handling for FFI boundary.
//!
//! All `spy_*` functions set a thread-local error string on failure.
//! Retrieve with `spy_last_error()`, clear with `spy_last_error_clear()`.
//!
//! **Ownership rules:**
//! - `spy_last_error()` returns a pointer owned by TLS — do NOT free it.
//! - `spy_to_str()` / `spy_to_json()` etc. return leaked CStrings — MUST free with `spy_cstr_free()`.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

/// Store an error message in thread-local storage.
pub fn set_last_error(msg: impl Into<String>) {
    let msg = msg.into();
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

/// Clear the thread-local error.
pub fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

/// Get a pointer to the last error message, or null if none.
///
/// The returned pointer is valid until the next `spy_*` call.
/// Do NOT free this pointer.
#[no_mangle]
pub extern "C" fn spy_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        cell.borrow()
            .as_ref()
            .map_or(std::ptr::null(), |s| s.as_ptr())
    })
}

/// Clear the last error message.
#[no_mangle]
pub extern "C" fn spy_last_error_clear() {
    clear_last_error();
}

/// Free a CString that was returned by `spy_to_str`, `spy_to_json`, etc.
///
/// Do NOT call this on pointers returned by `spy_last_error()`.
///
/// # Safety
/// `ptr` must be a pointer previously returned by a `spy_*` function
/// that documents "must free with `spy_cstr_free`", or null.
#[no_mangle]
pub unsafe extern "C" fn spy_cstr_free(ptr: *mut c_char) {
    if !ptr.is_null() {
        drop(unsafe { CString::from_raw(ptr) });
    }
}
