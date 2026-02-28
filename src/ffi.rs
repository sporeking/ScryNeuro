//! C ABI exports for Scryer Prolog FFI.
//!
//! Every `spy_*` function follows this contract:
//!
//! 1. Clears the last error
//! 2. Acquires the Python GIL
//! 3. Performs the operation
//! 4. On error: stores message via `set_last_error`, returns sentinel
//! 5. On success: returns the result
//!
//! **Sentinel values:**
//! - Handle functions (`-> isize`): `0` means error
//! - Status functions (`-> i32`):   `-1` means error, `0` means success
//! - String functions (`-> cstr`):  `""` means error (check `spy_last_error`)
//!
//! **Memory:**
//! - Returned `cstr` pointers use a TLS buffer — valid until the next `spy_*` call.
//!   Scryer copies the string into a Prolog atom before the predicate succeeds.
//! - Handles (`isize`) are opaque references to Python objects in the registry.
//!   They MUST be freed with `spy_drop` when no longer needed.

use std::ffi::CString;
use std::os::raw::c_char;

use pyo3::prelude::*;
use pyo3::types::*;

use crate::convert;
use crate::error::{clear_last_error, set_last_error};
use crate::registry;

// ==================== Internal Helpers ====================

/// Acquire GIL, run closure, return handle (0 on error).
fn gil_handle(f: impl FnOnce(Python<'_>) -> Result<isize, String>) -> isize {
    clear_last_error();
    Python::with_gil(|py| match f(py) {
        Ok(h) => h,
        Err(e) => {
            set_last_error(e);
            0
        }
    })
}

/// Acquire GIL, run closure, return status (0=ok, -1=error).
fn gil_status(f: impl FnOnce(Python<'_>) -> Result<(), String>) -> i32 {
    clear_last_error();
    Python::with_gil(|py| match f(py) {
        Ok(()) => 0,
        Err(e) => {
            set_last_error(e);
            -1
        }
    })
}

/// Acquire GIL, run closure, return string via TLS buffer ("" on error).
fn gil_str(f: impl FnOnce(Python<'_>) -> Result<String, String>) -> *const c_char {
    clear_last_error();
    Python::with_gil(|py| match f(py) {
        Ok(s) => convert::set_return_str(s),
        Err(e) => {
            set_last_error(e);
            convert::set_return_str(String::new())
        }
    })
}

/// Shorthand: convert PyErr → descriptive String.
fn pe(py: Python<'_>, e: PyErr) -> String {
    convert::pyerr_to_string(py, e)
}

/// Shorthand: read a C string argument (unsafe).
///
/// # Safety
/// Pointer must be valid, null-terminated, UTF-8.
unsafe fn arg_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    unsafe { convert::cstr_to_str(ptr) }
}

/// Insert a `Bound<'py, PyAny>` into the registry, returning its handle.
fn reg_insert(obj: Bound<'_, PyAny>) -> Result<isize, String> {
    registry::insert(obj.unbind())
}

fn args_handle_to_tuple<'py>(py: Python<'py>, args: isize) -> Result<Bound<'py, PyTuple>, String> {
    let args_obj = registry::get(py, args)?;
    let args_bound = args_obj.bind(py);
    match args_bound.downcast::<PySequence>() {
        Ok(seq) => seq.to_tuple().map_err(|e| pe(py, e)),
        Err(_) => Err("Args must be a Python sequence".to_string()),
    }
}

// ==================== Lifecycle ====================

/// Initialize the Python runtime and handle registry.
///
/// On Linux, attempts to re-open `libpython` with `RTLD_GLOBAL` so that
/// Python C extensions (NumPy, PyTorch) can resolve Python symbols.
///
/// Returns `0` on success, `-1` on error.
#[no_mangle]
pub extern "C" fn spy_init() -> i32 {
    clear_last_error();

    // RTLD_GLOBAL: make Python symbols globally visible for C extensions.
    #[cfg(target_os = "linux")]
    {
        unsafe {
            use libc::{dlopen, RTLD_GLOBAL, RTLD_NOLOAD, RTLD_NOW};
            let candidates: &[&[u8]] = &[
                b"libpython3.so\0",
                b"libpython3.13.so\0",
                b"libpython3.12.so\0",
                b"libpython3.11.so\0",
                b"libpython3.10.so\0",
            ];
            for name in candidates {
                let ptr = dlopen(name.as_ptr() as *const i8, RTLD_NOW | RTLD_NOLOAD);
                if !ptr.is_null() {
                    dlopen(name.as_ptr() as *const i8, RTLD_NOW | RTLD_GLOBAL);
                    break;
                }
            }
        }
    }

    // Initialize Python without installing signal handlers (embedding mode).
    pyo3::prepare_freethreaded_python();
    registry::init_registry();

    // Add current working directory and python/ subdirectory to sys.path.
    // This allows importing scryer_py_runtime and scryer_rl_runtime without
    // requiring the user to set PYTHONPATH.
    let result: PyResult<()> = Python::with_gil(|py| {
        let sys = py.import("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("insert", (0i32, "."))?;
        path.call_method1("insert", (0i32, "./python"))?;
        Ok(())
    });

    match result {
        Ok(()) => 0,
        Err(e) => {
            Python::with_gil(|py| set_last_error(pe(py, e)));
            -1
        }
    }
}

/// Finalize: drop all Python object handles and clean up.
#[no_mangle]
pub extern "C" fn spy_finalize() {
    clear_last_error();
    Python::with_gil(|_py| {
        registry::destroy_registry();
    });
}

// ==================== Evaluation ====================

/// Evaluate a Python expression and return a handle to the result.
///
/// Returns `0` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_eval(code: *const c_char) -> isize {
    let code_str = match unsafe { arg_str(code) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let c_code = CString::new(code_str).map_err(|e| format!("CString: {e}"))?;
        let result = py
            .eval(c_code.as_c_str(), None, None)
            .map_err(|e| pe(py, e))?;
        reg_insert(result.into_any())
    })
}

/// Execute Python statements (no return value).
///
/// Returns `0` on success, `-1` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_exec(code: *const c_char) -> i32 {
    let code_str = match unsafe { arg_str(code) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return -1;
        }
    };
    gil_status(|py| {
        let c_code = CString::new(code_str).map_err(|e| format!("CString: {e}"))?;
        py.run(c_code.as_c_str(), None, None).map_err(|e| pe(py, e))
    })
}

// ==================== Modules ====================

/// Import a Python module and return a handle.
///
/// Returns `0` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_import(module_name: *const c_char) -> isize {
    let name = match unsafe { arg_str(module_name) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let module = py.import(name.as_str()).map_err(|e| pe(py, e))?;
        reg_insert(module.into_any())
    })
}

// ==================== Attribute Access ====================

/// Get an attribute from a Python object.
///
/// Returns `0` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_getattr(obj: isize, name: *const c_char) -> isize {
    let attr_name = match unsafe { arg_str(name) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        registry::with_object(py, obj, |bound| {
            let attr = bound.getattr(&*attr_name).map_err(|e| pe(py, e))?;
            reg_insert(attr)
        })
    })
}

/// Set an attribute on a Python object.
///
/// Returns `0` on success, `-1` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_setattr(obj: isize, name: *const c_char, value: isize) -> i32 {
    let attr_name = match unsafe { arg_str(name) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return -1;
        }
    };
    gil_status(|py| {
        let val = registry::get(py, value)?;
        registry::with_object(py, obj, |bound| {
            bound
                .setattr(&*attr_name, val.bind(py))
                .map_err(|e| pe(py, e))
        })
    })
}

// ==================== Direct Calls (callable objects) ====================

/// Call a callable handle with no arguments: `callable()`.
/// FFI: spy_call0(ptr) -> ptr
#[no_mangle]
pub extern "C" fn spy_call0(callable: isize) -> isize {
    gil_handle(|py| {
        registry::with_object(py, callable, |bound| {
            let result = bound.call0().map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a callable handle with 1 argument: `callable(arg1)`.
/// FFI: spy_call1(ptr, ptr) -> ptr
#[no_mangle]
pub extern "C" fn spy_call1(callable: isize, arg1: isize) -> isize {
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        registry::with_object(py, callable, |bound| {
            let result = bound.call1((a1.bind(py),)).map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a callable handle with 2 arguments: `callable(arg1, arg2)`.
/// FFI: spy_call2(ptr, ptr, ptr) -> ptr
#[no_mangle]
pub extern "C" fn spy_call2(callable: isize, arg1: isize, arg2: isize) -> isize {
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        let a2 = registry::get(py, arg2)?;
        registry::with_object(py, callable, |bound| {
            let result = bound
                .call1((a1.bind(py), a2.bind(py)))
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a callable handle with 3 arguments: `callable(arg1, arg2, arg3)`.
/// FFI: spy_call3(ptr, ptr, ptr, ptr) -> ptr
#[no_mangle]
pub extern "C" fn spy_call3(callable: isize, arg1: isize, arg2: isize, arg3: isize) -> isize {
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        let a2 = registry::get(py, arg2)?;
        let a3 = registry::get(py, arg3)?;
        registry::with_object(py, callable, |bound| {
            let result = bound
                .call1((a1.bind(py), a2.bind(py), a3.bind(py)))
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

#[no_mangle]
pub extern "C" fn spy_calln(callable: isize, args: isize) -> isize {
    gil_handle(|py| {
        let args_tuple = args_handle_to_tuple(py, args)?;
        registry::with_object(py, callable, |bound| {
            let result = bound.call1(args_tuple).map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

// ==================== Method Invocations ====================

/// Call a method with no arguments: `obj.method()`.
/// FFI: spy_invoke0(ptr, cstr) -> ptr
#[no_mangle]
pub unsafe extern "C" fn spy_invoke0(obj: isize, method: *const c_char) -> isize {
    let meth = match unsafe { arg_str(method) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        registry::with_object(py, obj, |bound| {
            let result = bound.call_method0(&*meth).map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a method with 1 argument: `obj.method(arg1)`.
/// FFI: spy_invoke1(ptr, cstr, ptr) -> ptr
#[no_mangle]
pub unsafe extern "C" fn spy_invoke1(obj: isize, method: *const c_char, arg1: isize) -> isize {
    let meth = match unsafe { arg_str(method) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        registry::with_object(py, obj, |bound| {
            let result = bound
                .call_method1(&*meth, (a1.bind(py),))
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a method with 2 arguments: `obj.method(arg1, arg2)`.
/// FFI: spy_invoke2(ptr, cstr, ptr, ptr) -> ptr
#[no_mangle]
pub unsafe extern "C" fn spy_invoke2(
    obj: isize,
    method: *const c_char,
    arg1: isize,
    arg2: isize,
) -> isize {
    let meth = match unsafe { arg_str(method) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        let a2 = registry::get(py, arg2)?;
        registry::with_object(py, obj, |bound| {
            let result = bound
                .call_method1(&*meth, (a1.bind(py), a2.bind(py)))
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

/// Call a method with 3 arguments: `obj.method(arg1, arg2, arg3)`.
/// FFI: spy_invoke3(ptr, cstr, ptr, ptr, ptr) -> ptr
#[no_mangle]
pub unsafe extern "C" fn spy_invoke3(
    obj: isize,
    method: *const c_char,
    arg1: isize,
    arg2: isize,
    arg3: isize,
) -> isize {
    let meth = match unsafe { arg_str(method) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let a1 = registry::get(py, arg1)?;
        let a2 = registry::get(py, arg2)?;
        let a3 = registry::get(py, arg3)?;
        registry::with_object(py, obj, |bound| {
            let result = bound
                .call_method1(&*meth, (a1.bind(py), a2.bind(py), a3.bind(py)))
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

#[no_mangle]
pub unsafe extern "C" fn spy_invoken(obj: isize, method: *const c_char, args: isize) -> isize {
    let meth = match unsafe { arg_str(method) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let args_tuple = args_handle_to_tuple(py, args)?;
        registry::with_object(py, obj, |bound| {
            let result = bound
                .call_method1(&*meth, args_tuple)
                .map_err(|e| pe(py, e))?;
            reg_insert(result)
        })
    })
}

// ==================== Type Conversion: Python → C ====================

/// Get `str(obj)`. Returns via TLS buffer — valid until next `spy_*` call.
#[no_mangle]
pub extern "C" fn spy_to_str(obj: isize) -> *const c_char {
    gil_str(|py| {
        registry::with_object(py, obj, |bound| {
            let s = bound.str().map_err(|e| pe(py, e))?;
            s.to_str()
                .map(|s| s.to_owned())
                .map_err(|e| format!("str conversion: {e}"))
        })
    })
}

/// Get `repr(obj)`. Returns via TLS buffer.
#[no_mangle]
pub extern "C" fn spy_to_repr(obj: isize) -> *const c_char {
    gil_str(|py| {
        registry::with_object(py, obj, |bound| {
            let s = bound.repr().map_err(|e| pe(py, e))?;
            s.to_str()
                .map(|s| s.to_owned())
                .map_err(|e| format!("repr conversion: {e}"))
        })
    })
}

/// Extract an integer from a Python object.
///
/// Returns `0` on error — check `spy_last_error` to distinguish from a real zero.
#[no_mangle]
pub extern "C" fn spy_to_int(obj: isize) -> i64 {
    clear_last_error();
    Python::with_gil(|py| {
        match registry::with_object(py, obj, |bound| {
            bound.extract::<i64>().map_err(|e| pe(py, e))
        }) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(e);
                0
            }
        }
    })
}

/// Extract a float from a Python object.
///
/// Returns `0.0` on error — check `spy_last_error`.
#[no_mangle]
pub extern "C" fn spy_to_float(obj: isize) -> f64 {
    clear_last_error();
    Python::with_gil(|py| {
        match registry::with_object(py, obj, |bound| {
            bound.extract::<f64>().map_err(|e| pe(py, e))
        }) {
            Ok(v) => v,
            Err(e) => {
                set_last_error(e);
                0.0
            }
        }
    })
}

/// Extract a boolean from a Python object.
///
/// Returns `1` for true, `0` for false, `-1` on error.
#[no_mangle]
pub extern "C" fn spy_to_bool(obj: isize) -> i32 {
    clear_last_error();
    Python::with_gil(|py| {
        match registry::with_object(py, obj, |bound| {
            bound.extract::<bool>().map_err(|e| pe(py, e))
        }) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    })
}

// ==================== Type Conversion: C → Python ====================

/// Create a Python `int` from a Rust `i64`.
#[no_mangle]
pub extern "C" fn spy_from_int(val: i64) -> isize {
    gil_handle(|py| {
        let obj = val
            .into_pyobject(py)
            .map_err(|e| format!("into_pyobject: {e}"))?
            .into_any()
            .unbind();
        registry::insert(obj)
    })
}

/// Create a Python `float` from a Rust `f64`.
#[no_mangle]
pub extern "C" fn spy_from_float(val: f64) -> isize {
    gil_handle(|py| {
        let obj = val
            .into_pyobject(py)
            .map_err(|e| format!("into_pyobject: {e}"))?
            .into_any()
            .unbind();
        registry::insert(obj)
    })
}

/// Create a Python `bool` from a Rust `i32` (0 = False, nonzero = True).
#[no_mangle]
pub extern "C" fn spy_from_bool(val: i32) -> isize {
    gil_handle(|py| {
        let obj = PyBool::new(py, val != 0).to_owned().into_any().unbind();
        registry::insert(obj)
    })
}

/// Create a Python `str` from a C string.
#[no_mangle]
pub unsafe extern "C" fn spy_from_str(val: *const c_char) -> isize {
    let s = match unsafe { arg_str(val) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let obj = s
            .into_pyobject(py)
            .map_err(|e| format!("into_pyobject: {e}"))?
            .into_any()
            .unbind();
        registry::insert(obj)
    })
}

// ==================== None ====================

/// Get a handle to Python `None`.
#[no_mangle]
pub extern "C" fn spy_none() -> isize {
    gil_handle(|py| {
        let obj = py.None();
        registry::insert(obj)
    })
}

/// Check if a handle points to Python `None`.
///
/// Returns `1` if None, `0` if not, `-1` on error.
#[no_mangle]
pub extern "C" fn spy_is_none(obj: isize) -> i32 {
    clear_last_error();
    Python::with_gil(
        |py| match registry::with_object(py, obj, |bound| Ok(bound.is_none())) {
            Ok(true) => 1,
            Ok(false) => 0,
            Err(e) => {
                set_last_error(e);
                -1
            }
        },
    )
}

// ==================== JSON Bridge ====================

/// Serialize a Python object to JSON. Returns via TLS buffer.
///
/// Uses Python's `json.dumps()` internally.
#[no_mangle]
pub extern "C" fn spy_to_json(obj: isize) -> *const c_char {
    gil_str(|py| {
        registry::with_object(py, obj, |bound| {
            convert::py_to_json(py, bound).map_err(|e| pe(py, e))
        })
    })
}

/// Deserialize a JSON string to a Python object.
///
/// Uses Python's `json.loads()` internally.
#[no_mangle]
pub unsafe extern "C" fn spy_from_json(json: *const c_char) -> isize {
    let json_str = match unsafe { arg_str(json) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        let obj = convert::json_to_py(py, &json_str).map_err(|e| pe(py, e))?;
        reg_insert(obj)
    })
}

// ==================== Collections ====================

/// Create a new empty Python `list`.
#[no_mangle]
pub extern "C" fn spy_list_new() -> isize {
    gil_handle(|py| {
        let list = PyList::empty(py);
        reg_insert(list.into_any())
    })
}

/// Append an item to a Python list.
///
/// Returns `0` on success, `-1` on error.
#[no_mangle]
pub extern "C" fn spy_list_append(list: isize, item: isize) -> i32 {
    gil_status(|py| {
        let item_obj = registry::get(py, item)?;
        registry::with_object(py, list, |bound| {
            let pylist: &Bound<'_, PyList> =
                bound.downcast().map_err(|_| "Not a list".to_string())?;
            pylist.append(item_obj.bind(py)).map_err(|e| pe(py, e))
        })
    })
}

/// Get an item from a Python list by index.
///
/// Returns `0` on error.
#[no_mangle]
pub extern "C" fn spy_list_get(list: isize, index: i64) -> isize {
    gil_handle(|py| {
        registry::with_object(py, list, |bound| {
            let pylist: &Bound<'_, PyList> =
                bound.downcast().map_err(|_| "Not a list".to_string())?;
            let item = pylist.get_item(index as usize).map_err(|e| pe(py, e))?;
            reg_insert(item)
        })
    })
}

/// Get the length of a Python list.
///
/// Returns `-1` on error.
#[no_mangle]
pub extern "C" fn spy_list_len(list: isize) -> i64 {
    clear_last_error();
    Python::with_gil(|py| {
        match registry::with_object(py, list, |bound| {
            let pylist: &Bound<'_, PyList> =
                bound.downcast().map_err(|_| "Not a list".to_string())?;
            Ok(pylist.len() as i64)
        }) {
            Ok(n) => n,
            Err(e) => {
                set_last_error(e);
                -1
            }
        }
    })
}

/// Create a new empty Python `dict`.
#[no_mangle]
pub extern "C" fn spy_dict_new() -> isize {
    gil_handle(|py| {
        let dict = PyDict::new(py);
        reg_insert(dict.into_any())
    })
}

/// Set a key-value pair in a Python dict (key is a string).
///
/// Returns `0` on success, `-1` on error.
#[no_mangle]
pub unsafe extern "C" fn spy_dict_set(dict: isize, key: *const c_char, value: isize) -> i32 {
    let key_str = match unsafe { arg_str(key) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return -1;
        }
    };
    gil_status(|py| {
        let val_obj = registry::get(py, value)?;
        registry::with_object(py, dict, |bound| {
            let pydict: &Bound<'_, PyDict> =
                bound.downcast().map_err(|_| "Not a dict".to_string())?;
            pydict
                .set_item(&*key_str, val_obj.bind(py))
                .map_err(|e| pe(py, e))
        })
    })
}

/// Get a value from a Python dict by string key.
///
/// Returns `0` if key not found or on error.
#[no_mangle]
pub unsafe extern "C" fn spy_dict_get(dict: isize, key: *const c_char) -> isize {
    let key_str = match unsafe { arg_str(key) } {
        Ok(s) => s.to_owned(),
        Err(e) => {
            clear_last_error();
            set_last_error(e);
            return 0;
        }
    };
    gil_handle(|py| {
        registry::with_object(py, dict, |bound| {
            let pydict: &Bound<'_, PyDict> =
                bound.downcast().map_err(|_| "Not a dict".to_string())?;
            match pydict.get_item(&*key_str).map_err(|e| pe(py, e))? {
                Some(val) => reg_insert(val),
                None => Err(format!("Key not found: {key_str}")),
            }
        })
    })
}

// ==================== Memory Management ====================

/// Release a Python object handle.
///
/// After this call, the handle is invalid. Using it will produce an error.
/// The underlying Python object is decref'd (potentially freed).
#[no_mangle]
pub extern "C" fn spy_drop(handle: isize) {
    clear_last_error();
    Python::with_gil(|_py| {
        if let Err(e) = registry::remove(handle) {
            set_last_error(e);
        }
        // The Py<PyAny> is dropped here, decrementing the Python refcount.
        // The GIL is held, so this is safe.
    });
}

/// Return the number of live handles (for debugging/diagnostics).
#[no_mangle]
pub extern "C" fn spy_handle_count() -> i64 {
    match registry::len() {
        Ok(n) => n as i64,
        Err(e) => {
            set_last_error(e);
            -1
        }
    }
}
