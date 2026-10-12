use std::{path::Path, process::Command};

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYO3_CONFIG_FILE");
    println!("cargo:rerun-if-env-changed=PYO3_USE_ABI3_FORWARD_COMPATIBILITY");
    println!("cargo:rerun-if-env-changed=PATH");

    // Use the exact configuration shared with PyO3, including its default
    // interpreter discovery and explicit PYO3_CONFIG_FILE selection.
    let config = pyo3_build_config::get();
    let python = config.executable.as_deref().expect(
        "PyO3 configuration has no runnable Python interpreter; ScryNeuro requires a local shared Python library",
    );
    assert!(
        config.shared,
        "PyO3 selected Python without a shared library"
    );
    let output = Command::new(python)
        .args([
            "-c",
            "import sys, sysconfig; print(sys.version.split()[0]); print(sysconfig.get_config_var('LIBDIR') or ''); print(sysconfig.get_config_var('LDLIBRARY') or ''); print(sysconfig.get_config_var('PYTHONFRAMEWORK') or ''); print(sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX') or '')",
        ])
        .output()
        .unwrap_or_else(|err| panic!("Cannot run PyO3-selected Python interpreter {python:?}: {err}"));
    assert!(
        output.status.success(),
        "Cannot query PyO3-selected Python interpreter {python:?}: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    let metadata = String::from_utf8(output.stdout).expect("Python metadata is not UTF-8");
    let fields: Vec<_> = metadata.lines().collect();
    assert!(
        fields.len() == 5
            && fields[..3]
                .iter()
                .all(|field| !field.is_empty() && *field != "None"),
        "Python {python:?} did not report version, LIBDIR and LDLIBRARY"
    );
    let (version, libdir, ldlibrary) = (fields[0], fields[1], fields[2]);
    assert_eq!(
        version.split('.').take(2).collect::<Vec<_>>().join("."),
        config.version.to_string(),
        "Python version differs from PyO3's actual build configuration"
    );
    let mut library = Path::new(libdir).join(ldlibrary);
    if !library.is_file() && !fields[3].is_empty() && fields[3] != "None" {
        assert!(
            !fields[4].is_empty() && fields[4] != "None",
            "Python framework metadata requires PYTHONFRAMEWORKPREFIX"
        );
        library = Path::new(fields[4]).join(ldlibrary);
    }
    assert!(
        library.is_file(),
        "Required Python shared library does not exist: {} (interpreter {python:?})",
        library.display()
    );
    if let Some(config_libdir) = config.lib_dir.as_deref() {
        assert_eq!(
            Path::new(libdir)
                .canonicalize()
                .expect("Cannot resolve Python LIBDIR"),
            Path::new(config_libdir)
                .canonicalize()
                .expect("Cannot resolve PyO3 library directory"),
            "Python LIBDIR differs from PyO3's actual linker directory"
        );
    }
    println!("cargo:warning=Building with Python {version}: interpreter={python}, LIBDIR={libdir}, LDLIBRARY={ldlibrary}, shared library={}", library.display());
    println!("cargo:rustc-link-search=native={libdir}");
    println!(
        "cargo:rustc-env=SCRYNEURO_PYTHON_LIBRARY={}",
        library.display()
    );
    println!("cargo:rustc-env=SCRYNEURO_PYTHON_LDLIBRARY={ldlibrary}");
    println!("cargo:rerun-if-changed={}", library.display());
}
