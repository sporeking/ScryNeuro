use std::env;
use std::path::PathBuf;
use std::process::Command;

fn python_executable() -> PathBuf {
    if let Some(value) = env::var_os("PYO3_PYTHON").filter(|value| !value.is_empty()) {
        return PathBuf::from(value);
    }

    if let Some(venv) = env::var_os("VIRTUAL_ENV") {
        let candidate = if cfg!(windows) {
            PathBuf::from(venv).join("Scripts").join("python.exe")
        } else {
            PathBuf::from(venv).join("bin").join("python")
        };
        if candidate.is_file() {
            return candidate;
        }
    }

    if cfg!(windows) {
        PathBuf::from("python.exe")
    } else {
        PathBuf::from("python3")
    }
}

fn main() {
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");

    // Detect Python version and configure library path for linking.
    // PyO3's build script handles most of this, but we add extra guidance
    // for finding libpython on Unix when building the cdylib.
    let python = python_executable();

    // Print Python version for build diagnostics
    if let Ok(output) = Command::new(&python).args(["--version"]).output() {
        if output.status.success() {
            let version_output = if output.stdout.is_empty() {
                &output.stderr
            } else {
                &output.stdout
            };
            let version = String::from_utf8_lossy(version_output);
            println!(
                "cargo:warning=Building with {} ({})",
                version.trim(),
                python.display()
            );
        }
    }

    // Windows links against python3xx.lib/python3xx.dll. PyO3 performs that
    // setup itself using PYO3_PYTHON; Unix additionally benefits from LIBDIR.
    let target_is_windows = env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows");
    if !target_is_windows {
        // Get Python's lib directory and add it to the linker search path.
        if let Ok(output) = Command::new(&python)
            .args([
                "-c",
                "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))",
            ])
            .output()
        {
            if output.status.success() {
                let libdir = String::from_utf8_lossy(&output.stdout);
                let libdir = libdir.trim();
                if !libdir.is_empty() && libdir != "None" {
                    println!("cargo:rustc-link-search=native={libdir}");
                }
            }
        }
    }
}
