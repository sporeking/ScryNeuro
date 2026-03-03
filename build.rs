use std::process::Command;

fn main() {
    // Detect Python version and configure library path for linking.
    // PyO3's build script handles most of this, but we add extra guidance
    // for finding libpython when building the cdylib.

    // Print Python version for build diagnostics
    if let Ok(output) = Command::new("python3").args(["--version"]).output() {
        if output.status.success() {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("cargo:warning=Building with {}", version.trim());
        }
    }

    // Get Python's lib directory and add it to the linker search path
    if let Ok(output) = Command::new("python3")
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
