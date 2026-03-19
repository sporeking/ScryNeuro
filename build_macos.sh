#!/bin/bash
set -euo pipefail

# ── Detect Python shared library path ──────────────────────────────
PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
if [ -z "$PYLIB" ] || [ "$PYLIB" = "None" ]; then
    echo "ERROR: Cannot detect Python LIBDIR. Is python3 in PATH?" >&2
    exit 1
fi
echo "Python LIBDIR: $PYLIB"

# ── Verify libpython exists ────────────────────────────────────────
PYVER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYMAJOR=$(python3 -c "import sys; print(sys.version_info.major)")
PYMINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if ! ls "$PYLIB"/libpython${PYVER}*.dylib &>/dev/null; then
    echo "WARNING: libpython${PYVER}.dylib not found in $PYLIB" >&2
    echo "  If using pyenv, rebuild with: PYTHON_CONFIGURE_OPTS='--enable-framework' pyenv install ${PYVER}" >&2
fi

if [ "$PYMAJOR" -eq 3 ] && [ "$PYMINOR" -ge 14 ]; then
    export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
    echo "Detected Python ${PYVER}; enabling PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1"
fi

# ── Build ──────────────────────────────────────────────────────────
echo "Building ScryNeuro (release)..."
cargo build --release
unset PYO3_USE_ABI3_FORWARD_COMPATIBILITY || true

# ── Copy dynamic library to project root ───────────────────────────
cp target/release/libscryneuro.dylib ./
echo "Copied libscryneuro.dylib to project root"

# ── Embed rpath so the dylib resolves libpython at its loader path ──
# Workaround for macOS SIP stripping DYLD_LIBRARY_PATH in some contexts.
install_name_tool -add_rpath "@loader_path/." libscryneuro.dylib 2>/dev/null || true
install_name_tool -add_rpath "$PYLIB"         libscryneuro.dylib 2>/dev/null || true

# ── Export environment variables ───────────────────────────────────
export PYLIB
export DYLD_LIBRARY_PATH=".:$PYLIB:${DYLD_LIBRARY_PATH:-}"
export SCRYNEURO_HOME="$(pwd)"

echo ""
echo "Build complete. Environment:"
echo "  PYLIB=$PYLIB"
echo "  DYLD_LIBRARY_PATH=$DYLD_LIBRARY_PATH"
echo "  SCRYNEURO_HOME=$SCRYNEURO_HOME"
if [ "${PYO3_USE_ABI3_FORWARD_COMPATIBILITY:-}" = "1" ]; then
    echo "  PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1"
fi
echo ""
echo "To use in current shell, run:  source build_macos.sh"
echo "To run an example:  scryer-prolog examples/basic.pl"
