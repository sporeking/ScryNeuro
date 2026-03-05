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
if ! ls "$PYLIB"/libpython${PYVER}*.so* &>/dev/null; then
    echo "WARNING: libpython${PYVER}.so not found in $PYLIB" >&2
    echo "  Python may not have been built with --enable-shared" >&2
fi

# ── Build ──────────────────────────────────────────────────────────
echo "Building ScryNeuro (release)..."
cargo build --release

# ── Copy dynamic library to project root ───────────────────────────
cp target/release/libscryneuro.so ./
echo "Copied libscryneuro.so to project root"

# ── Export environment variables ───────────────────────────────────
export PYLIB
export LD_LIBRARY_PATH=".:$PYLIB:${LD_LIBRARY_PATH:-}"
export SCRYNEURO_HOME="$(pwd)"

echo ""
echo "Build complete. Environment:"
echo "  PYLIB=$PYLIB"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "  SCRYNEURO_HOME=$SCRYNEURO_HOME"
echo ""
echo "To use in current shell, run:  source build_linux.sh"
echo "To run an example:  scryer-prolog examples/basic.pl"
