#!/bin/bash

# Keep caller options and PyO3 settings intact when this file is sourced.
_scryneuro_build_linux() {
    local python="${PYO3_PYTHON:-python3}" metadata version major minor libdir ldlibrary home
    if ! metadata=$("$python" -c "import sys, sysconfig; print(sys.version.split()[0]); print(sysconfig.get_config_var('LIBDIR') or ''); print(sysconfig.get_config_var('LDLIBRARY') or '')"); then
        echo "ERROR: Cannot query Python interpreter: $python" >&2
        return 1
    fi
    {
        IFS= read -r version
        IFS= read -r libdir
        IFS= read -r ldlibrary
    } <<< "$metadata"
    if [ -z "$version" ] || [ -z "$libdir" ] || [ "$libdir" = None ] || [ -z "$ldlibrary" ] || [ "$ldlibrary" = None ]; then
        echo "ERROR: Python $python must report version, LIBDIR and LDLIBRARY" >&2
        return 1
    fi
    case "$ldlibrary" in
        *.a) echo "ERROR: Python LDLIBRARY is static, a shared library is required: $libdir/$ldlibrary" >&2; return 1 ;;
    esac
    if [ ! -f "$libdir/$ldlibrary" ]; then
        echo "ERROR: Required Python shared library not found: $libdir/$ldlibrary (interpreter: $python)" >&2
        return 1
    fi
    major=${version%%.*}
    minor=${version#*.}
    minor=${minor%%.*}
    case "$major:$minor" in
        *[!0-9:]*|:*|*:) echo "ERROR: Invalid Python version: $version" >&2; return 1 ;;
    esac
    home=$(pwd) || return 1
    echo "Python interpreter: $python"
    echo "Python version: $version"
    echo "Python LIBDIR: $libdir"
    echo "Python LDLIBRARY: $ldlibrary"
    echo "Using Python shared library: $libdir/$ldlibrary"
    (
        set -euo pipefail
        # PyO3 normally tries 'python' first: pin it to the interpreter probed above.
        export PYO3_PYTHON="$python"
        if [ "${PYO3_USE_ABI3_FORWARD_COMPATIBILITY+x}" = x ]; then
            export PYO3_USE_ABI3_FORWARD_COMPATIBILITY
        elif [ "$major" -eq 3 ] && [ "$minor" -ge 14 ]; then
            export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
            echo "Detected Python $version; enabling PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 for this build"
        fi
        echo "Building ScryNeuro (release)..."
        # Explicit checks also work if the caller sources us inside an if/|| list,
        # where Bash suppresses errexit even in a function's subshell.
        cargo build --release || exit "$?"
        cp target/release/libscryneuro.so ./ || exit "$?"
    ) || return "$?"
    export PYLIB="$libdir"
    export LD_LIBRARY_PATH=".:$PYLIB:${LD_LIBRARY_PATH:-}"
    export SCRYNEURO_HOME="$home"
    echo "Build complete. PYLIB=$PYLIB"
    echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
    echo "SCRYNEURO_HOME=$SCRYNEURO_HOME"
    echo "To use in current shell, run: source build_linux.sh"
}

if _scryneuro_build_linux; then
    unset -f _scryneuro_build_linux
else
    unset -f _scryneuro_build_linux
    # Return from a sourced script; exit only when executed as its own process.
    return 1 2>/dev/null || exit 1
fi
