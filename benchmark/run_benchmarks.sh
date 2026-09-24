#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

if [ "$#" -gt 1 ]; then
    echo "Usage: $0 [positive-iterations]" >&2
    exit 2
fi
ITERATIONS="${1-10000}"
if [[ ! "$ITERATIONS" =~ ^[1-9][0-9]*$ ]]; then
    echo "Iterations must be a positive integer: $ITERATIONS" >&2
    exit 2
fi

case "$(uname -s)" in
    Darwin) LIB_EXT=dylib ;;
    Linux) LIB_EXT=so ;;
    *) echo "Unsupported platform" >&2; exit 1 ;;
esac

SCRYNEURO_HOME="${SCRYNEURO_HOME:-$PROJECT_ROOT}"
if [ ! -f "$SCRYNEURO_HOME/libscryneuro.$LIB_EXT" ]; then
    echo "Bridge library not found: $SCRYNEURO_HOME/libscryneuro.$LIB_EXT" >&2
    exit 1
fi
SCRYNEURO_HOME="$(cd "$SCRYNEURO_HOME" && pwd)"
export SCRYNEURO_HOME

PYLIB="$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')")"
if [ -z "$PYLIB" ]; then
    echo "Could not determine the active Python library directory" >&2
    exit 1
fi
if [ "$LIB_EXT" = dylib ]; then
    export DYLD_LIBRARY_PATH="$SCRYNEURO_HOME:$PYLIB${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
else
    export LD_LIBRARY_PATH="$SCRYNEURO_HOME:$PYLIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

printf 'Python: %s\n' "$(python3 --version)"
printf 'Library: %s\n' "$SCRYNEURO_HOME/libscryneuro.$LIB_EXT"
echo "Measurements are end-to-end call costs, not isolated FFI overhead."
echo "MNIST benchmarks are separate and are not included in this comparison."
python3 benchmark/bench_native.py "$ITERATIONS"
scryer-prolog benchmark/bench_ffi.pl -- "$ITERATIONS"
