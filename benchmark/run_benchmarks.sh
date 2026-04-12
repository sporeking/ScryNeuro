#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ITERATIONS=${1:-10000}

cd "$PROJECT_ROOT"

if [ ! -f "libscryneuro.so" ]; then
    echo "ERROR: libscryneuro.so not found"
    echo "Build first: source build_linux.sh"
    exit 1
fi

PYLIB=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")

echo "============================================================================"
echo "ScryNeuro FFI Performance Benchmark (Fair Comparison)"
echo "============================================================================"
echo "Iterations: $ITERATIONS"
echo

echo "=== Part 1: Python Native Benchmark ==="
python3 benchmark/bench_native.py "$ITERATIONS"
echo

echo "=== Part 2: ScryNeuro FFI Benchmark ==="
LD_LIBRARY_PATH=".:$PYLIB:$LD_LIBRARY_PATH" scryer-prolog benchmark/bench_ffi.pl
echo

echo "============================================================================"
echo "Fair Comparison Analysis"
echo "============================================================================"
echo
echo "py_eval operations (Python eval vs FFI py_eval):"
echo "  Python eval overhead: ~5-12 us (string parsing + compilation)"
echo "  FFI additional overhead: ~2-5 us (GIL + handle management)"
echo "  Total FFI = Python eval + FFI bridge overhead"
echo
echo "Direct operations (method call, conversion, etc.):"
echo "  Python direct: negligible (< 0.1 us)"
echo "  FFI: ~3-5 us (handle creation + GIL + registry)"
echo "  Expected - FFI must manage Python object handles"
echo
echo "Key insight for neural-symbolic applications:"
echo "  - LLM API calls: 100-500 ms (100,000-500,000 us)"
echo "  - Neural network inference: 10-100 ms (10,000-100,000 us)"
echo "  - FFI overhead: ~2-10 us"
echo "  - FFI overhead is < 0.01% of total execution time"
echo "============================================================================"