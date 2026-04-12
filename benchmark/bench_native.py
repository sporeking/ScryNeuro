import time
import json
import sys

DEFAULT_ITERATIONS = 10000


def bench(name, func, n):
    start = time.perf_counter()
    for _ in range(n):
        func()
    end = time.perf_counter()
    elapsed_us = (end - start) * 1_000_000
    avg_us = elapsed_us / n
    print(f"{name}: total={elapsed_us:.1f} us, avg={avg_us:.4f} us/op")


def run_benchmarks(n=DEFAULT_ITERATIONS):
    print(f"=== Python Native Benchmark (N={n}) ===")
    print()

    print("--- py_eval equivalent (using eval) ---")
    print("These match FFI py_eval() which calls Python eval()")
    bench("int_add", lambda: eval("1 + 1"), n)
    bench("float_mul", lambda: eval("1.5 * 2.5"), n)
    bench("str_concat", lambda: eval("'hello' + 'world'"), n)
    bench("list_create", lambda: eval("[1, 2, 3, 4, 5]"), n)
    bench("builtin_call", lambda: eval("len([1, 2, 3, 4, 5])"), n)

    print("\n--- py_call equivalent (direct method call) ---")
    print("These match FFI py_call() which calls method directly")
    s = "hello world"
    bench("method_call", lambda: s.upper(), n)

    print("\n--- py_from/py_to equivalent (direct conversion) ---")
    print("These match FFI type conversion operations")
    bench("convert_int", lambda: int(42), n)
    bench("convert_float", lambda: float(3.14159), n)
    bench("convert_str", lambda: str("benchmark test string"), n)

    print("\n--- py_to_json/py_from_json equivalent ---")
    data = {"name": "test", "value": 42}
    bench("json_roundtrip", lambda: json.loads(json.dumps(data)), n)

    print("\n--- py_import equivalent ---")
    bench("import_attr", lambda: getattr(__import__("math"), "pi"), n)

    print("\n--- py_list_new/append/get equivalent ---")
    bench("list_ops", lambda: [1, 2][0], n)

    print("\n--- Baseline: Direct operations (no eval) ---")
    print("Python raw speed without eval overhead:")
    bench("int_add_direct", lambda: 1 + 1, n)
    bench("float_mul_direct", lambda: 1.5 * 2.5, n)
    bench("str_concat_direct", lambda: "hello" + "world", n)
    bench("list_create_direct", lambda: [1, 2, 3, 4, 5], n)

    print("\n=== Benchmark Complete ===")
    print()
    print("Comparison guide:")
    print("  - 'int_add' (eval) vs FFI 'int_add' = fair comparison")
    print("  - 'int_add_direct' vs FFI 'int_add' = unfair (eval vs direct)")
    print("  - True FFI overhead = FFI result - eval result")


def main():
    n = DEFAULT_ITERATIONS
    if len(sys.argv) > 1:
        try:
            n = int(sys.argv[1])
        except ValueError:
            print(f"Invalid iteration count: {sys.argv[1]}, using default {n}")
    run_benchmarks(n)


if __name__ == "__main__":
    main()
