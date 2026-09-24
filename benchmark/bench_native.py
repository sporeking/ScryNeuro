import argparse
import json
import statistics
import time

WARMUP = 100
ROUNDS = 5
DEFAULT_ITERATIONS = 10000


def bench(name, operation, iterations):
    for _ in range(WARMUP):
        operation()

    samples = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(iterations):
            operation()
        samples.append((time.perf_counter() - start) * 1_000_000 / iterations)

    print(
        f"{name}: median={statistics.median(samples):.2f} us/op, "
        f"min={min(samples):.2f} us/op, max={max(samples):.2f} us/op"
    )


def run_benchmarks(iterations):
    print(f"=== Python native (N={iterations}, warmup={WARMUP}, rounds={ROUNDS}) ===")
    bench("int_add", lambda: eval("1 + 1"), iterations)
    bench("float_mul", lambda: eval("1.5 * 2.5"), iterations)
    bench("str_concat", lambda: eval("'hello' + 'world'"), iterations)
    bench("list_create", lambda: eval("[1, 2, 3, 4, 5]"), iterations)
    bench("builtin_call", lambda: eval("len([1, 2, 3, 4, 5])"), iterations)

    text = "hello world"
    bench("method_call", lambda: text.upper(), iterations)
    bench("import_attr", lambda: getattr(__import__("math"), "pi"), iterations)

    data = {"name": "test", "value": 42}
    bench("json_roundtrip", lambda: json.loads(json.dumps(data)), iterations)


def main():
    parser = argparse.ArgumentParser(description="Measure native Python operations")
    parser.add_argument("iterations", nargs="?", type=int, default=DEFAULT_ITERATIONS)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be a positive integer")
    run_benchmarks(args.iterations)


if __name__ == "__main__":
    main()
