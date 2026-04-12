"""
MNIST CNN Training Benchmark - Python Native
=============================================
Direct Python execution of MNIST CNN training.
Uses the same mnist_cnn_module as the FFI version for fair comparison.

Run: python benchmark/bench_mnist_cnn_native.py
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))

from mnist_cnn_module import MnistPipeline


def benchmark_mnist_cnn_native(epochs=3):
    print(f"=== MNIST CNN Training Benchmark (Python Native) ===")
    print(f"Epochs: {epochs}")
    print()

    pipeline = MnistPipeline()

    print("[Step 1] Creating pipeline...")
    start_create = time.perf_counter()
    end_create = time.perf_counter()
    print(f"  Pipeline creation: {end_create - start_create:.3f}s")

    print("[Step 2] Loading MNIST data...")
    start_load = time.perf_counter()
    info = pipeline.load_data()
    end_load = time.perf_counter()
    load_time = end_load - start_load
    print(f"  Train samples: {info['train_size']}")
    print(f"  Test samples: {info['test_size']}")
    print(f"  Data loading time: {load_time:.3f}s (I/O, excluded from FFI comparison)")

    print("[Step 3] Setting up model...")
    start_setup = time.perf_counter()
    device = pipeline.setup()
    end_setup = time.perf_counter()
    setup_time = end_setup - start_setup
    print(f"  Device: {device}")
    print(f"  Setup time: {setup_time:.3f}s")

    print(f"\n[Step 4] Training ({epochs} epochs)...")
    train_times = []
    for epoch in range(1, epochs + 1):
        start_epoch = time.perf_counter()
        stats = pipeline.train_one_epoch()
        end_epoch = time.perf_counter()
        epoch_time = end_epoch - start_epoch
        train_times.append(epoch_time)
        print(
            f"  Epoch {epoch}: loss={stats['loss']:.4f}, acc={stats['accuracy']:.2f}%, time={epoch_time:.3f}s"
        )

    total_train_time = sum(train_times)
    avg_epoch_time = total_train_time / epochs
    print(f"  Total training time: {total_train_time:.3f}s")
    print(f"  Average epoch time: {avg_epoch_time:.3f}s")

    print("\n[Step 5] Evaluating...")
    start_eval = time.perf_counter()
    test_acc = pipeline.evaluate()
    end_eval = time.perf_counter()
    eval_time = end_eval - start_eval
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Evaluation time: {eval_time:.3f}s")

    print("\n[Step 6] Inference benchmark (100 predictions)...")
    start_infer = time.perf_counter()
    for i in range(100):
        pipeline.predict_digit(i)
    end_infer = time.perf_counter()
    infer_time = end_infer - start_infer
    infer_per_sample = infer_time / 100
    print(f"  Total inference time (100 samples): {infer_time:.3f}s")
    print(f"  Per-sample inference: {infer_per_sample * 1000:.3f}ms")

    print("\n=== Summary ===")
    print(f"Setup time:       {setup_time:.3f}s")
    print(f"Training time:    {total_train_time:.3f}s ({epochs} epochs)")
    print(f"Evaluation time:  {eval_time:.3f}s")
    print(f"Inference time:   {infer_per_sample * 1000:.3f}ms/sample (x100)")

    return {
        "setup": setup_time,
        "train_total": total_train_time,
        "train_avg_epoch": avg_epoch_time,
        "eval": eval_time,
        "infer_per_sample": infer_per_sample,
    }


if __name__ == "__main__":
    epochs = 3
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            print(f"Invalid epochs: {sys.argv[1]}, using default {epochs}")

    benchmark_mnist_cnn_native(epochs)
