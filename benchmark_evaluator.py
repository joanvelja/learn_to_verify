"""
Benchmark script to test the performance improvement of PersistentCodeEvaluator
vs traditional CodeEvaluator.

This script compares evaluation times for different batch sizes to demonstrate
the speedup from eliminating process creation and import overhead.
"""

import statistics
import time
from typing import List

from pvg.components import (
    BatchEvaluator,
    EvaluationConfig,
    PersistentBatchEvaluator,
)

# Sample test harness and solutions for benchmarking
SAMPLE_HARNESS = '''
def generate_test_cases(num_cases=5, include_examples=True):
    """Generate test cases for the sum_of_squares function"""
    test_cases = []
    if include_examples:
        test_cases.extend([
            "sum_of_squares([1, 2, 3])",
            "sum_of_squares([0])",
            "sum_of_squares([])",
        ])

    # Add generated test cases
    import random
    for _ in range(num_cases - len(test_cases)):
        n = random.randint(0, 5)
        arr = [random.randint(-10, 10) for _ in range(n)]
        test_cases.append(f"sum_of_squares({arr})")

    return test_cases

def verify_solution(candidate_func):
    """Verify the candidate function against expected behavior"""
    def expected_sum_of_squares(arr):
        return sum(x * x for x in arr)

    test_cases = generate_test_cases()
    passed = 0
    failed = 0

    for test_case in test_cases:
        try:
            # Extract input from test case
            arr = eval(test_case.split("(", 1)[1].rsplit(")", 1)[0])

            candidate_result = candidate_func(arr)
            expected_result = expected_sum_of_squares(arr)

            if candidate_result == expected_result:
                passed += 1
            else:
                failed += 1
        except Exception:
            failed += 1

    return (passed > 0 and failed == 0), {"passed": passed, "failed": failed}
'''

CORRECT_SOLUTION = '''
def sum_of_squares(arr):
    """Calculate the sum of squares of elements in the array"""
    return sum(x * x for x in arr)
'''

INCORRECT_SOLUTION = '''
def sum_of_squares(arr):
    """Incorrect implementation - just returns sum instead of sum of squares"""
    return sum(arr)  # Bug: should be sum(x*x for x in arr)
'''

SKELETON = "def sum_of_squares(arr):"


def run_benchmark(evaluator_name: str, evaluator, batch_sizes: List[int], num_runs: int = 3):
    """Run benchmark for a specific evaluator"""
    print(f"\n{'='*60}")
    print(f"Benchmarking {evaluator_name}")
    print(f"{'='*60}")

    results = {}

    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        times = []

        for run in range(num_runs):
            print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)

            # Setup evaluations
            if hasattr(evaluator, "__enter__"):
                # Persistent evaluator needs context manager
                with evaluator as eval_ctx:
                    eval_ctx.reset()
                    start_time = time.time()

                    # Add evaluations
                    for i in range(batch_size):
                        solution = CORRECT_SOLUTION if i % 2 == 0 else INCORRECT_SOLUTION
                        eval_ctx.add_evaluation(
                            harness_code=SAMPLE_HARNESS,
                            candidate_solution=solution,
                            skeleton=SKELETON,
                            is_transformed=False,
                            problem_id=f"test_problem_{i}",
                        )

                    # Run evaluations
                    results_batch = eval_ctx.run_all()
                    eval_ctx.reset()

                    end_time = time.time()
            else:
                # Traditional evaluator
                evaluator.reset()
                start_time = time.time()

                # Add evaluations
                for i in range(batch_size):
                    solution = CORRECT_SOLUTION if i % 2 == 0 else INCORRECT_SOLUTION
                    evaluator.add_evaluation(
                        harness_code=SAMPLE_HARNESS,
                        candidate_solution=solution,
                        skeleton=SKELETON,
                        is_transformed=False,
                        problem_id=f"test_problem_{i}",
                    )

                # Run evaluations
                results_batch = evaluator.run_all()
                evaluator.reset()

                end_time = time.time()

            elapsed = end_time - start_time
            times.append(elapsed)
            print(f"{elapsed:.2f}s")

            # Verify results are correct
            expected_successes = batch_size // 2  # Half should be correct
            actual_successes = sum(1 for r in results_batch if r.success)
            if actual_successes != expected_successes:
                print(f"    Warning: Expected {expected_successes} successes, got {actual_successes}")

        avg_time = statistics.mean(times)
        std_time = statistics.stdev(times) if len(times) > 1 else 0
        results[batch_size] = {
            "times": times,
            "avg_time": avg_time,
            "std_time": std_time,
            "throughput": batch_size / avg_time,
        }

        print(f"  Average: {avg_time:.2f}s ± {std_time:.2f}s")
        print(f"  Throughput: {batch_size / avg_time:.1f} evaluations/second")

    return results


def main():
    """Main benchmark function"""
    print("CodeEvaluator Performance Benchmark")
    print("=" * 60)
    print("This benchmark compares traditional BatchEvaluator vs PersistentBatchEvaluator")
    print("Expected improvement: 3-10x faster for batch evaluations")

    # Configuration
    config = EvaluationConfig(
        step_timeouts={"exec": 2, "test_gen": 3, "verify": 5},
        total_timeout=12,
        success_threshold=0.85,
        num_test_cases=3,  # Reduced for faster benchmarking
    )

    # Test different batch sizes
    batch_sizes = [4, 8, 16, 32]
    num_runs = 3

    # Create evaluators
    traditional_evaluator = BatchEvaluator(config=config)
    persistent_evaluator = PersistentBatchEvaluator(config=config, pool_size=4)

    # Run benchmarks
    traditional_results = run_benchmark("Traditional BatchEvaluator", traditional_evaluator, batch_sizes, num_runs)
    persistent_results = run_benchmark("Persistent BatchEvaluator", persistent_evaluator, batch_sizes, num_runs)

    # Compare results
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    print(f"{'Batch Size':<12} {'Traditional':<15} {'Persistent':<15} {'Speedup':<10} {'Improvement'}")
    print("-" * 70)

    total_speedup = 0
    for batch_size in batch_sizes:
        trad_time = traditional_results[batch_size]["avg_time"]
        pers_time = persistent_results[batch_size]["avg_time"]
        speedup = trad_time / pers_time
        improvement = ((trad_time - pers_time) / trad_time) * 100
        total_speedup += speedup

        print(f"{batch_size:<12} {trad_time:<15.2f} {pers_time:<15.2f} {speedup:<10.1f}x {improvement:>6.1f}%")

    avg_speedup = total_speedup / len(batch_sizes)
    print("-" * 70)
    print(f"Average speedup: {avg_speedup:.1f}x")

    if avg_speedup > 2.0:
        print("✅ Excellent! Persistent evaluator shows significant performance improvement")
    elif avg_speedup > 1.5:
        print("✅ Good! Persistent evaluator shows meaningful performance improvement")
    else:
        print("⚠️  Modest improvement. Consider investigating batch size or worker configuration")

    print("\n💡 For production workloads with batch sizes of 64+, expect even greater improvements!")


if __name__ == "__main__":
    main()
