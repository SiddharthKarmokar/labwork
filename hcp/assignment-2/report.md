# Parallel Sorting Check: Performance Analysis Report
By Siddharth Karmokar, 123cs0061

## 1. Objective

The goal of this experiment was to design and compare different parallel programming approaches for checking if a large list of integers is sorted in ascending order. We evaluated the performance scaling of OpenMP (Implicit, Explicit, Tasks, SIMD) and Pthreads across varying list sizes and thread counts.

## 2. Methodology

Five distinct parallel approaches were implemented in C:

1. **OMP_Loop (Implicit):** Uses `#pragma omp parallel for` with static scheduling.
2. **OMP_Explicit:** Manually decomposes the array index domain based on thread ID.
3. **OMP_Tasks:** Uses a recursive divide-and-conquer strategy with OpenMP tasks.
4. **OMP_SIMD:** Uses Single Instruction Multiple Data (AVX/SSE) vectorization on a single core.
5. **Pthreads:** Uses the OS-level POSIX threads library with manual struct passing.

**Hardware & Environment:**

* **Platform:** WSL (Ubuntu) on Windows
* **Compiler:** GCC with `-O2` optimization
* **Thread Counts Tested:** 2, 4, 8 (plus single-thread SIMD)
* **Data Sizes:** 10k, 50k, 500k, 2M integers

## 3. Observations from Results

### A. Small Data Sizes (10,000 - 50,000)

* **Trend:** As thread count increased (2  8), execution time **increased** significantly for `OMP_Explicit` and `OMP_Tasks`.
* **Reason:** The workload is too small. The overhead of thread creation and context switching dominates the execution time. The actual comparison logic takes microseconds, while setting up OpenMP tasks takes milliseconds.
* **Winner:** `OMP_SIMD` (Purple) was the most efficient as it avoids threading overhead entirely.

### B. Large Data Sizes (2,000,000)

* **OMP_Explicit (Orange):** Showed the worst scaling. The manual overhead of calculating chunk boundaries and atomic updates proved inefficient compared to compiler-optimized loops.
* **OMP_Tasks (Green):** Also scaled poorly. The recursive overhead of creating thousands of small tasks is unnecessary for a linear array traversal.
* **Pthreads (Red) vs. OMP_Loop (Blue):** These remained relatively flat.
* **Pthreads** showed surprising stability, likely because the OS thread pool is efficient, but it did not provide a significant speedup over the baseline.
* **OMP_Loop** performed consistently but did not achieve perfect linear speedup.



## 4. Key Analysis: Why didn't it get faster?

The graphs show a counter-intuitive result where parallelization often made things slower. This is due to **Memory Bandwidth Saturation**.

* **Memory Bound:** Checking `arr[i] > arr[i+1]` is a memory-bound operation, not compute-bound. The CPU can compare numbers much faster than the RAM can deliver them.
* Adding more threads just leads to more cores fighting over the same memory bus bandwidth.
* **False Sharing:** If threads update the shared `is_sorted` flag frequently (even with atomic writes), it causes cache invalidation storms, slowing down all threads.

## 5. Conclusion

1. **SIMD is Superior for Simple Checks:** For linear traversal operations like `is_sorted`, vectorization (`OMP_SIMD`) provides the best performance boost without the overhead of multi-threading.
2. **Overhead Matters:** For input sizes under 1 million, the cost of spawning threads outweighs the benefit of parallel execution.
3. **Implicit is Better than Explicit:** The compiler-optimized `OMP_Loop` consistently outperformed the manual `OMP_Explicit` and `OMP_Tasks` approaches.
4. **Recommendation:** For this specific problem, a hybrid approach using **Sequential SIMD** or a **Coarse-Grained Parallel Loop** (only for very large arrays, e.g., N > 10^7) is ideal.

## 6. Future Improvements

* Increase array size to 100M+ to potentially see the benefits of multi-threading.
* Implement "Early Exit" strategies more aggressively in Pthreads to stop all threads immediately once an unsorted element is found.