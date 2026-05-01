# LAKER Benchmark Results

**Date:** 2026-04-30  
**Platform:** Darwin (macOS)  
**PyTorch:** 2.11.0  
**Dtype:** float32 (default)  
**Seed:** 42  

---

## Kernel Matvec

| n | chunk_size | mean (ms) | std (ms) |
|---|------------|-----------|----------|
| 1000 | 1024 | 0.816 | 0.121 |
| 2000 | 1024 | 1.917 | 0.306 |
| 5000 | 1024 | 13.283 | 0.805 |

## Approximation Matvec Comparison (n=2000)

| method | mean (ms) | std (ms) |
|--------|-----------|----------|
| exact | 2.594 | 0.270 |
| nystrom | 0.030 | 0.003 |
| rff | 0.060 | 0.018 |
| knn | 0.932 | 0.022 |
| ski | 20.059 | 0.804 |

## Preconditioner Build

| n | N_r | time (ms) |
|---|-----|-----------|
| 1000 | 100 | 6.77 |
| 2000 | 100 | 11.79 |
| 5000 | 100 | 57.23 |

## PCG Solve

| n | N_r | time (ms) | iters |
|---|-----|-----------|-------|
| 1000 | 100 | 256.36 | 305 |
| 2000 | 100 | 920.54 | 348 |
| 5000 | 100 | 14601.04 | 410 |

## Full Fit

| n | time (ms) | PCG iters |
|---|-----------|-----------|
| 200 | 34.69 | 212 |
| 500 | 79.26 | 278 |
| 1000 | 274.71 | 330 |
