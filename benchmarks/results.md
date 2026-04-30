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
| 1000 | 1024 | 1.910 | 0.290 |
| 2000 | 1024 | 5.368 | 0.531 |
| 5000 | 1024 | 33.476 | 4.890 |

## Approximation Matvec Comparison (n=2000)

| method | mean (ms) | std (ms) |
|--------|-----------|----------|
| exact | 6.815 | 0.376 |
| nystrom | 0.042 | 0.016 |
| rff | 0.085 | 0.055 |
| knn | 4.305 | 1.008 |
| ski | 60.409 | 9.848 |

## Preconditioner Build

| n | N_r | time (ms) |
|---|-----|-----------|
| 1000 | 100 | 10.44 |
| 2000 | 100 | 18.78 |
| 5000 | 100 | 121.68 |

## PCG Solve

| n | N_r | time (ms) | iters |
|---|-----|-----------|-------|
| 1000 | 100 | 441.64 | 191 |
| 2000 | 100 | 1248.40 | 216 |
| 5000 | 100 | 17590.87 | 260 |

## Full Fit

| n | time (ms) | PCG iters |
|---|-----------|-----------|
| 200 | 61.29 | 130 |
| 500 | 137.13 | 175 |
| 1000 | 503.81 | 207 |
