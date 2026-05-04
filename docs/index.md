# LAKER Documentation

**LAKER** (Learning-based Attention Kernel Regression) is a production-ready Python package for large-scale spectrum cartography and radio map reconstruction.

> **Note:** This repository is an independent implementation of the LAKER algorithm. I am not an author of the original paper.

It solves regularised attention kernel regression problems of the form

$$\min_\alpha \|G \alpha - y\|_2^2 + \lambda \, \alpha^\top G \alpha$$

where $G = \exp(E E^\top)$ is an exponential attention kernel induced by learned embeddings $E \in \mathbb{R}^{n \times d_e}$. The key innovation is a **learned data-dependent preconditioner** obtained via a shrinkage-regularised Convex-Concave Procedure (CCCP), which reduces condition numbers by up to three orders of magnitude and enables near size-independent Preconditioned Conjugate Gradient (PCG) convergence.

Based on [*Accelerating Regularized Attention Kernel Regression for Spectrum Cartography*](https://arxiv.org/html/2604.25138v1) (Tao \& Tan, 2026).

---

## Contents

- [Quick Start](quickstart.md) — Installation, basic usage, CLI, and new features
- [Mathematical Background](theory.md) — Problem formulation, CCCP preconditioner, kernel approximations
- [API Reference](api.md) — Complete reference for all public classes and functions
- [Examples](examples.md) — Reproducible worked examples: spectral, bilevel, uncertainty, and more
- [Design Patterns](patterns.md) — Conventions for executors, naming, and logging

---

## Design Philosophy

LAKER follows a class-based design with no semi-private naming conventions:

- **Exact and approximate kernels** live in a single module (`laker.kernels`): exact attention, Nyström, Random Fourier Features (RFF), sparse k-NN, and Structured Kernel Interpolation (SKI).
- **All utilities** are consolidated in `laker.utils`: numerical stability helpers (`trace_normalize`, `eigh_stable`, `adaptive_shrinkage_rho`) plus a lightweight Bayesian Optimisation surrogate (`GPSurrogate`).
- **Logging replaces prints** — all diagnostic output uses Python's standard `logging` module.
- **Thin functional wrappers** are provided for convenience, but every critical path is implemented as a class with a clean public API.
