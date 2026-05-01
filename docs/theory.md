# Mathematical Background

This page explains the mathematics behind LAKER in enough depth that you can reason about hyperparameters, diagnose convergence, and adapt the method to new problems.

---

## Problem Formulation

LAKER solves the *regularised attention kernel regression* problem

$$\min_{\alpha \in \mathbb{R}^n} \; \|\, G \alpha - y \,\|_2^2 \; + \; \lambda \, \alpha^{\!\top} G \alpha \qquad (1)$$

where

- $y \in \mathbb{R}^n$ are noisy measurements,
- $\lambda > 0$ is a Tikhonov regularisation parameter,
- $G \in \mathbb{R}^{n \times n}$ is the **exponential attention kernel**

$$G_{ij} = \exp\!\bigl( \langle e_i, e_j \rangle \bigr) \qquad (2)$$

induced by learned embeddings $E = [e_1, \dots, e_n]^{\!\top} \in \mathbb{R}^{n \times d_e}$.

The first term in (1) is a data-fitting loss; the second is a smoothness penalty that prefers $\alpha$ lying in the span of the kernel. Differentiating (1) yields the linear system

$$(\lambda I + G) \, \alpha = y \qquad (3)$$

which is what LAKER actually solves.

### Why an Exponential Attention Kernel?

In spectrum cartography the received signal strength (RSS) at location $x_i$ is a smooth function of spatial coordinates. A Gaussian (RBF) kernel could be used, but it requires a carefully tuned length-scale. The *attention kernel* (2) is parameterised by the embedding inner products $\langle e_i, e_j \rangle$. Because the embeddings are themselves learned (or at least data-dependent, via `PositionEmbedding`), the kernel adaptively reshapes its "similarity landscape" to the geometry of the measurements.

The exponential guarantees **strict positive definiteness** and gives rapid decay for dissimilar points, which is exactly the inductive bias needed for radio-map reconstruction. Furthermore, the kernel is *parameter-free* in the sense that no bandwidth hyperparameter is required — the embedding network learns the appropriate metric.

---

## The Ill-Conditioning Challenge

For $n \gtrsim 10\,000$ the matrix $\lambda I + G$ is dense, $O(n^2)$ in memory and $O(n^3)$ to factorise. Worse, the condition number

$$\kappa(\lambda I + G) = \frac{\lambda + \lambda_{\max}(G)}{\lambda + \lambda_{\min}(G)}$$

can easily exceed $10^8$, because the spectrum of an exponential kernel typically has a handful of very large eigenvalues (corresponding to global structure) and a long tail of tiny eigenvalues (high-frequency variations). A standard Conjugate Gradient (CG) solver therefore needs hundreds or thousands of iterations, each costing an $O(n^2)$ matvec.

LAKER attacks both problems simultaneously:

1. **Matrix-free matvecs** — the kernel is never formed explicitly; chunked dot-products keep memory at $O(n \cdot \text{chunk\_size})$.
2. **Learned data-dependent preconditioner** — a cheaply-computable $P \approx (\lambda I + G)^{-1/2}$ is learned via random probes so that the preconditioned system $P(\lambda I + G)$ has a compressed spectrum and CG converges in *tens* of iterations, essentially independently of $n$.

---

## The CCCP Preconditioner (Algorithm 1)

The paper proposes learning a preconditioner by treating the kernel matrix as a covariance and estimating its inverse square-root $\Sigma^{-1/2}$ from a small number of random probe directions.

### Random Probes

Draw $N_r$ random directions $R \in \mathbb{R}^{n \times N_r}$ and apply the operator:

$$U = (\lambda I + G) \, R$$

Each column $u_k$ is a "probe response". The responses are normalised to unit length, giving $\bar{U}$, and an economy QR factorisation yields an orthonormal basis $Q \in \mathbb{R}^{n \times N_r}$:

$$\bar{U} = Q \, R_{\mathrm{qr}}$$

The key insight is that the *span* of the probe responses captures the dominant eigenspaces of the operator, because random vectors have non-negligible overlap with *every* eigenvector. By working entirely in the $Q$-basis we reduce the effective dimension from $n$ to $N_r$.

### Maximum-Likelihood Objective

The paper derives a regularised log-likelihood for $\Sigma$ given the probe responses. In the $Q$-basis the covariance has a *low-rank-plus-isotropic* structure:

$$\Sigma = a I + Q C Q^{\!\top} \qquad (4)$$

where $a > 0$ is the isotropic coefficient and $C \in \mathbb{R}^{N_r \times N_r}$ is the low-rank correction. The objective is non-convex, so the authors apply the **Convex-Concave Procedure (CCCP)**.

### CCCP Iteration

At each CCCP step we form the surrogate

$$F_{\gamma} = \frac{1}{1 + \gamma/n} \Bigl( \sum_{k=1}^{N_r} w_k \, \bar{u}_k \bar{u}_k^{\!\top} + \gamma I \Bigr)$$

where $\gamma$ is a regularisation parameter and the weights $w_k$ depend on the *previous* iterate's estimate of $\Sigma$. The CCCP update then solves a tractable quadratic program whose closed-form solution is the new $\Sigma$.

### Shrinkage Regularisation

Because $N_r \ll n$ the probe sample is *undersampled*. A pure maximum-likelihood estimate would overfit to the probe directions and give an ill-conditioned preconditioner. LAKER therefore applies **isotropic shrinkage**:

$$\Sigma_{\rho} = (1 - \rho) \, F_{\gamma} + \rho \, I$$

where the shrinkage parameter $\rho$ is *adaptive*: it increases automatically when the undersampling ratio $N_r/n$ is small or when $\gamma$ is large, providing a smooth interpolation between the learned low-rank structure and a safe identity preconditioner.

The adaptive rule is

$$\rho = \rho_0 + (1 - \rho_0) \bigl(1 - \tfrac{N_r}{n}\bigr) \min(1, 10\gamma) \qquad \text{clamped to } [0, 0.5]$$

where $\rho_0$ is the base shrinkage (`base_rho`, default $0.05$). When $N_r \geq n$ we simply return $\rho_0$.

### Trace Normalisation

Shrinkage biases the eigenvalues toward 1, but it also changes the mean eigenvalue. To keep the preconditioner neutral (i.e. not rescale the overall magnitude of the linear system), LAKER applies **trace normalisation** after each CCCP step:

$$\Sigma \leftarrow \frac{n}{\operatorname{tr}(\Sigma)} \, \Sigma$$

This ensures $\operatorname{tr}(\Sigma) = n$, so the mean eigenvalue remains 1 and the preconditioner does not artificially inflate or deflate the residual norm monitored by PCG.

---

## The Factored $O(N_r^3)$ Representation

A naive implementation of CCCP would require $O(n^3)$ work per iteration (full eigendecompositions of an $n \times n$ matrix). LAKER exploits the fixed random-probe structure to reduce this to $O(N_r^3)$, independent of $n$.

Because $\Sigma$ always has the form (4), every matrix operation inside CCCP can be rewritten in the $N_r$-dimensional $Q$-basis:

- Inverting $\Sigma$ becomes inverting the $N_r \times N_r$ matrix $M = a I + C$.
- Eigenvalue computations are performed on $M$, not on the full $n \times n$ matrix.
- The preconditioner apply $P = \Sigma^{-1/2}$ decomposes as

$$P x = a^{-1/2} x + Q \, V \, \bigl(\lambda_i^{-1/2} - a^{-1/2}\bigr) \, V^{\!\top} \, Q^{\!\top} x$$

where $M = V \operatorname{diag}(\lambda_i) V^{\!\top}$. This is an $O(n N_r)$ operation, again independent of the condition number.

In practice $N_r \approx 2\sqrt{n}$ (the adaptive heuristic used when `num_probes=None`), so for $n = 100\,000$ we have $N_r \approx 630$ and the cubic term is negligible.

---

## Preconditioned Conjugate Gradient

Once $P$ is built, LAKER solves (3) with standard PCG. The preconditioned system is

$$P(\lambda I + G) \, \alpha = P y$$

Because $P \approx (\lambda I + G)^{-1/2}$, the spectrum of the preconditioned operator is clustered around 1. The paper reports condition number reductions of **up to three orders of magnitude**, which translates directly into PCG convergence in **20–40 iterations** instead of several thousand, even for $n = 10^5$.

LAKER supports both 1-D (single right-hand side) and 2-D (batch of RHS) PCG, and includes optional residual replacement every 50 iterations to combat floating-point drift. The `restart_freq` parameter controls this; it is disabled by default because explicit residual recomputation can hurt `float32` stability.

---

## Complexity Summary

Let $n$ be the number of measurements and $N_r$ the number of random probes.

| Step | Time | Memory |
|------|------|--------|
| Embedding (forward) | $O(n \, d_e^2)$ | $O(n \, d_e)$ |
| Kernel matvec (chunked) | $O(n^2 \, d_e)$ | $O(n \cdot \text{chunk\_size})$ |
| CCCP preconditioner build (per iteration) | $O(N_r^3 + n \, N_r)$ | $O(n \, N_r)$ |
| PCG solve | $O(\text{iters} \cdot n^2 \, d_e)$ | $O(n)$ |

Because $\text{iters}$ is $O(1)$ thanks to the preconditioner, the dominant cost is the matvecs inside PCG, i.e. $O(n^2 \, d_e)$ total for the solve. The preconditioner learning itself is a small additive overhead.

---

## Kernel Approximations

For very large $n$ the exact $O(n^2)$ matvec can become a bottleneck. LAKER provides four approximation strategies, all implementing the same `matvec` / `diagonal` / `kernel_eval` / `to_dense` interface.

### Nyström Low-Rank

Approximates $G \approx G_{nm} G_{mm}^{-1} G_{nm}^{\!\top}$ using $m$ landmark points selected via k-means++ greedy sampling. Matvec cost drops from $O(n^2)$ to $O(n \, m)$. The number of landmarks defaults to $\max(50, \lfloor\sqrt{n}\rfloor)$.

### Random Fourier Features (RFF)

Approximates the kernel by a finite-dimensional feature map $\phi(x) \in \mathbb{R}^{2r}$ so that $G \approx \Phi \Phi^{\top}$. The feature dimension $r$ defaults to $\max(100, 2\lfloor\sqrt{n}\rfloor)$. RFF is particularly fast because matvecs reduce to two matrix–vector products with $\Phi$.

### Sparse k-NN

Retains only the $k$ largest kernel values per row (nearest neighbours in Euclidean embedding space). The matrix is stored as a sparse COO tensor, reducing storage from $O(n^2)$ to $O(n \, k)$. Symmetrisation and strict diagonal dominance are enforced for guaranteed positive definiteness.

### Structured Kernel Interpolation (SKI)

Builds a regular product grid in the embedding space and uses multilinear interpolation weights $W$ so that $G \approx W G_{\text{grid}} W^{\!\top}$. The grid kernel $G_{\text{grid}}$ is materialised explicitly (since the grid is small), and matvecs cost $O(n \, g)$ where $g$ is the grid size. SKI is practical when the embedding dimension $d_e \leq 10$.

---

## Practical Take-aways

- **Start with the defaults.** `gamma=0.1`, `num_probes=None`, and `base_rho=0.05` are the values reported in the paper and work well for $n \in [10^3, 10^5]$.
- **If PCG takes > 100 iterations**, increase `num_probes` or decrease `gamma` (less regularisation lets the preconditioner learn more structure).
- **If the preconditioner build is slow**, you can decrease `num_probes` manually at the cost of slightly more PCG iterations.
- **For $n < 5\,000$** you can set `chunk_size=None` to form the full kernel explicitly; for larger problems the auto-selected chunk size prevents out-of-memory errors.
- **Use `float64`** when the condition number is very high ($\kappa \gtrsim 10^{10}$); otherwise `float32` is usually fine and twice as fast on modern GPUs.
- **Choose approximations wisely.** Nyström and RFF are good general-purpose speedups. Sparse k-NN shines when the data has local structure. SKI is best for low-dimensional embeddings ($d_e \leq 10$) and moderate grid sizes.
