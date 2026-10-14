"""
cesNMF: Covariate-Embedded Sparse Non-Negative Matrix Factorization
=======================================================================

Novel NMF algorithm for bulk tumor CAF deconvolution that embeds clinical
and genomic covariates directly within the factorization objective function.

Objective function:
    L = ||V - WH||²_F
      + λ₁ ||H - BX||²_F    (covariate regularization)
      + λ₂ ||W||²_F          (L2 on W)
      + λ₃ ||W||₁            (L1 sparsity)

Update rules (multiplicative, derived from first principles):
    H ← H ⊙ [WᵀV + λ₁(BX)₊] / [WᵀWH + λ₁H]
    W ← W ⊙ [VHᵀ]           / [WHHᵀ + λ₂W + λ₃]
    B ← HXᵀ(XXᵀ + εI)⁻¹    (closed-form OLS)

References:
    Lee & Seung (1999) Nature 401:788
    Lee & Seung (2001) NIPS 13:556
    Boutsidis & Gallopoulos (2008) Pattern Recognit 41:1350

Author:  Yasmine Baca
Version: 0.1.0
"""

import numpy as np
from scipy.linalg import svd as scipy_svd


# ── NNDSVD Initialization ─────────────────────────────────────────────────────

def nndsvd_init(V, k):
    """
    Non-Negative Double Singular Value Decomposition initialization.

    Computes a mathematically structured, non-negative starting point for
    NMF by applying SVD twice and selecting the positive parts of singular
    vectors. Substantially better than random initialization.

    Reference: Boutsidis & Gallopoulos (2008) Pattern Recognit 41:1350

    Parameters
    ----------
    V : np.ndarray, shape (n_genes, n_samples)
        Non-negative input matrix.
    k : int
        Number of programs (rank).

    Returns
    -------
    W : np.ndarray, shape (n_genes, k)
        Initialized gene loading matrix, W >= 0.
    H : np.ndarray, shape (k, n_samples)
        Initialized sample score matrix, H >= 0.
    """
    U, S, Vt = scipy_svd(V, full_matrices=False)
    U  = U[:, :k]
    S  = S[:k]
    Vt = Vt[:k, :]

    n, m = V.shape
    W = np.zeros((n, k), dtype=np.float32)
    H = np.zeros((k, m), dtype=np.float32)

    # First component: always use positive parts
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(Vt[0, :])

    # Remaining components: choose positive/negative part with larger signal
    for j in range(1, k):
        uj = U[:, j]
        vj = Vt[j, :]
        up = np.maximum(uj, 0);  un = np.maximum(-uj, 0)
        vp = np.maximum(vj, 0);  vn = np.maximum(-vj, 0)
        nup = np.linalg.norm(up); nun = np.linalg.norm(un)
        nvp = np.linalg.norm(vp); nvn = np.linalg.norm(vn)
        if nup * nvp >= nun * nvn:
            W[:, j] = np.sqrt(S[j]) * (nvp / (nup + 1e-10)) * up
            H[j, :] = np.sqrt(S[j]) * (nup / (nvp + 1e-10)) * vp
        else:
            W[:, j] = np.sqrt(S[j]) * (nvn / (nun + 1e-10)) * un
            H[j, :] = np.sqrt(S[j]) * (nun / (nvn + 1e-10)) * vn

    return np.maximum(W, 1e-10), np.maximum(H, 1e-10)


def biological_seed_init(V, k, signature_matrix, gene_names,
                          k_known=2, epsilon=1e-8):
    """
    Biologically-seeded NNDSVD initialization.

    Combines NNDSVD mathematical initialization with scRNA-seq-derived
    CAF signatures as biological priors for known program slots. Novel
    program slots remain fully data-driven (NNDSVD only).

    Parameters
    ----------
    V : np.ndarray, shape (n_shared_genes, n_samples)
        Preprocessed expression matrix restricted to shared genes.
    k : int
        Total number of programs (known + novel).
    signature_matrix : np.ndarray, shape (n_shared_genes, k_known)
        Signature matrix columns for known subtypes (e.g., iCAF, myCAF).
        Column order determines program slot assignment.
    gene_names : list of str
        Gene names corresponding to rows of V.
    k_known : int, default=2
        Number of known biological programs to seed (first k_known columns).
    epsilon : float, default=1e-8
        Ridge term for numerical stability.

    Returns
    -------
    W_init : np.ndarray, shape (n_shared_genes, k)
    H_init : np.ndarray, shape (k, n_samples)
    """
    # Step 1: NNDSVD gives structured non-negative starting point
    W_init, H_init = nndsvd_init(V, k)

    # Step 2: Normalize signature columns to match NNDSVD column scale
    W_scale = np.mean([np.linalg.norm(W_init[:, j]) for j in range(k)])
    sig_norm = np.zeros_like(signature_matrix, dtype=np.float32)
    for j in range(k_known):
        col = np.maximum(signature_matrix[:, j], 0)
        norm = np.linalg.norm(col)
        sig_norm[:, j] = col / (norm + 1e-10) * W_scale

    # Step 3: Replace first k_known columns with biological signatures
    W_init[:, :k_known] = sig_norm
    W_init = np.maximum(W_init, 1e-10)

    # Step 4: Re-solve H from seeded W via OLS for geometric consistency
    WtW = W_init.T @ W_init
    WtV = W_init.T @ V
    H_init = np.maximum(
        np.linalg.solve(WtW + epsilon * np.eye(k), WtV),
        1e-10
    )

    return W_init.astype(np.float32), H_init.astype(np.float32)


# ── Core cesNMF Algorithm ─────────────────────────────────────────────────────

def cesNMF(V, X, k,
           lambda1=0.1,
           lambda2=0.01,
           lambda3=0.01,
           max_iter=500,
           tol=1e-6,
           W_init=None,
           H_init=None,
           seed=42,
           verbose=True,
           program_names=None):
    """
    Covariate-Embedded Sparse Non-Negative Matrix Factorization.

    Discovers latent gene co-activation programs from bulk RNA-seq data
    with clinical and genomic covariates embedded directly in the
    factorization objective. Produces:
      - W : gene loading matrix (which genes define each program)
      - H : sample score matrix (how active each program is per patient)
      - B : covariate coefficient matrix (how covariates shape programs)

    Objective function:
        L = ||V - WH||²_F
          + λ₁ ||H - BX||²_F    (covariate regularization)
          + λ₂ ||W||²_F          (L2 ridge on W)
          + λ₃ ||W||₁            (L1 LASSO sparsity on W)

    Parameters
    ----------
    V : np.ndarray, shape (n_genes, n_samples)
        Preprocessed, non-negative expression matrix.
        Typically: log2(TPM+1), confounder-regressed, shifted >= 0.
    X : np.ndarray, shape (n_covariates, n_samples)
        Covariate matrix. Binary or continuous. Each row is one covariate
        (e.g., KRAS_mut, metastasis, age). Columns align with V columns.
    k : int
        Number of programs to discover (rank).
    lambda1 : float, default=0.1
        Covariate regularization strength. Higher = programs more
        constrained to be consistent with covariate structure X.
        lambda1=0 reduces to standard sparse NMF.
    lambda2 : float, default=0.01
        L2 ridge penalty on W. Prevents any single gene from dominating
        a program. Primarily ensures numerical stability.
    lambda3 : float, default=0.01
        L1 LASSO penalty on W. Forces most gene loadings to exactly zero,
        producing sparse, interpretable programs. Target: 40-70% sparsity.
    max_iter : int, default=500
        Maximum number of multiplicative update iterations.
    tol : float, default=1e-6
        Convergence tolerance. Stops when relative change in L < tol.
    W_init : np.ndarray or None, shape (n_genes, k)
        Initial W matrix. If None, uses random initialization.
        Recommended: use biological_seed_init() for best results.
    H_init : np.ndarray or None, shape (k, n_samples)
        Initial H matrix. If None, uses random initialization.
    seed : int, default=42
        Random seed for reproducibility (used only when W_init=None).
    verbose : bool, default=True
        Print progress every 50 iterations.
    program_names : list of str or None
        Names for each program. If None, auto-generates
        ['Program_1', ..., 'Program_k'].

    Returns
    -------
    result : dict with keys:
        'W'          : np.ndarray (n_genes, k) — gene loadings
        'H'          : np.ndarray (k, n_samples) — sample program scores
        'B'          : np.ndarray (k, n_covariates) — covariate coefficients
        'loss'       : list of float — total loss per iteration
        'recon_loss' : list of float — reconstruction loss per iteration
        'cov_loss'   : list of float — covariate alignment loss per iteration
        'final_loss' : float — converged total loss
        'sparsity_W' : float — fraction of W entries < 1e-3
        'n_iter'     : int — iterations to convergence
        'prog_names' : list of str — program names
        'k'          : int — number of programs
    """
    rng = np.random.RandomState(seed)
    n, m = V.shape
    p = X.shape[0]

    if V.min() < 0:
        raise ValueError(
            "V contains negative values. Apply V = V - V.min() before calling cesNMF."
        )
    if X.shape[1] != m:
        raise ValueError(
            f"X has {X.shape[1]} samples but V has {m} samples. Must match."
        )

    # Initialize W and H
    if W_init is not None and H_init is not None:
        if W_init.shape[0] != n:
            raise ValueError(
                f"W_init has {W_init.shape[0]} rows but V has {n} rows. "
                "Pass the shared-gene submatrix as V, not the full matrix."
            )
        W = W_init.copy().astype(np.float32)
        H = H_init.copy().astype(np.float32)
        if verbose:
            print("Initialization: biological seed (NNDSVD + signatures)")
    else:
        W = np.abs(rng.randn(n, k).astype(np.float32)) * 0.1
        H = np.abs(rng.randn(k, m).astype(np.float32)) * 0.1
        if verbose:
            print("Initialization: random")

    # Initialize B (covariate coefficient matrix, unconstrained)
    B = np.zeros((k, p), dtype=np.float32)

    # Pre-compute XX' inverse (constant across iterations)
    XXt = X @ X.T
    XXt_inv = np.linalg.inv(XXt + np.eye(p) * 1e-8)

    # Program names
    if program_names is None:
        program_names = [f"Program_{i+1}" for i in range(k)]

    # Loss history
    loss_hist  = []
    recon_hist = []
    cov_hist   = []

    if verbose:
        print(f"k={k}  λ₁={lambda1:.3f}  λ₂={lambda2:.4f}  λ₃={lambda3:.4f}")
        print(f"V: [{n} genes × {m} samples]  X: [{p} covariates × {m} samples]")
        print(f"{'Iter':>5}  {'Total':>12}  {'Recon':>12}  {'Cov':>10}  {'L1':>8}")

    # ── Main iteration loop ────────────────────────────────────────────────
    for it in range(max_iter):

        # Update H: multiplicative rule
        # H ← H ⊙ [WᵀV + λ₁(BX)₊] / [WᵀWH + λ₁H]
        BX    = B @ X
        H_num = W.T @ V + lambda1 * np.maximum(BX, 0)
        H_den = (W.T @ W) @ H + lambda1 * H
        H     = H * np.maximum(H_num, 1e-10) / np.maximum(H_den, 1e-10)
        H     = np.maximum(H, 1e-10)

        # Update W: multiplicative rule
        # W ← W ⊙ [VHᵀ] / [WHHᵀ + λ₂W + λ₃]
        W_num = V @ H.T
        W_den = W @ (H @ H.T) + lambda2 * W + lambda3
        W     = W * np.maximum(W_num, 1e-10) / np.maximum(W_den, 1e-10)
        W     = np.maximum(W, 1e-10)

        # Update B: closed-form OLS
        # B ← HXᵀ(XXᵀ + εI)⁻¹
        B = (H @ X.T) @ XXt_inv

        # Compute total loss
        WH    = W @ H
        r     = np.linalg.norm(V - WH, 'fro') ** 2
        cv    = lambda1 * np.linalg.norm(H - B @ X, 'fro') ** 2
        l2t   = lambda2 * np.linalg.norm(W, 'fro') ** 2
        l1t   = lambda3 * W.sum()
        loss  = r + cv + l2t + l1t

        loss_hist.append(loss)
        recon_hist.append(r)
        cov_hist.append(cv)

        # Check convergence
        if it > 0:
            rel_change = abs(loss_hist[-1] - loss_hist[-2]) / (loss_hist[-2] + 1e-10)
            if rel_change < tol:
                if verbose:
                    print(f"\nConverged at iteration {it + 1}  (rel_change={rel_change:.2e})")
                break

        if verbose and (it + 1) % 50 == 0:
            print(f"{it+1:>5}  {loss:>12.2f}  {r:>12.2f}  {cv:>10.2f}  {l1t:>8.2f}")

    sparsity = float((W < 1e-3).mean())

    if verbose:
        print(f"\nFinal loss    : {loss_hist[-1]:.4f}")
        print(f"W sparsity    : {sparsity * 100:.1f}%")
        print(f"Programs      : {' | '.join(program_names)}")

    return dict(
        W          = W,
        H          = H,
        B          = B,
        loss       = loss_hist,
        recon_loss = recon_hist,
        cov_loss   = cov_hist,
        final_loss = loss_hist[-1],
        sparsity_W = sparsity,
        n_iter     = it + 1,
        prog_names = program_names,
        k          = k,
    )
