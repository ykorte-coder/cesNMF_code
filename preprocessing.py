"""
cesNMF preprocessing utilities
================================
Functions for preparing bulk RNA-seq data and covariates
for input to cesNMF.
"""

import numpy as np
import pandas as pd


def log_transform(tpm_matrix):
    """
    Apply log2(TPM + 1) transformation.

    Compresses right-skewed TPM distribution toward approximate
    Gaussian, justifying the Frobenius norm loss in cesNMF.

    Parameters
    ----------
    tpm_matrix : np.ndarray or pd.DataFrame
        Raw TPM expression values, shape (n_genes, n_samples). Must be >= 0.

    Returns
    -------
    np.ndarray, shape (n_genes, n_samples)
        Log2-transformed matrix.
    """
    if isinstance(tpm_matrix, pd.DataFrame):
        values = tpm_matrix.values.astype(np.float32)
    else:
        values = np.array(tpm_matrix, dtype=np.float32)
    if values.min() < 0:
        raise ValueError("TPM matrix contains negative values.")
    return np.log2(values + 1)


def regress_confounders(log_matrix, confounder_matrix):
    """
    Regress demographic or technical confounders from expression data.

    Uses ordinary least squares (OLS) to remove confounder effects from
    each gene's expression profile while retaining the biological mean.
    Equivalent to limma's removeBatchEffect() in R.

    Parameters
    ----------
    log_matrix : np.ndarray, shape (n_genes, n_samples)
        Log-transformed expression matrix.
    confounder_matrix : np.ndarray, shape (n_confounders, n_samples)
        Confounder variables to regress out (e.g., age, sex).
        Each row is one confounder; columns align with log_matrix.

    Returns
    -------
    adjusted : np.ndarray, shape (n_genes, n_samples)
        Expression matrix with confounder effects removed.
    """
    n_samples = log_matrix.shape[1]
    n_conf    = confounder_matrix.shape[0]

    # Build design matrix: [intercept | confounders]
    intercept  = np.ones((1, n_samples))
    design     = np.vstack([intercept, confounder_matrix]).T  # (n_samples, n_conf+1)

    # OLS: coef = (XᵀX)⁻¹ Xᵀ Y  (Y = log_matrix.T)
    XtX     = design.T @ design
    XtX_inv = np.linalg.inv(XtX + np.eye(n_conf + 1) * 1e-8)
    coef    = XtX_inv @ design.T @ log_matrix.T   # (n_conf+1, n_genes)

    # Subtract only confounder contribution (not intercept) to retain mean
    fitted_conf = design[:, 1:] @ coef[1:, :]   # (n_samples, n_genes)
    adjusted    = log_matrix - fitted_conf.T      # (n_genes, n_samples)

    return adjusted.astype(np.float32)


def enforce_nonnegative(matrix):
    """
    Shift matrix to ensure all values are >= 0.

    Subtracts the global minimum, required before NMF factorization.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_genes, n_samples)

    Returns
    -------
    shifted : np.ndarray, same shape
        matrix - min(matrix), all values >= 0.
    shift_value : float
        Amount subtracted (the original minimum).
    """
    shift = matrix.min()
    return (matrix - shift).astype(np.float32), float(shift)


def filter_variable_genes(matrix, gene_names, n_top=4000,
                           guaranteed_genes=None):
    """
    Filter to top N most variable genes by per-gene variance.

    Signature matrix genes are guaranteed to be included regardless
    of their variance rank.

    Parameters
    ----------
    matrix : np.ndarray, shape (n_genes, n_samples)
        Expression matrix.
    gene_names : list of str
        Gene names corresponding to rows of matrix.
    n_top : int, default=4000
        Number of high-variance genes to retain.
    guaranteed_genes : list of str or None
        Genes to force-include (e.g., signature matrix genes).

    Returns
    -------
    filtered_matrix : np.ndarray, shape (n_selected, n_samples)
    selected_genes : list of str
    """
    variances  = matrix.var(axis=1)
    sorted_idx = np.argsort(variances)[::-1]
    top_idx    = sorted_idx[:n_top]
    top_genes  = set(np.array(gene_names)[top_idx])

    if guaranteed_genes:
        gene_set  = set(gene_names)
        force_in  = [g for g in guaranteed_genes if g in gene_set and g not in top_genes]
        force_idx = [gene_names.index(g) for g in force_in]
        all_idx   = list(top_idx) + force_idx
    else:
        all_idx = list(top_idx)

    all_idx       = sorted(set(all_idx))
    filtered      = matrix[all_idx, :]
    selected_genes = [gene_names[i] for i in all_idx]

    return filtered.astype(np.float32), selected_genes


def preprocess_pipeline(tpm_matrix, gene_names,
                         confounder_matrix=None,
                         n_top_genes=4000,
                         guaranteed_genes=None):
    """
    Full preprocessing pipeline: log → regress → shift → filter.

    Parameters
    ----------
    tpm_matrix : np.ndarray or pd.DataFrame, shape (n_genes, n_samples)
        Raw TPM values.
    gene_names : list of str
        Gene names (rows of tpm_matrix).
    confounder_matrix : np.ndarray or None, shape (n_confounders, n_samples)
        Confounder variables to regress out. None skips this step.
    n_top_genes : int, default=4000
        Number of high-variance genes to retain.
    guaranteed_genes : list of str or None
        Genes to force-include regardless of variance rank.

    Returns
    -------
    V : np.ndarray, shape (n_selected_genes, n_samples)
        NMF-ready expression matrix. Non-negative.
    selected_genes : list of str
        Gene names corresponding to rows of V.
    """
    print("Preprocessing pipeline:")

    # Step 1: log2(TPM + 1)
    log_mat = log_transform(tpm_matrix)
    print(f"  [1] log2(TPM+1)  range: [{log_mat.min():.2f}, {log_mat.max():.2f}]")

    # Step 2: regress confounders
    if confounder_matrix is not None:
        log_mat = regress_confounders(log_mat, confounder_matrix)
        print(f"  [2] Confounder regression applied  "
              f"({confounder_matrix.shape[0]} variables)")
    else:
        print("  [2] No confounder regression")

    # Step 3: shift to non-negative
    log_mat, shift = enforce_nonnegative(log_mat)
    print(f"  [3] Shifted by {-shift:.4f} → non-negative")

    # Step 4: filter to top variable genes
    log_mat, selected_genes = filter_variable_genes(
        log_mat, list(gene_names),
        n_top=n_top_genes,
        guaranteed_genes=guaranteed_genes
    )
    sig_cov = (len([g for g in (guaranteed_genes or []) if g in selected_genes]))
    print(f"  [4] Retained {len(selected_genes)} genes "
          f"(top {n_top_genes} by variance")
    if guaranteed_genes:
        print(f"       + {sig_cov}/{len(guaranteed_genes)} guaranteed genes)")
    print(f"  Non-negativity check: {'PASSED' if log_mat.min() >= 0 else 'FAILED'}")

    return log_mat, selected_genes


def build_covariate_matrix(metadata_df, covariate_cols, scale_continuous=True):
    """
    Build a covariate matrix X from a metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        Sample metadata, indexed by sample ID.
    covariate_cols : list of str
        Column names to include as covariates.
    scale_continuous : bool, default=True
        Z-score continuous covariates (mean=0, std=1).

    Returns
    -------
    X : np.ndarray, shape (n_covariates, n_samples)
        Covariate matrix.
    cov_names : list of str
        Covariate names (rows of X).
    """
    rows    = []
    names   = []
    n_samp  = len(metadata_df)

    for col in covariate_cols:
        vals = metadata_df[col].fillna(0).astype(float).values
        if scale_continuous and np.unique(vals).size > 2:
            std  = vals.std()
            vals = (vals - vals.mean()) / (std + 1e-10)
        rows.append(vals)
        names.append(col)

    X = np.vstack(rows).astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    return X, names
