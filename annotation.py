"""
cesNMF program annotation and analysis utilities
==================================================
Functions for annotating NMF programs, running cross-validation
against DWLS, and computing program-level statistics.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def annotate_programs(W, shared_genes, sig_matrix,
                       known_marker_dict=None,
                       novelty_threshold=0.30,
                       top_n=30):
    """
    Annotate NMF programs using Spearman correlation and marker overlap.

    Programs seeded from the signature matrix are labeled as known subtypes.
    Novel programs are those with max |r| < novelty_threshold with all
    reference subtypes.

    Parameters
    ----------
    W : np.ndarray, shape (n_genes, k)
        Gene loading matrix from cesNMF.
    shared_genes : list of str
        Gene names corresponding to rows of W.
    sig_matrix : pd.DataFrame, shape (n_genes, n_subtypes)
        Reference signature matrix.
    known_marker_dict : dict or None
        {'subtype_name': [list of marker genes], ...}
        Used to compute marker overlap fraction.
    novelty_threshold : float, default=0.30
        Programs with max |r| < threshold are labeled novel.
    top_n : int, default=30
        Number of top genes per program to use for annotation.

    Returns
    -------
    annotations : pd.DataFrame
        One row per program with columns:
        annotation, max_r, max_r_subtype, top_genes, marker_overlap.
    correlation_matrix : pd.DataFrame
        Spearman r between each program and each reference subtype.
    """
    k         = W.shape[1]
    sig_genes = [g for g in sig_matrix.index if g in shared_genes]
    sig_idx   = [shared_genes.index(g) for g in sig_genes]
    sig_vals  = sig_matrix.loc[sig_genes].values   # (n_sig_genes, n_subtypes)
    W_sig     = W[sig_idx, :]                       # (n_sig_genes, k)

    # Spearman correlation matrix
    n_sub  = sig_matrix.shape[1]
    cor_mat = np.zeros((k, n_sub))
    for i in range(k):
        for j in range(n_sub):
            r, _ = spearmanr(W_sig[:, i], sig_vals[:, j])
            cor_mat[i, j] = r if not np.isnan(r) else 0.0

    cor_df = pd.DataFrame(
        cor_mat,
        columns=sig_matrix.columns,
        index=[f"Program_{i+1}" for i in range(k)]
    )

    # Top genes per program
    top_genes_dict = {}
    for i in range(k):
        idx = np.argsort(W[:, i])[::-1][:top_n]
        top_genes_dict[f"Program_{i+1}"] = [shared_genes[j] for j in idx]

    # Marker overlap
    overlap_dict = {}
    if known_marker_dict:
        for i in range(k):
            prog = f"Program_{i+1}"
            top  = set(top_genes_dict[prog])
            overlap_dict[prog] = {
                subtype: round(len(top & set(mks)) / max(len(mks), 1), 3)
                for subtype, mks in known_marker_dict.items()
            }

    # Build annotation table
    rows = []
    for i in range(k):
        prog     = f"Program_{i+1}"
        max_r    = cor_df.loc[prog].abs().max()
        max_sub  = cor_df.loc[prog].abs().idxmax()
        top_gene_str = ', '.join(top_genes_dict[prog][:10])

        if max_r >= novelty_threshold:
            ann = f"{max_sub}-like (r={cor_df.loc[prog, max_sub]:.3f})"
        else:
            ann = f"Novel program (max|r|={max_r:.3f})"

        rows.append({
            'program'        : prog,
            'annotation'     : ann,
            'max_abs_r'      : round(max_r, 4),
            'max_r_subtype'  : max_sub,
            'top_10_genes'   : top_gene_str,
            'is_novel'       : max_r < novelty_threshold,
        })

        if known_marker_dict and prog in overlap_dict:
            for subtype, frac in overlap_dict[prog].items():
                rows[-1][f'marker_overlap_{subtype}'] = frac

    annotations = pd.DataFrame(rows).set_index('program')
    return annotations, cor_df


def rank_selection_survey(V, k_range=range(2, 7),
                           n_runs=10, max_iter=300,
                           seed=0, verbose=True):
    """
    Survey reconstruction error and stability across rank values.

    Runs NMF (via sklearn) at each k with n_runs random initializations.
    Reports mean RSS (elbow method) and SD of RSS (stability).

    Parameters
    ----------
    V : np.ndarray, shape (n_genes, n_samples)
        Preprocessed, non-negative expression matrix.
    k_range : iterable, default=range(2, 7)
        Rank values to survey.
    n_runs : int, default=10
        Number of random seeds per rank.
    max_iter : int, default=300
        Max iterations per NMF run.
    seed : int, default=0
        Base random seed.
    verbose : bool, default=True

    Returns
    -------
    results : pd.DataFrame
        Columns: k, mean_rss, std_rss.
    """
    try:
        from sklearn.decomposition import NMF as skNMF
    except ImportError:
        raise ImportError(
            "scikit-learn is required for rank_selection_survey(). "
            "Install with: pip install scikit-learn"
        )

    if verbose:
        print(f"Running rank survey: k={list(k_range)}, {n_runs} runs each...")

    records = []
    for k in k_range:
        errors = []
        for run in range(n_runs):
            model = skNMF(
                n_components=k, init='nndsvda',
                random_state=seed + run,
                max_iter=max_iter,
                l1_ratio=0.0, alpha_W=0.0
            )
            W_r   = model.fit_transform(V.T)
            recon = np.linalg.norm(V.T - W_r @ model.components_, 'fro') ** 2
            errors.append(recon)

        mean_err = np.mean(errors)
        std_err  = np.std(errors)
        records.append({'k': k, 'mean_rss': mean_err, 'std_rss': std_err})

        if verbose:
            print(f"  k={k}  mean RSS={mean_err:.2f}  SD={std_err:.2f}")

    return pd.DataFrame(records)


def compute_program_summary(H, gene_names_W, W,
                             sample_names=None, top_n=30):
    """
    Compute summary statistics for each program.

    Parameters
    ----------
    H : np.ndarray, shape (k, n_samples)
        Sample program scores.
    gene_names_W : list of str
        Gene names corresponding to rows of W.
    W : np.ndarray, shape (n_genes, k)
        Gene loading matrix.
    sample_names : list of str or None
    top_n : int, default=30

    Returns
    -------
    summary : pd.DataFrame
        One row per program: mean_score, median_score, sparsity, top_genes.
    """
    k = H.shape[0]
    rows = []
    for i in range(k):
        top_idx  = np.argsort(W[:, i])[::-1][:top_n]
        top_gene_str = ', '.join([gene_names_W[j] for j in top_idx[:10]])
        rows.append({
            'program'         : f"Program_{i+1}",
            'mean_score'      : round(float(H[i, :].mean()), 4),
            'median_score'    : round(float(np.median(H[i, :])), 4),
            'std_score'       : round(float(H[i, :].std()), 4),
            'W_col_sparsity'  : round(float((W[:, i] < 1e-3).mean()), 4),
            'top_10_genes'    : top_gene_str,
        })
    return pd.DataFrame(rows).set_index('program')
