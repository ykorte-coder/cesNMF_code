"""
cesNMF signature matrix construction
======================================
Functions for building two-tier CAF signature matrices from
scRNA-seq differential expression output (Seurat format).
"""

import numpy as np
import pandas as pd


def build_signature_matrix(deg_df,
                            fc_col='avg_log2FC',
                            pct1_col='pct.1',
                            pct2_col='pct.2',
                            tier1_fc=1.5,
                            tier1_pct_self=0.15,
                            tier1_pct_other=0.25,
                            tier2_fc=1.0,
                            tier2_pct_self=0.10,
                            tier2_pct_other=0.30,
                            force_include_pos=None,
                            force_include_neg=None,
                            col_names=('iCAF', 'myCAF')):
    """
    Build a two-tier signature matrix from scRNA-seq DEG output.

    Positive FC genes → first signature column (e.g., iCAF).
    Negative FC genes → second signature column (e.g., myCAF).

    Column weighting: sig[g] = |avg_log2FC| × pct_source
    This weights genes by both fold change and per-cell detection rate.

    Parameters
    ----------
    deg_df : pd.DataFrame
        Differential expression output (Seurat FindMarkers format).
        Index must be gene names. Required columns: fc_col, pct1_col, pct2_col.
    fc_col : str, default='avg_log2FC'
        Column containing average log2 fold change.
    pct1_col : str, default='pct.1'
        Column: fraction of cells in source subtype expressing this gene.
    pct2_col : str, default='pct.2'
        Column: fraction of cells in other subtype expressing this gene.
    tier1_fc : float, default=1.5
        Tier 1 (seed) fold change threshold.
    tier1_pct_self : float, default=0.15
        Tier 1 minimum per-cell detection in source subtype.
    tier1_pct_other : float, default=0.25
        Tier 1 maximum per-cell detection in other subtype.
    tier2_fc : float, default=1.0
        Tier 2 (full signature) fold change threshold.
    tier2_pct_self : float, default=0.10
        Tier 2 minimum per-cell detection in source subtype.
    tier2_pct_other : float, default=0.30
        Tier 2 maximum per-cell detection in other subtype.
    force_include_pos : list of str or None
        Canonical markers for positive-FC subtype to force-include in Tier 2.
    force_include_neg : list of str or None
        Canonical markers for negative-FC subtype to force-include in Tier 2.
    col_names : tuple of str, default=('iCAF', 'myCAF')
        Column names for positive-FC and negative-FC subtypes.

    Returns
    -------
    tier1_genes : dict
        {'pos': list, 'neg': list} — high-confidence seed genes.
    sig_matrix : pd.DataFrame
        Tier 2 signature matrix, shape (n_sig_genes, 2). L2-normalized.
    """
    fc   = deg_df[fc_col]
    pct1 = deg_df[pct1_col]
    pct2 = deg_df[pct2_col]

    # ── Tier 1: seed genes (strict thresholds, no manual additions) ────────
    tier1_pos = deg_df[
        (fc >  tier1_fc) &
        (pct1 > tier1_pct_self) &
        (pct2 < tier1_pct_other)
    ].index.tolist()

    tier1_neg = deg_df[
        (fc < -tier1_fc) &
        (pct2 > tier1_pct_self) &
        (pct1 < tier1_pct_other)
    ].index.tolist()

    print(f"Tier 1 (seed)  — {col_names[0]}: {len(tier1_pos)} | "
          f"{col_names[1]}: {len(tier1_neg)}")

    # ── Tier 2: full signature (relaxed thresholds + force-inclusions) ─────
    tier2_pos = deg_df[
        (fc >  tier2_fc) &
        (pct1 > tier2_pct_self) &
        (pct2 < tier2_pct_other)
    ].index.tolist()

    tier2_neg = deg_df[
        (fc < -tier2_fc) &
        (pct2 > tier2_pct_self) &
        (pct1 < tier2_pct_other)
    ].index.tolist()

    # Force-include canonical markers present in the DEG output
    if force_include_pos:
        available = [g for g in force_include_pos if g in deg_df.index]
        tier2_pos = list(set(tier2_pos) | set(available))
        n_forced  = len([g for g in available if g not in tier2_pos])
        print(f"  Force-included {len(available)} canonical "
              f"{col_names[0]} markers in Tier 2")

    if force_include_neg:
        available = [g for g in force_include_neg if g in deg_df.index]
        tier2_neg = list(set(tier2_neg) | set(available))
        print(f"  Force-included {len(available)} canonical "
              f"{col_names[1]} markers in Tier 2")

    print(f"Tier 2 (full)  — {col_names[0]}: {len(tier2_pos)} | "
          f"{col_names[1]}: {len(tier2_neg)}")

    # ── Build weighted signature matrix columns ────────────────────────────
    all_sig_genes = list(set(tier2_pos) | set(tier2_neg))
    sig_df = pd.DataFrame(0.0, index=all_sig_genes,
                           columns=list(col_names))

    for g in all_sig_genes:
        if g not in deg_df.index:
            continue
        f  = deg_df.loc[g, fc_col]
        p1 = deg_df.loc[g, pct1_col]
        p2 = deg_df.loc[g, pct2_col]
        if f > 0:
            sig_df.loc[g, col_names[0]] = f * p1
        else:
            sig_df.loc[g, col_names[1]] = abs(f) * p2

    # L2-normalize each column
    for col in sig_df.columns:
        norm = np.linalg.norm(sig_df[col].values)
        sig_df[col] = sig_df[col] / (norm + 1e-10)

    tier1_genes = {'pos': tier1_pos, 'neg': tier1_neg}
    return tier1_genes, sig_df


def get_shared_genes(sig_matrix, bulk_gene_names):
    """
    Find genes present in both signature matrix and bulk data.

    Parameters
    ----------
    sig_matrix : pd.DataFrame
        Signature matrix indexed by gene names.
    bulk_gene_names : list of str
        Gene names from the bulk expression matrix.

    Returns
    -------
    shared_genes : list of str
    """
    bulk_set    = set(bulk_gene_names)
    shared      = [g for g in sig_matrix.index if g in bulk_set]
    n_sig       = len(sig_matrix)
    n_shared    = len(shared)
    coverage    = 100 * n_shared / n_sig if n_sig > 0 else 0

    print(f"Shared genes: {n_shared} / {n_sig} "
          f"({coverage:.1f}% signature coverage)")

    if n_shared < 20:
        raise ValueError(
            f"Only {n_shared} shared genes found. "
            "Check that both datasets use the same gene name convention "
            "(HGNC symbols expected)."
        )
    return shared
