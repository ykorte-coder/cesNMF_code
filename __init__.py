"""
cesNMF: Covariate-Embedded Sparse Non-Negative Matrix Factorization
=======================================================================

A novel algorithm for bulk tumor CAF deconvolution that embeds clinical
and genomic covariates directly within the NMF factorization objective.

Quickstart
----------
>>> from cesNMF import cesNMF, nndsvd_init, biological_seed_init
>>> from cesNMF.preprocessing import preprocess_pipeline
>>> from cesNMF.signature import build_signature_matrix, get_shared_genes
>>> from cesNMF.annotation import annotate_programs

>>> result = cesNMF(V, X, k=4, lambda1=0.1, lambda2=0.01, lambda3=0.01)
>>> W, H, B = result['W'], result['H'], result['B']

Reference
---------
Baca Y. et al. Covariate-Embedded Sparse NMF for CAF Deconvolution
in Pancreatic Ductal Adenocarcinoma. Bioinformatics (2026, in submission).

GitHub: https://github.com/ykorte-coder/cesNMF_code
"""

from .core import (
    cesNMF,
    nndsvd_init,
    biological_seed_init,
)

from .preprocessing import (
    log_transform,
    regress_confounders,
    enforce_nonnegative,
    filter_variable_genes,
    preprocess_pipeline,
    build_covariate_matrix,
)

from .signature import (
    build_signature_matrix,
    get_shared_genes,
)

from .annotation import (
    annotate_programs,
    rank_selection_survey,
    compute_program_summary,
)

__version__ = "0.1.0"
__author__  = "Yasmine Baca"
__email__   = "ykorte@asu.edu"
__license__ = "MIT"

__all__ = [
    # Core algorithm
    "cesNMF",
    "nndsvd_init",
    "biological_seed_init",
    # Preprocessing
    "log_transform",
    "regress_confounders",
    "enforce_nonnegative",
    "filter_variable_genes",
    "preprocess_pipeline",
    "build_covariate_matrix",
    # Signature matrix
    "build_signature_matrix",
    "get_shared_genes",
    # Annotation and analysis
    "annotate_programs",
    "rank_selection_survey",
    "compute_program_summary",
]
