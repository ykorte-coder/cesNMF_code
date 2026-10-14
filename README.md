# cesNMF: Covariate-Embedded Sparse NMF for CAF Deconvolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A novel Non-Negative Matrix Factorization algorithm for bulk tumor
cancer-associated fibroblast (CAF) deconvolution that embeds clinical
and genomic covariates **directly within the factorization objective function**,
rather than testing them post-hoc.

---

## Overview

Standard NMF discovers latent gene programs that explain expression variation.
**cesNMF** extends this by requiring programs to also be consistent with patient
clinical and genomic covariate structure — ensuring that the programs discovered
are simultaneously transcriptionally coherent and clinically relevant.

### Novel objective function

```
L = ‖V − WH‖²_F          (reconstruction)
  + λ₁‖H − BX‖²_F        (covariate embedding — novel term)
  + λ₂‖W‖²_F             (L2 ridge on W)
  + λ₃‖W‖₁               (L1 sparsity on W)
```

Where:
- **V** (genes × samples) — preprocessed bulk RNA-seq expression matrix
- **W** (genes × programs) — gene loading matrix
- **H** (programs × samples) — sample score matrix
- **B** (programs × covariates) — covariate coefficient matrix (estimated jointly)
- **X** (covariates × samples) — clinical and genomic covariate matrix

### Three algorithmic contributions

1. **Novel objective function** with derived multiplicative update rules for W and H,
   and a closed-form OLS update for B
2. **Biologically-seeded NNDSVD initialization** — anchors known CAF programs from
   scRNA-seq data while preserving unsupervised discovery capacity
3. **Dual-decomposition framework** — cesNMF (novel programs) paired with DWLS
   (reference-based confirmatory) for cross-validation

---

## Installation

```bash
git clone https://github.com/[username]/cesNMF.git
cd cesNMF
pip install -e .
pip install -r requirements.txt
```

---

## Quickstart

```python
from cesNMF import cesNMF, biological_seed_init
from cesNMF.preprocessing import preprocess_pipeline, build_covariate_matrix
from cesNMF.signature import build_signature_matrix, get_shared_genes
from cesNMF.annotation import annotate_programs

# Run cesNMF
result = cesNMF(
    V       = V_sub,      # genes × samples
    X       = X,          # covariates × samples
    k       = 4,
    lambda1 = 0.1,        # covariate embedding strength
    lambda2 = 0.01,       # L2 ridge on W
    lambda3 = 0.01,       # L1 sparsity on W
    W_init  = W_init,     # biologically seeded initialization
    H_init  = H_init,
    program_names = ['iCAF', 'myCAF', 'Novel_1', 'Novel_2'],
)

W = result['W']    # gene loadings
H = result['H']    # sample program scores
B = result['B']    # covariate contributions
```

---

## Demo

Run the complete pipeline on a synthetic dataset (no patient data required):

```bash
python data/generate_synthetic.py
python examples/demo_synthetic.py
```

---

## Run tests

```bash
pip install pytest
pytest tests/ -v
```

---

## Hyperparameter guide

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | — | Number of programs — select via RSS elbow + stability analysis |
| `lambda1` | 0.1 | Covariate embedding strength — novel parameter |
| `lambda2` | 0.01 | L2 ridge on W — numerical stability |
| `lambda3` | 0.01 | L1 sparsity on W — target 40-70% W sparsity |
| `max_iter` | 500 | Maximum iterations |
| `tol` | 1e-6 | Convergence tolerance |

---

## Data availability

The algorithm code is freely available in this repository. The clinical PDAC
cohort used in the manuscript (Caris Life Sciences) is not publicly available
due to institutional data use restrictions. The scRNA-seq reference dataset
(GSE155698; Steele et al. 2021) is publicly available via NCBI GEO.

---

## Citation

```bibtex
@article{baca2026cesnmf,
  title   = {Covariate-Embedded Sparse Non-Negative Matrix Factorization
             for Cancer-Associated Fibroblast Deconvolution in Pancreatic
             Ductal Adenocarcinoma},
  author  = {Baca, Yasmine},
  journal = {Bioinformatics},
  year    = {2026},
  note    = {In submission}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Yasmine Baca · [your email]
Biomedical Informatics and Data Science · Arizona State University
Senior Molecular Research Analyst · Caris Life Sciences
