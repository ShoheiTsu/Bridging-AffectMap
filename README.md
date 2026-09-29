# Bridging-AffectMap

Analysis code and Japanese group-mean data for the Bridging-AffectMap project:
frozen CLIP features, group-specific affective readouts, affine bridges across gender and culture, and exemplar translation.

## Contents

| Path | Description |
|------|-------------|
| `data/japan_group_means_va.csv` | Japanese cohort group means (96 OASIS images) |
| `code/` | Scripts used for the reported analyses and figures |
| `requirements.txt` | Package versions used in the reported runs |

## Japanese group means

See `data/japan_group_means_va.csv`. Columns: `file_name`, `valence_mean`, `valence_sd`, `arousal_mean`, `arousal_sd`, `n_subjects` (1–7 scale; 43–48 raters per image).

## Code included here

Scripts cover:

- conditional autoencoder criteria (shared vs split encoders, swap recovery, CKA)
- five-method mappings (including OT) and λ vs Φ comparisons
- shared-reference generalisation and minimal-coverage anchors
- Japan culture-bridge reliability, residuals, and leave-one-subject-out checks
- exemplar / predictive translation
- main-text and Supplementary figure generation for the submitted set

Withdrawn exploratory Supplementary figures (near/mid/far layers, UMAP twist field, category-restricted 4×4 matrices, reliability-shield figure panels, case-translation validation/split figure panels, and related one-offs) are **not** included. Superseded testfig plotters and private-data prep utilities are also omitted.

Paths inside the scripts assume a project root that also contains OASIS score tables and CLIP feature arrays. Adjust `code/config.py` if needed.

## Environment

Python 3.11; Ubuntu 22.04; Intel Core i9-13900KS; NVIDIA RTX 6000 Ada Generation; PyTorch CUDA 12.1.

```bash
pip install -r requirements.txt
```

## Stimuli

OASIS images and normative scores follow Kurdi, Lozano & Banaji (2017). Redistribute only as permitted by that licence.
