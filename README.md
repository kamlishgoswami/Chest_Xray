# Counterfactual Shortcut Audit + Reliance Certificate for Chest X-rays

> Code for the paper (target: **Scientific Reports**, Nature Portfolio, Q1):
> *A Counterfactual Shortcut Reliance Certificate that Predicts Cross-Domain Failure and
> Miscalibration in Chest X-ray Classifiers.*

We introduce a **Counterfactual Shortcut Audit (CSA)** that causally measures how much a chest
X-ray classifier relies on nuisance "shortcut" channels, distilled into a per-model **Shortcut
Reliance Certificate (SRC)** that **predicts cross-source accuracy collapse and miscalibration**.
See [`PAPER_OUTLINE.md`](PAPER_OUTLINE.md) for the full plan and [`METHODS_SECTION.md`](METHODS_SECTION.md) for draft methods.

## Thesis in one sentence

By intervening on distinct shortcut channels (image borders/markers, background outside the lung
field, source-specific intensity signatures) while preserving pathology, CSA causally quantifies a
model's shortcut reliance; the resulting SRC certificate predicts which models fail under
cross-source shift and become miscalibrated — an auditable pre-deployment reliability instrument.

## Repository structure

```
src/
  data/         Phase 0 — source-labeled dataset acquisition + loaders
  models/       7-model zoo + two-phase transfer learning
  training/     Trainer
  shortcut/     CORE — csa.py (audit), src_certificate.py (SRC), cross_domain.py (cross-source + confounder sep.)
  robustness/   Covariate-shift suite (SSP) — CORROBORATING ONLY (cite Hendrycks 2019; not claimed novel)
  xai/          XAI faithfulness/stability — corroborates SRC (C4)
  evaluation/   Metrics, calibration (ECE/Brier/NLL + temp scaling), selective prediction, statistics
  utils/        Seeding/determinism (done)
configs/        Experiment configs (YAML)
scripts/        Entrypoints (download data, run each phase)
data/           raw/ + masks/ (gitignored)
results/        Outputs (gitignored)
notebooks/      Colab notebooks
paper/          LaTeX, figures, tables
legacy/         Archived original 30-model pipeline (salvage source for C4/C5)
docs/_legacy/   Archived old guides
```

## Pipeline phases (maps to PAPER_OUTLINE.md §10)

| Phase | What | Modules |
|---|---|---|
| **0** | Acquire + source-label datasets (Kaggle), build loaders, lung masks | `src/data` |
| **1** | Train 7 models; in-domain Table A incl. ~99% reproduction | `src/models`, `src/training`, `src/evaluation` |
| **2** | **CORE: CSA audit + negative controls → SRC certificate** | `src/shortcut/csa.py`, `src_certificate.py` |
| **3** | Cross-source collapse + **confounder separation** + C3 coupling regression | `src/shortcut/cross_domain.py`, `src/evaluation` |
| **4** | Corroboration: XAI-vs-SRC (C4) + certificate-gated abstention (C5) | `src/xai`, `src/evaluation` |
| **5** | Figures, tables, manuscript, reproducibility package | `paper/` |

## Distinct mechanisms (do not conflate — see PAPER_OUTLINE §3.7)

- **Cross-source / domain shift** — real source change; the *outcome* SRC predicts.
- **Covariate-shift perturbations (SSP)** — synthetic whole-image corruptions; *corroborating only*, not novel.
- **Counterfactual interventions (CSA)** — targeted per-channel causal probes; *the method*. Never called "data shift".

## Status

🚧 v4 scaffold (single paper, Scientific Reports, soundness-optimized). Core modules stubbed in
`src/shortcut/`. Legacy pipeline archived in `legacy/old_pipeline/` and salvaged for C4/C5.

## Reproducibility

Seeded (`SEED=42`), TRIPOD-AI + CLAIM compliant. Determinism in `src/utils/seeding.py`;
exact hyperparameters in `configs/`. Negative controls (CSA §3.3.3) are the soundness centerpiece.
