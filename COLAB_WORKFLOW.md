# Colab Workflow — what runs, what you get, how you decide

Short, plain guide to running this project on Google Colab. (Full research plan: `PAPER_OUTLINE.md`.)

---

## The one idea

We test whether a chest-X-ray model relies on **shortcuts** (borders, source signatures, lung-background)
instead of real disease. We measure that reliance as a single score — the **SRC** — and check whether SRC
**predicts** how badly the model breaks on data from a *different* hospital/source (accuracy drop + worse
calibration). That predictive link is called **C3**, and it is the whole paper.

---

## Two notebooks (run on Colab, GPU)

| Notebook | Models | Purpose | When |
|---|---|---|---|
| `notebooks/colab_small.ipynb` | 3 (lenet5, densenet201, resnet50) | **Go/No-Go test** — does SRC predict collapse at all? | **Run this FIRST** |
| `notebooks/colab_full.ipynb` | all 7 | Final results for the paper | Run after a GO (or directly if you want all 7) |

Both call the **same** code (`scripts/run_pipeline.py`) — only the model list and epochs differ.

---

## What one run does (9 stages, automatic)

`run_pipeline.py` chains these in order, with live progress in the cell:

1. **train** — train the models (2-phase transfer learning)
2. **eval** — in-domain accuracy + calibration (ECE/Brier/NLL)
3. **audit** — CSA interventions → **SRC certificate** per model
4. **cross** — accuracy/ECE on a *different source* → **C3 coupling** (the key result)
5. **xai** — Grad-CAM + Integrated Gradients faithfulness, in-lung saliency
6. **robust** — 7 image-corruption perturbations
7. **abstain** — accuracy/coverage (when to let the model say "I don't know")
8. **stats** — Friedman / McNemar / Holm significance tests
9. **report** — figures (PNG) + LaTeX tables

---

## What you get out (saved to Google Drive)

Everything lands in `MyDrive/cxr_data/results/` **as it is produced** (see crash-safety below):

- `results/<model>/<model>_best.keras` — trained model
- `results/<model>/metrics.json` — accuracy, ECE, etc.
- `results/<model>/certificate.json` — the **SRC** score + per-channel breakdown + validity flag
- `results/cross_source.json` — cross-source accuracy/ECE drop per model
- `results/c3_coupling.json` — **the verdict file** (does SRC predict the drop?)
- `results/<model>/{xai,robustness,abstention}.json` — supporting evidence
- `results/stats_summary.json` — significance tests
- `results/figures/*.png`, `results/tables/*.tex` — paper-ready

---

## Crash safety (the fix for the 2-hour-lost-everything problem)

- **Cell 3 symlinks `results/` to Google Drive.** Every file is written straight to Drive — a disconnect
  loses nothing.
- **Training backs up every epoch to Drive** (`BackupAndRestore`). If it dies at epoch 60/100, you resume
  from 60, not from scratch.
- **If Colab disconnects:** reconnect → re-run Cells 1–7 → re-run the pipeline cell. Finished models are
  skipped (`--resume`); a half-trained model continues from its last epoch. **Minutes lost, not hours.**

---

## How we decide (the C3 go/no-go)

After the small run, open `results/c3_coupling.json`. The notebook prints:

```
SRC -> delta_acc:  r=...  R2=...
```

Decide the threshold **before** looking (no p-hacking). Default rule:

- **GO** (`r > 0` and `R² ≥ 0.3`): SRC predicts collapse → run `colab_full.ipynb`.
  Paper spine = **"SRC predicts cross-source failure and miscalibration"** (the strong claim).
- **NO-GO** (flat / negative): SRC does not predict collapse → do **not** burn 7-model compute.
  Paper spine = **"a validated causal shortcut-audit instrument"** with descriptive findings (the safe fallback).

Either way we have a publishable paper; the verdict only decides the framing. (Details: `PAPER_OUTLINE.md` §8b.)

---

## One-time setup before Colab

1. Locally: `python scripts/package_for_drive.py` → makes data zips in `data/drive_staging/`.
2. Upload `data/drive_staging/*.zip` + `data/manifest.csv` to `MyDrive/cxr_data/`.
3. Code is already on GitHub; the notebooks clone it automatically.

---

## Repo map (what each piece is)

```
scripts/run_pipeline.py     one entrypoint, runs stages 1-9 (notebooks call this)
scripts/train.py            stage 1 (training)
scripts/evaluate.py         stage 2 (in-domain metrics)
scripts/build_manifest.py   one-time: build the source-labeled image index
scripts/package_for_drive.py one-time: zip data for Drive upload

src/shortcut/csa.py             CSA — the shortcut interventions
src/shortcut/src_certificate.py SRC — the certificate score
src/shortcut/cross_domain.py    cross-source eval + C3 coupling + LOMO
src/shortcut/abstention.py      accuracy/coverage curves
src/xai/explain.py              Grad-CAM / Integrated Gradients + faithfulness
src/robustness/perturbations.py 7 corruption families
src/reporting/figures.py        figures + LaTeX tables
src/evaluation/metrics.py       metrics, calibration, significance tests
src/data/, src/models/, src/training/   data pipeline, model zoo, trainer
```
