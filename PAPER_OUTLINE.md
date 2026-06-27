# Paper Outline — MERGED MASTER (v5, single paper, Scientific Reports)

> **Status:** v5 — merges the novelty ledger (former PAPER_OUTLINE §0a) with the verified Q1
> benchmark set (former BENCHMARK_PAPERS.md) into ONE plan for ONE paper.
> **Target:** Scientific Reports (Nature Portfolio). *(Verify current quartile / IF / review criterion
> on the journal site before relying on them — not independently confirmed here.)*
> **Core:** Counterfactual Shortcut Audit (CSA) + Shortcut Reliance Certificate (SRC) for chest X-rays.
> **Owner:** Kamlish Goswami · **Last updated:** 2026-06-26
>
> **VERIFICATION RULE for this document:** every factual claim about an external paper traces to a
> full-text-read PDF (11 papers — see §0a + §0b). Every claim about *our* datasets/models/code traces to
> the repo (`configs/datasets.yaml`, `configs/train.yaml`, `src/`). Planned-but-not-yet-achieved results
> are marked **[PLANNED]**. Anything unverified is marked **⛔ / ⚠️**. No assumptions.

---

## 00. The two-part novelty (twin contributions, equal billing)

This paper sits at the intersection of two literatures we have each read in full:

1. **Applied CXR-classification papers (§0b, 5 verified Q1)** — reach 96–99% on POOLED multi-source data,
   and *each, in its own words*, either omits or only-qualitatively-asserts shortcut/generalization checks.
2. **Shortcut / causal-audit papers (§0a, 6 verified)** — develop shortcut detection, but none deliver a
   per-model *certificate* coupled to calibration, and the closest (DeGrave 2021) does its CXR intervention
   by hand on single images without a portable score.

**Twin contribution (equal billing):**

- **(M) Method:** the **SRC** — a per-model, per-channel shortcut-reliance *certificate* with a validity
  gate — and its **predictive coupling** (SRC → cross-source ΔAcc AND ECE). *No paper in either literature
  does this.*
- **(A) Applied audit:** we run that instrument on **the same kind of 96–99% CXR classifiers the benchmark
  papers publish**, on **the same canonical datasets**, and show the cross-source collapse + shortcut
  reliance those papers never measured.

> **Honest scope (do NOT overclaim):** "first to causally intervene on CXR shortcuts with a null control"
> is FALSE — DeGrave, Janizek & Lee (Nat. Mach. Intell. 2021) did that. Our method delta is *certification +
> calibration coupling + automation/per-channel systematization*, not the existence of causal intervention.
> See the ⚠️ HARDEST guard in §0a.

---

## 0a. Novelty Ledger — shortcut/causal neighbors (6, full-text read)

Venues vary (2 Q1 journals: Nat. Mach. Intell. 2021, MTAP/Springer 2024; 3 conferences: ECCV/NeurIPS/ICLR;
1 workshop preprint). **NOT all disclaim our exact capability** — DeGrave (2021) already performs
null-controlled counterfactual edits on CXR shortcuts. Do not present this ledger as proof of blanket novelty.

| Neighbor                                                                                        | What it does (verified)                                                                                                                                                                                                                                                                                           | Our delta                                                                                                                                                                                                                                                                                                                                    | Link                                                                      |
| ----------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Occlusion eval — Springer 2024 (MTAP)** ✅READ                                         | Single black-box lung occlusion + random-region control; reliance correlates with cross-dataset AUC instability (*SD_all*). COVID-vs-negative; 6 models incl. DenseNet-121, DarkCovidNet, ResNet, Conv-4                                                                                                        | No per-channel decomposition; no certificate; no causal-intervention framing; no calibration analysis                                                                                                                                                                                                                                        | https://doi.org/10.1007/s11042-024-18543-y                                |
| **FastDiME — ECCV 2024 (conference, not journal)** ✅READ                                | Diffusion counterfactual*generation* adding/removing ONE named shortcut (pacemaker/drain/ruler) + a per-model reliance pipeline (MAD/MD). CheXpert/NIH, ISIC, CelebA                                                                                                                                            | Flips one annotated semantic shortcut; needs annotations + a trained generator (they flag generator bias); no orthogonal-channel taxonomy, no sham null, no certificate; reliance = in-vs-balanced AUROC gap only — no cross-source OOD, no calibration                                                                                     | https://doi.org/10.1007/978-3-031-73016-0_20 · arXiv 2312.14223          |
| **Causal Attribution — Gordaliza et al. 2025** ✅READ                                    | Shapley + do-operator + importance-sampling attribution of MS-lesion**segmentation** drops to which *real mechanism shifted* — acquisition P(X\|S) vs annotation P(Y\|X,A). nnU-Net, MSSEG2016                                                                                                           | Attributes*real shifts that occurred*, not a model's reliance on a shortcut channel; segmentation not classification; no shortcut intervention, no sham, no certificate, no calibration                                                                                                                                                    | arXiv 2512.09094 (NeurIPS 2025**Workshop preprint**, not a journal) |
| **Spuriosity Rankings — NeurIPS 2023 (CV)** ✅READ                                       | Ranks within-class images by spuriosity via human-annotated neural features; bias = spurious gap (high/low-spuriosity acc) across 89 ImageNet models. App. B runs gray/blur/patch-rotate region interventions                                                                                                     | Natural per-class cues on ImageNet/Waterbirds/CelebA, not orthogonal nuisance channels; needs feature annotation; bias = in-distribution gap, not cross-source OOD; no calibration, no certificate                                                                                                                                           | arXiv 2212.02648 (**NeurIPS 2023**, not CVPR)                       |
| **GroupDRO — ICLR 2020 (CV)** ✅READ                                                     | Worst-group-loss training; needs per-example group labels; works only with strong regularization in overparam. regime. Waterbirds/CelebA/MultiNLI                                                                                                                                                                 | Training-time*mitigation*, not an audit; needs annotated groups (we need none); no causal intervention, no sham, no certificate, no calibration, no cross-source eval                                                                                                                                                                      | arXiv 1911.08731                                                          |
| **DeGrave, Janizek & Lee — Nat. Mach. Intell. 2021 (Q1) — CLOSEST CXR NEIGHBOR** ✅READ | Names CXR shortcut channels (markers, border radiopacity, positioning, AP/PA projection, sex) via Expected-Gradients + CycleGAN. Cross-source OOD shown (AUC 0.99→0.70–0.76). Runs**interventional edits WITH a random-patch null control** (Fig 3, single-image + population p-values). 10 architectures | Their edits are**manual, per-feature, single-image, qualitative** — not a systematic orthogonal-channel taxonomy with an inside-lung null and an automated metric; **no per-model certificate + validity gate**; **no calibration coupling** (AUC/log-odds only); reliance not distilled into a portable predictive score | https://doi.org/10.1038/s42256-021-00338-7                                |

> **Honesty note:** all 6 rows are full-PDF verified. Rows once cited but never read were REMOVED
> (CLIP Fairness Springer 2026, "Lung Attention Ratio", Maguolo & Nanni, DB-structure bias, ElRep, PDE) —
> re-add only after reading. Gordaliza is a workshop preprint, not a Q1 journal.

> **Do NOT claim C3 (occlusion-reliance ↔ cross-dataset drop) as first-of-kind:** Haynes et al. (2024)
> already show it correlationally (*SD_all*). Our delta = causal per-channel decomposition + certificate +
> calibration coupling; frame C3 as *quantifying and certifying* a link they observed qualitatively.

> **Do NOT claim "counterfactual intervention to quantify shortcut reliance" as first-of-kind:**
> FastDiME (ECCV 2024) flips a shortcut counterfactually and measures the prediction change (MAD/MD).
> Our delta: (a) orthogonal channel decomposition vs. one named shortcut; (b) annotation-free, generator-free,
> deterministic intervention; (c) sham null-control validity gate; (d) certificate + cross-source-OOD +
> calibration coupling.

> **⚠️ HARDEST guard — DeGrave et al. (2021).** Do NOT describe it as "shortcuts exist / correlational."
> In the SAME COVID-CXR domain they already (i) name the shortcut channels, (ii) show the cross-source AUC
> collapse, (iii) run counterfactual edits with a random-patch NULL control and population-level significance.
> **"We are first to causally intervene on CXR shortcuts with a null control" = instant desk-reject if a
> reviewer is DeGrave.** Our ONLY defensible delta: (a) systematic orthogonal-channel taxonomy + inside-lung
> null; (b) automated metric across a model suite (theirs manual/single-image); (c) per-model certificate +
> validity gate; (d) calibration coupling. **Lead related work with this paper.**

---

## 0b. Benchmark Ledger — applied CXR-classification papers (5 verified Q1; full-text read)

These are the "reach 96–99% on pooled data, never verify disease-not-dataset" papers our applied audit targets.
**Anchor differentiators (all rows):** they POOL + random-split → we keep SOURCE-LABELED + cross-source;
they report accuracy/heatmaps → we causally certify reliance; they stop at "it works" → we predict collapse.

| #  | Paper                                        | Venue / yr                   | Dataset (verified)                                                        | Their admitted gap = our contribution                                                                                                                         |
| -- | -------------------------------------------- | ---------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| B1 | Deepak & Bhat — multi-stage 17-class ✅READ | Elsevier ESWA 2025 (Q1)      | pooled Kermany+Patel+QU-Ex+NIH,**1,700 imgs**, 100/class test       | *Own words:* no external validation, no XAI, no ablation; admits small-test "statistical uncertainty"                                                       |
| B2 | Patel et al. — explainable TL ⭐ ✅READ     | Elsevier AEJ 2024 (Q1)       | pooled Cohen+Kermany+NIAID+LIDC,**7,389 imgs**, 80/20               | *Own words:* Grad-CAM is **"correlative, rather than causative"**; single pooled set, no faithfulness metric, no calibration/abstention/bias analysis |
| B4 | Aljuaid et al. — 6-CNN comparison ✅READ    | Elsevier BSPC 2026 (Q1)      | single balanced Kaggle,**6,939 imgs**, 5-fold                       | *Claims* "no spurious correlations" but only via **qualitative Grad-CAM eyeballing**, no cross-dataset test to verify; no calibration                 |
| B5 | Deepak & Bhat — 3-stage COVID/TB ✅READ     | Springer Soft Comp 2026 (Q1) | **COVID-QU-Ex + TB-DB — OUR EXACT SOURCES**, 33,920 imgs           | *Own words:* defers cross-independent-dataset validation, interpretability, AND imaging-condition robustness to "future work"                               |
| B6 | Slimani & Bentourkia — DWT+seg+GAN ✅READ   | Springer JIIM 2025 (Q1)      | **COVID-QU-Ex + Shenzhen + Montgomery + JSRT — OUR EXACT SOURCES** | No XAI at all; no cross-source*classification* test; no calibration/bias. (Bonus: their U-Net++ is a usable lung-mask source for our background channel)    |

> **B2 (Patel) ⭐** = cite FIRST in applied related work: its admitted "correlative, not causative" XAI is
> the cleanest motivation for CSA. **B5 + B6** = strongest overlap (our exact datasets + backbones) → ideal
> "same data, we go further" comparison.
> **Benchmark-use rule:** we do NOT compare our 4-class cross-source numbers to their pooled 96–99%
> (different tasks). We reproduce a pooled high-accuracy baseline ourselves, then expose collapse.
> **Dropped (decision 2026-06):** the self-imposed "need a valid IEEE 2024–26 paper for publisher balance"
> requirement — cite the 5 verified Elsevier/Springer benchmarks as-is; no fabricated IEEE entry.

---

## 1. Working Title (candidates)

1. **"A Counterfactual Shortcut Reliance Certificate that Predicts Cross-Domain Failure and Miscalibration
   in Chest X-ray Classifiers"** *(recommended)*
2. "Certifying What Chest X-ray Models Look At: Causal Per-Channel Shortcut Auditing for Reliable Deployment"
3. "From 99% to a Certificate: Auditing Shortcut Reliance in High-Accuracy Chest-Radiography AI"

---

## 2. Contributions (each independently verifiable = "sound")

- **C1 — CSA (method):** standardized, annotation-free, generator-free *interventional* protocol isolating
  each shortcut channel's causal effect. *Delta vs DeGrave: systematic per-channel + automated + inside-lung null.*
- **C2 — SRC (method, TWIN-A):** auditable per-model certificate + per-channel breakdown + bootstrap CI +
  **validity gate** (sham must be null). *No read paper does this.*
- **C3 — Predictive coupling (method, TWIN-A core):** SRC predicts cross-source ΔAcc AND ECE (regression,
  R², CI). *The calibration link is unique among all 11 read papers.*
- **C4 — Applied benchmark audit (TWIN-B):** run CSA/SRC on 96–99%-style CXR classifiers (cf. B1–B6) on
  the canonical datasets; show the cross-source collapse + reliance they assert-away or never test.
- **C5 — XAI corroboration + certificate-gated abstention (TWIN-A support):**
  (a) **two independent-family** saliency methods (Grad-CAM + Integrated Gradients) quantified by
  **insertion/deletion AUC, faithfulness correlation, stability/SSIM** — these degrade predictably as SRC rises;
  (b) **in-lung saliency fraction** (saliency mass inside the lung mask, reusing CSA masks) — an *independent*
  cross-check: high-SRC models place more saliency outside the lung; (c) SRC + calibration →
  accuracy/coverage abstention curves.

*(C1–C3 = method core; C4 = applied core; C5 = completeness. All Colab-feasible — the audit is inference-only.)*

---

## 3. Reviewer Defense Matrix

| Objection                                    | Answer                                                                                                                                                                                                       |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| "Shortcuts are old news / DeGrave did this." | Conceded and cited as closest neighbor. Our delta is the**certificate + calibration coupling + automated per-channel systematization**, NOT the existence of causal CXR intervention (see §0a guard). |
| "Just the 2024 occlusion paper?"             | Binary/correlational/single-region. We add per-channel causal attribution + certificate + calibration coupling — none of which it attempts.                                                                 |
| "Isn't this FastDiME?"                       | FastDiME needs annotations + a trained generator (own-stated bias risk); reliance = in-vs-balanced AUROC only. We are annotation/generator-free, with sham null, certificate, cross-source OOD, calibration. |
| "Real shortcuts or noise?"                   | **Sham (no-op) + inside-lung negative controls** show null effect — soundness centerpiece + SRC validity gate.                                                                                        |
| "Is the collapse shortcut or genuine shift?" | C3b confounder-separation: regress ΔAcc on SRC + check CSA-masking selectively recovers cross-source accuracy.                                                                                              |
| "Why does auditing 99% papers matter?"       | B1/B4/B5*explicitly* defer or only-eyeball generalization/XAI; B2 concedes correlational XAI. We supply the test they omit, on their datasets.                                                             |
| "Generality / reproducible / over-claiming?" | 7 architectures × multiple datasets; seeds/configs/public data/released code (TRIPOD-AI/CLAIM); we claim a*validated instrument* with stated limits (§8).                                                |

---

## 4. Methods

- **4.1 Datasets, source labeling, leakage control.** Verified registry in `configs/datasets.yaml`:
  classes **[Covid, Normal, Pneumonia, TB]**, seed 42, 224×224; patient-level split (0.70/0.15/0.15),
  perceptual-hash dedup before split. (See §6 table.)
- **4.2 Model zoo (7).** Verified in `configs/train.yaml`: densenet201, efficientnetb0, resnet50,
  mobilenetv3large, xception, vit, lenet5 (no-transfer baseline). Two-phase transfer learning.
- **4.2b Minimal preprocessing (THESIS-CRITICAL).** CSA runs on **minimally preprocessed** images
  (resize + [0,1] normalize only). Heavy enhancement (DAE / CLAHE / gamma / morphological opening) would
  *normalize away the very signals CSA measures* — gamma/CLAHE flatten the **source-signature** channel,
  morphological opening erases **border/marker** artifacts — confounding the audit. Any enhancement is
  therefore studied as an **ablation axis** ("does preprocessing mask shortcut reliance?"), never baked in
  as a fixed front-end.
- **4.3 CSA — C1.** (a) Channel operators: border/marker masking; background replacement outside lung field
  (masks from `datasets.yaml` `masks:true` sources + U-Net for Kermany); source-signature normalization.
  (b) Causal estimator: Δ class-prob per intervention vs control, aggregated per model, with CI.
  (c) **Validity controls:** sham (no-op) + inside-lung interventions → expect null. (d) Pathology-preservation
  spot check.
- **4.4 SRC — C2.** Definition, per-channel decomposition, bootstrap CI, cross-model normalization, validity gate.
- **4.5 Evaluation suite.** Standard metrics + calibration (ECE/Brier/NLL + temperature scaling) + robustness/SSP
  + quantitative XAI + selective prediction.
- **4.5b XAI corroboration of SRC — C5.** Exactly **two** saliency methods from *different families* for
  genuine cross-method robustness: **Grad-CAM** (CAM/gradient) + **Integrated Gradients** (path/axiomatic).
  Deliberately excluded: **occlusion** (perturbation-based → overlaps CSA, not an independent check);
  **LIME/SHAP** (slow, model-agnostic → add cost, not evidence). Quantitative faithfulness: insertion/deletion
  AUC, faithfulness correlation, stability/SSIM under mild perturbation; plus the **in-lung saliency fraction**
  vs SRC. Hypothesis: faithfulness ↓ and out-of-lung saliency ↑ as SRC ↑ — an independent confirmation of the
  intervention-based audit. *(Frame as corroboration, not novelty: DeGrave already showed attention falls
  outside the lung qualitatively.)*
- **4.6 Predictive coupling — C3.** Regress cross-source ΔAcc and ECE on SRC; R², slope, CI; partial
  correlation controlling for in-domain accuracy.
- **4.6a SRC vs. simpler-predictor baselines — C3 (MUST, near-free).** Regress ΔAcc/ΔECE on each trivial
  alternative — in-domain accuracy, mean softmax confidence, predictive entropy, ECE, and the Grad-CAM
  out-of-lung fraction (already computed in §4.5b) — and report head-to-head R² vs SRC. *Establishes SRC adds
  predictive value beyond what's freely available; answers "is SRC even necessary?"* Reuses existing artifacts
  (`metrics.json`, `xai.json`, `certificate.json`); no new training. **[PLANNED]**
- **4.6b Leave-one-model-out (LOMO) prediction — C3 (MUST, earns the word "predicts").** Fit the SRC→ΔAcc/ΔECE
  regression on 6 models, predict the held-out 7th, repeat; report predicted-vs-actual + LOMO R²/MAE. *Converts
  the in-sample fit into genuine out-of-sample prediction — pre-empts the "your R² is circular" objection.*
  Pure analysis on existing values; near-zero cost. **[PLANNED]**
- **4.6c Post-temperature-scaling miscalibration — C3 (MUST, defends the "miscalibration" claim).** Apply the
  val-fitted temperature T to the cross-source logits, recompute ECE, and re-run the SRC→ΔECE coupling. *Shows
  SRC predicts miscalibration that temperature scaling CANNOT fix — i.e. SRC is not merely ordinary calibration
  error.* Reuses `fit_temperature` (already in `metrics.py`). **[PLANNED]**
- **4.6d Seeds + bootstrap CI on the coupling — C3 (addresses n=7).** Train 3 seeds/model (21 points) and report
  a bootstrap CI on the regression slope. *Defeats the "n too small for regression" objection.* **Compute-gated:
  run seeds on the 3-model go/no-go FIRST; expand to 7×3 only if the coupling holds** (§8b). Per-class/per-source
  breakdown and mixed-effects models are deliberately **excluded** (over-engineering at this n; reads as fishing).
  **[PLANNED]**
- **4.7 Confounder-separation — C3b.** Decompose collapse into shortcut-attributable (explained by SRC + CSA)
  vs genuine-shift residual; corroborate by CSA-masking the cross-source test set.
- **4.8 Terminology discipline.** (a) cross-source/domain shift = the *outcome* SRC predicts; (b) covariate-shift
  perturbations (SSP) = synthetic whole-image corruptions, *corroborating only, NOT a contribution*;
  (c) counterfactual interventions (CSA) = the *method*. Never call (c) "data shift"; never claim (b) as novel.
- **4.9 Statistics.** Bootstrap CIs (1000×), Friedman, McNemar + Holm–Bonferroni.

---

## 4z. UNIFIED PIPELINE DESIGN (Stage 1 — the single coherent flow)

**Focus decision:** the paper has ONE spine, locked **after the C3 verdict** (see §8b). The pipeline below is
**spine-agnostic** — the same components/outputs are produced either way; only the *framing and which figure is
Fig 1* change. So we build once, then frame once.

- **Spine A (if C3 strong):** "An SRC certificate that *predicts* cross-source failure + miscalibration."
  Marquee = Fig 4 (coupling). Method is the headline; applied audit is supporting proof.
- **Spine B (if C3 weak):** "Published 96–99% CXR models rely on shortcuts and collapse cross-source — a
  validated causal audit." Marquee = Table A2 + Fig 2 (audit + effects). Audit is the headline; SRC is the tool.

**Pipeline stages → code module → emitted artifacts** (this is also the Stage-3 output checklist):

| # | Stage | Module (✅built / ⛔todo) | Emits (files the small-data test must produce) |
|---|---|---|---|
| P0 | Data → manifest | `src/data` ✅, `scripts/build_manifest.py` ✅ | `data/manifest.csv` |
| P1 | Train zoo | `scripts/train.py` ✅, `src/models/zoo.py` ✅, `trainer.py` ✅ | `results/<m>/<m>_best.keras`, `_history.json` |
| P2 | In-domain eval + calibration | `scripts/evaluate.py` ✅, `metrics.py` ✅ | `results/<m>/metrics.json`, `table_a_in_domain.csv` |
| P3 | CSA audit + SRC | `csa.py` ✅, `src_certificate.py` ✅ | `results/<m>/certificate.json` |
| P4 | Cross-source + C3 coupling | `cross_domain.py` ✅ | `results/cross_source.json`, `c3_coupling.json` |
| P5 | XAI corroboration (C5) | `src/xai/explain.py` ✅ | `results/<m>/xai.json` |
| P6 | Robustness/SSP | `src/robustness/perturbations.py` ✅ | `results/<m>/robustness.json` |
| P7 | Abstention curves (C5) | `src/shortcut/abstention.py` ✅ | `results/<m>/abstention.json` |
| P8 | Figures + LaTeX tables | `src/reporting/figures.py` ✅ | `results/figures/*.png`, `results/tables/*.tex` |
| P9 | Stats (Friedman/McNemar/Holm) | `metrics.py` ✅ (extended) | `results/stats_summary.json` |

**Build status (verified — all modules implemented, 0 `NotImplementedError`):** P0–P9 are real code and have
been run **end-to-end on a tiny real subset** (Stage-3 local test passed: every artifact above was produced).
Orchestrator `scripts/run_pipeline.py` ✅ chains P1→P9 with `--models`/`--stages`, with live streamed progress;
the local smoke test and both Colab notebooks (`colab_small.ipynb`, `colab_full.ipynb`) call this SAME entrypoint
(one codebase, three configs). **What remains is not coding but EXECUTION on Colab** (real training → real
numbers). All results are still **[PLANNED]** until the Colab runs complete (only lenet5 has a real checkpoint).

**Component scope (locked — no scope creep):** 7 models · 4 classes · 3 shortcut channels + 2 controls ·
2 XAI methods (Grad-CAM + Integrated Gradients) · 7 SSP perturbations · the figures/tables in §7 ONLY.
Anything not in this table is OUT (no 30-model zoo, no 7-XAI suite, no t-SNE/ensemble — see XAI rationale §4.5b).

**Reviewer-hardening additions ADOPTED (analyses on existing artifacts, NOT new contributions):**
§4.6a baselines · §4.6b LOMO · §4.6c post-TS ECE · §4.6d seeds+bootstrap (compute-gated) · Fig 8 failure panel ·
S1 intervention-strength (supplement). **REJECTED (do not add):** source-identity classifier (already proven by
DeGrave + dataset-bias lit; re-treads our closest competitor, dilutes novelty — cite instead, don't rebuild);
mixed-effects/hierarchical regression and per-class×source breakdown (over-engineering at n≈21; reads as fishing).

---

## 5. Experiments & Results  *(all numbers below are [PLANNED] — only lenet5 has run so far)*

- 5.1 In-domain reproduction (pooled high-accuracy baseline) — **Table A**.
- 5.2 CSA per-channel causal effects + **negative-control validation** — **Fig 2 + Table B**.
- 5.3 SRC per model (+ per-channel breakdown + CI) — **Table C / Fig 3**.
- 5.4 **C3 coupling (marquee):** SRC ↔ cross-source collapse & ECE — **Fig 4**.
- 5.5 **C4 applied audit:** benchmark-style models under source-labeled protocol — **Table A2**.
- 5.6 XAI vs SRC — **Fig 5 + Table D**. 5.7 Certificate-gated abstention — **Fig 6**.
- 5.8 Robustness/SSP corroboration — **Fig 7**. 5.9 Statistical-significance summary.

---

## 6. Datasets (verified against `configs/datasets.yaml`)

| Disease      | In-domain source       | Cross-source test       | Masks                        | Kaggle slug (verify at download)                                         |
| ------------ | ---------------------- | ----------------------- | ---------------------------- | ------------------------------------------------------------------------ |
| COVID-19     | COVID19-Radiography-DB | COVID-QU-Ex             | Yes                          | tawsifurrahman/covid19-radiography-database → anasmohammedtahir/covidqu |
| Normal       | COVID19-Radiography-DB | RSNA (optional, ~11 GB) | Partial                      | (above) → iamtapendu/rsna-pneumonia-processed-dataset                   |
| Pneumonia    | Kermany-Pediatric      | RSNA Pneumonia          | U-Net (Kermany has none)     | paultimothymooney/chest-xray-pneumonia → RSNA                           |
| Tuberculosis | Shenzhen               | Montgomery              | Yes (Montgomery ships masks) | kmader/pulmonary-chest-xray-abnormalities (split by folder in code)      |

Record license + version + access date (TRIPOD-AI/CLAIM). Masks define the CSA "background" channel.
**Overlap with benchmarks:** COVID-QU-Ex + Shenzhen/Montgomery are the exact sources of B5 and B6 → direct
"same data, we go further" comparison.

---

## 7. Figures & Tables

**Tables:** A in-domain baseline; A2 applied-audit (benchmark-style models, cross-source); B per-channel
effects + controls; C SRC; D XAI vs SRC; **E SRC vs. baseline predictors (R² head-to-head, §4.6a) + LOMO
predicted-vs-actual (§4.6b)**.
**Figures:** 1 CSA schematic; 2 per-channel effects + negative controls; 3 SRC bars; 4 **SRC↔collapse/ECE
(marquee, with post-TS ECE overlaid, §4.6c)**; 5 XAI vs SRC; 6 accuracy/coverage; 7 SSP heatmap;
**8 failure-case qualitative panel** (in-domain high-conf-correct · cross-source high-conf-wrong · high-SRC vs
low-SRC image · CSA intervention flips prediction · sham does not — makes the validity controls visible).
**Supplement:** S1 intervention-strength sensitivity (sweep border fraction; SRC model-rank stable by Spearman, §4.4 cross-ref).

---

## 8. Limitations (soundness venues reward honesty)

Chosen (non-exhaustive) shortcut channels; segmentation quality bounds the background channel; single
modality (CXR); retrospective public data; GPU non-determinism minimized not eliminated; SRC validated on the
datasets/models here — external re-validation needed before clinical claims. **Preprocessing interacts with
the audit:** heavy image enhancement can mask shortcut signals, so CSA is run on minimally preprocessed inputs
and preprocessing is studied as an ablation rather than baked in (§4.2b).

---

## 8b. ⚠️ CRITICAL-PATH RISK (read before sequencing work)

**C3 (SRC → cross-source ΔAcc AND ECE) is the single point of failure.** C1/C2/C4/C5 are mechanical once
data + models exist (and C1+C2 are already coded; data + 53k-row manifest already built). But the paper's
entire payoff is the coupling. **De-risk it FIRST, not last:**

- After even 2–3 models are trained, run a *preliminary* C3 regression. If SRC shows NO relationship to
  collapse/ECE, STOP and rethink the SRC definition before training all 7 — do not discover this at the end.
- Honest fallback if coupling is weak: the paper becomes "a validated shortcut-audit instrument + descriptive
  cross-source findings" (still publishable at a soundness venue), but the marquee claim must then be dropped.
  Decide the threshold for "coupling holds" (e.g. R² and CI) BEFORE looking at the result, to avoid p-hacking.

**Current build state (verified — all code complete):** data ✅ | manifest ✅ (53,625 imgs) |
P0–P9 all implemented ✅ (CSA, SRC, cross_domain/C3, XAI, robustness, abstention, stats, reporting,
orchestrator) | Stage-3 local end-to-end test PASSED | **models: 1/7 trained (lenet5 smoke checkpoint only)**.
The remaining risk is therefore purely the **scientific outcome of C3 on real Colab runs**, not any missing code.

---

## 9. Next Actions (after lock)

> **Coding is COMPLETE** (P0–P9 implemented, 0 `NotImplementedError`, Stage-3 local test passed — see §8b).
> What remains is **execution → validation → results → manuscript**, NOT building. Every component below already
> exists and is run via the single orchestrator `scripts/run_pipeline.py` (Colab notebooks call the same entry).
> All real numbers stay **[PLANNED]** until the corresponding Colab run completes.

1. **Run the 3-model go/no-go (`colab_small.ipynb`).** lenet5 + two representative transfer-learning models
   (densenet201, resnet50/vit); the full pipeline emits SRC, cross-source ΔAcc, ECE, **post-temperature-scaling
   ECE (§4.6c)**, and the preliminary **C3 regression + baselines (§4.6a)**. Decide the "coupling holds"
   threshold (R² and CI) **before** looking (§8b, no p-hacking).
2. **If C3 is promising → full 7-model run (`colab_full.ipynb`).** All artifacts at full epochs; adds
   **LOMO out-of-sample prediction (§4.6b)** to substantiate the word "predicts."
3. **If the 7-model C3 holds → compute-gated 3-seed extension (§4.6d).** 7 models × 3 seeds = 21 instances;
   report bootstrap CI on the coupling slope. (Skip if C3 is weak — do not spend the compute.)
4. **Generate final artifacts** (all from existing reporting code): Tables A, A2, B, C, D, **E**; Figures 1–7 +
   **Fig 8 failure-case panel**; Supplement **S1** intervention-strength sensitivity.
5. **Write the manuscript around the spine the C3 verdict selects** (framing only — no code changes; see §4z):
   - **Spine A (C3 strong):** SRC *predicts* cross-source failure and miscalibration. Marquee = Fig 4.
   - **Spine B (C3 weak):** a validated causal shortcut-audit instrument with descriptive cross-source findings.
     Marquee = Table A2 + Fig 2. The "predicts" claim is dropped (§8b fallback).
6. **Complete the reproducibility package:** dataset access dates + licenses (TRIPOD-AI/CLAIM), fixed seeds/configs
   (seed 42 in both configs), released code, and the TRIPOD-AI / CLAIM checklist (see §10).

---

## 10. Soundness Checklist (keep green before submission)

- [ ] Negative controls show null effect (CSA validity proven)
- [ ] Every result has CI + statistical test
- [ ] All 7 models × all datasets reported (no cherry-picking)
- [ ] Public datasets + licenses recorded; Kaggle slugs re-verified at download
- [ ] Seeds/configs fixed (seed 42 in both configs); code release-ready
- [ ] Coupling (C3) reported honestly even if modest
- [ ] DeGrave-delta framing kept honest (no "first causal CXR intervention" claim)
- [ ] CSA run on minimally preprocessed images (heavy enhancement studied as ablation, not baked in — §4.2b)
- [ ] XAI corroboration uses 2 independent-family methods + quantitative faithfulness (not eyeballed heatmaps)
- [ ] In-lung saliency fraction reported as independent SRC cross-check (framed as corroboration, not novelty)
- [ ] Limitations stated plainly; TRIPOD-AI / CLAIM checklist attached
- [ ] ⚠️ Resolve open items: confirm each cited paper's current Q1 quartile
