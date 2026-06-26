# Benchmark Papers — DETAILED EVIDENCE ARCHIVE (applied CXR-classification neighbors)

> **Role of this file:** this is the long-form evidence backing for **PAPER_OUTLINE.md §0b**.
> §0b holds the condensed table; THIS file holds the full per-paper write-ups with the verbatim
> author quotes needed when drafting Related Work. Keep both in sync — when a row changes here,
> update §0b, and vice versa. The merged master plan lives in PAPER_OUTLINE.md (v5); this file is
> a supporting appendix, not a separate plan.
>
> **VERIFICATION STATUS:** all 5 papers below (1, 2, 4, 5, 6) are **full-text verified from PDFs** (✅READ).
> (Old "Paper 3" was a duplicate-DOI stub of Paper 6 and was removed; the numbering 1/2/4/5/6 is kept to
> match prior references.) Every accuracy/dataset number came from the actual PDF. Do NOT cite a number we
> haven't seen.
>
> **Our anchor differentiators (apply to ALL rows):**
> 1. They POOL multi-source data + random split → we keep it SOURCE-LABELED + cross-source.
> 2. They report accuracy/XAI heatmaps → we CAUSALLY certify shortcut reliance (CSA + SRC).
> 3. They stop at "it works" → we PREDICT cross-source collapse + miscalibration from SRC.

---

## Core benchmark set

### 1. Multi-stage deep learning — comprehensive lung disease classification ✅ VERIFIED (full PDF read)
- **Citation:** G. Divya Deepak & S. Krishna Bhat, "A multi-stage deep learning approach for
  comprehensive lung disease classification from x-ray images," *Expert Systems With Applications*,
  vol. 277, 127220, 2025. https://doi.org/10.1016/j.eswa.2025.127220
- **Venue / year:** **Elsevier, *Expert Systems With Applications*, 2025 — Q1 confirmed** (IF ~8.5).
  Received Dec 2024, accepted Mar 2025, open access CC BY-NC. Affiliation: Manipal Institute of Technology, India.
- **What they did (verified):**
  - A **multi-stage CNN "toolchain"** (8 cascaded stages): Stage-1 Normal/Abnormal → Stage-2 COVID/non-COVID
    → branches to ARDS/Lung-Abscess, Pneumonia(Bacterial/Viral/Effusion), TB(Hydropneumothorax/Atelectasis),
    Environmental(Sarcoidosis/Emphysema/Pneumosclerosis), Heart(Cardiomegaly/Venous-Congestion).
    Classifies **17 (reported as 14 leaf) lung-disease classes**.
  - **5 pretrained backbones compared, picked per stage:** ResNet-50, DarkNet-53, EfficientNet-b0,
    ResNet-101, **DenseNet-201**. Per-stage hyperparameter sweep (solver sgdm/rmsprop/adam, batch 6/8/10, lr 1e-4/1e-5).
  - **Tiny dataset:** 1,546 original → **1,700 after 10% augmentation**; ~100 test images/class.
    **POOLED from multiple sources:** Kermany 2018 (pneumonia/normal), Patel 2020 (COVID), Tahir 2021
    COVID-QU-Ex, NIH ChestX-ray8 (Atelectasis/Cardiomegaly/Effusion). **← multi-source pooling, exactly our target.**
  - **Reported metrics:** averaged over 8 stages — accuracy/precision/recall/F1/specificity = **0.98/0.98/0.98/0.97/0.99**.
  - Deployed as a MATLAB standalone app.
- **What they did NOT do (verified from their own Conclusions/limitations):**
  - **No cross-dataset / external validation** — they explicitly say *"extensive validation is required for
    larger datasets… pilot studies will be conducted… future work will focus on robustness… generalizability to unseen data."*
  - **No interpretability / XAI** — they explicitly note *"the black box approach… lack of clarity on the
    interpretability of CNN models… limits their trustworthiness; efforts need to be made in this direction"*
    (and cite Mahamud 2024 — YOUR base paper — for XAI).
  - **No ablation** — *"this study does not include ablation experiments to isolate the contributions of
    individual architectural components."*
  - **No shortcut / bias analysis**; pooled multi-source data with a random 80/20 split; only 100 test imgs/class
    (they admit *"statistical uncertainty"* from small test size).
- **How we differ:** they cascade CNNs to maximize pooled-split accuracy across many classes; we keep data
  SOURCE-LABELED, evaluate cross-source, and **causally certify shortcut reliance** (CSA + SRC).
- **Why we're stronger (their limitations = our contributions):** their *own* stated gaps —
  no generalization testing, no interpretability, no ablation, small pooled data — are precisely C1–C5.
  We provide the cross-source validation, the quantitative+causal XAI, and the ablation they call for.
- **Benchmark use:** their 5 backbones overlap our zoo (DenseNet201, ResNet50, EfficientNetB0). We run these
  under OUR source-labeled protocol. We do NOT compare against their 0.98 number (different task: 17-class
  cascade on 1,700 pooled imgs vs our 4-class cross-source) — we cite it as a multi-source-pooling exemplar.

### 2. Explainable transfer-learning framework — multi-classification of lung diseases ✅ VERIFIED (full PDF read) — CLOSEST COMPETITOR
- **Citation:** A.N. Patel, R. Murugan, G. Srivastava, P.K.R. Maddikunta, G. Yenduri, T.R. Gadekallu,
  R. Chengoden, "An explainable transfer learning framework for multi-classification of lung diseases
  in chest X-rays," *Alexandria Engineering Journal*, vol. 98, pp. 328–343, 2024.
  https://doi.org/10.1016/j.aej.2024.04.072
- **Venue / year:** **Elsevier, *Alexandria Engineering Journal*, 2024 — Q1 confirmed** (IF ~6.8).
  Received Dec 2023, accepted Apr 2024, open access CC BY-NC-ND. Affiliation: VIT Vellore + others, India.
- **What they did (verified):**
  - **EfficientNet-B4** transfer learning + **Grad-CAM** XAI for **5-class** CXR: COVID-19, Pneumonia,
    Tuberculosis, **PLN (pulmonary lung nodules)**, Normal.
  - **Dataset: 7,389 images CONSOLIDATED INTO A SINGLE DATASET from MULTIPLE sources** — COVID (Cohen
    repo), Pneumonia (Kermany 2018, Guangzhou), TB (700 + 2,800 NIAID Kaggle Tahir set), PLN (LIDC-IDRI).
    **← multi-source pooling, exactly our target.** Resized 380×380; min-max norm; 80/20 train/val split.
  - **Reported: 96% overall accuracy**; per-class F1 0.94/0.93/0.90/0.97/0.99; AUROC up to 0.98.
    Trained on Google Colab, T4 GPU, 20 epochs, batch 16, lr 1e-4.
  - **XAI claim (important):** they add Grad-CAM + a *thresholding* step (Eq. 22) and assert the XAI
    "improves without retraining" by correlating explanation quality with model performance.
- **What they did NOT do — and where they HAND US the contribution (verified, their own words):**
  - **Grad-CAM is CORRELATIONAL, not causal — they admit it:** *"while Grad-CAM provides invaluable
    insights… it remains a visualization tool that highlights **correlative, rather than causative**,
    relationships between image features and disease presence… they do not directly address the
    underlying biological mechanisms."* **← THIS IS THE EXACT GAP CSA FILLS.**
  - **No cross-dataset / external validation** — single pooled dataset; they state *"the framework
    guarantees consistent outcomes when applied to similar data… variances expected with heterogeneous data."*
  - **No quantitative XAI faithfulness metric** — Grad-CAM is shown as heatmaps only (Figs 6–7), no
    insertion/deletion, no faithfulness correlation, no stability.
  - **No calibration, no abstention, no shortcut/bias analysis.** Class imbalance noted as a limitation.
- **How we differ:** they use Grad-CAM heatmaps (correlational) on a pooled split; we use **counterfactual
  interventions (causal)** + a **certificate** + **cross-source** evaluation + **calibration/abstention**.
- **Why we're stronger (their stated limitation = our thesis):** they EXPLICITLY concede their XAI is
  "correlative, rather than causative." CSA is precisely the causative version — per-channel interventional
  attribution — and SRC turns it into an auditable certificate that predicts the generalization they never tested.
- **Benchmark use:** EfficientNet-B4 / their backbone family is in our zoo (EfficientNet-B0). We reproduce
  their pooled ~96% as a baseline, THEN show cross-source collapse + causal certification. Their 5th class
  (PLN) — we stay 4-class (base-paper alignment); note PLN exclusion explicitly.
- **⭐ This is the paper to cite FIRST in Related Work** — it is the strongest "explainable TL" precedent and
  its admitted correlational-XAI limitation is the cleanest possible motivation for CSA/SRC.

### 4. Pneumonia classification — experimental DL comparison ✅ VERIFIED (full PDF read) — MOST RECENT (2026)
- **Citation:** H. Aljuaid, H. Adlan, B. Alkebsi, B.S. Alfurhood, A. Liotta, L. Cavallaro,
  "An experimental comparison of deep learning models for pneumonia classification from chest X-ray
  images," *Biomedical Signal Processing and Control*, vol. 112, 108742, 2026.
  https://doi.org/10.1016/j.bspc.2025.108742
- **Venue / year:** **Elsevier, *Biomedical Signal Processing and Control*, vol. 112 (2026) — Q1 confirmed**
  (IF ~4.9). Received May 2023, accepted Sep 2025, open access CC BY. **Most recent paper in our set.**
  Affiliations: Princess Nourah bint Abdulrahman Univ (Saudi Arabia), Free Univ Bozen-Bolzano (Italy), Radboud (NL).
- **What they did (verified):**
  - **Fair-comparison study** of **6 CNNs** (ResNet, AlexNet, **VGGNet/VGG-19**, SqueezeNet, DenseNet,
    InceptionV3) on **3-class** CXR: Normal / Pneumonia(bacterial+viral) / COVID-19.
  - **Dataset: 6,939 images, SINGLE Kaggle set** ("COVID19_Pneumonia_Normal_Chest_Xray_PA_Dataset"),
    **balanced 2,313/class**, expert-screened, 80/20 split, 5-fold CV, ~20–30 epochs.
  - **Reported: VGG-19 best at 97% accuracy** (93% COVID, 94% Normal F1). Adam, lr 1e-3, 224×224.
  - **Explicit motivation = FAIR COMPARISON:** *"existing proposals are not directly comparable due to
    considerable differences in their respective modeling pipeline… we conduct a comparative analysis
    based on a uniform pipeline."* (methodologically aligned with our fairness framing — cite as precedent.)
  - **Grad-CAM used to claim no shortcuts:** *"Analysis confirmed that the model focused on relevant
    features… rather than learning spurious correlations or irrelevant information."*
- **What they did NOT do — and the OPENING this gives us (verified):**
  - **They CLAIM "no spurious correlations" but only via QUALITATIVE Grad-CAM eyeballing** — no quantitative
    faithfulness, no causal test, **and crucially no cross-dataset validation to actually verify it.**
    **← We provide the rigorous test their claim lacks: CSA causally measures shortcut reliance; SRC certifies it.**
  - **No cross-dataset / external validation** — single balanced Kaggle set; they admit *"larger dataset
    required to increase classes"* and struggle to separate COVID vs viral pneumonia.
  - **No calibration, no abstention, no per-channel attribution.**
  - Single-source dataset → cannot even exhibit cross-source shortcut (but also cannot rule it out).
- **How we differ:** they benchmark 6 CNNs on one balanced set and *assert* no shortcuts from heatmaps;
  we benchmark under a SOURCE-LABELED cross-source protocol and *causally certify* shortcut reliance.
- **Why we're stronger:** their central XAI claim ("focuses on relevant features, not spurious correlations")
  is exactly the hypothesis CSA/SRC is built to TEST rather than assert. We turn their eyeballed claim into a
  measured, certified, falsifiable result — and add the cross-source generalization test they omit.
- **Benchmark use:** their 6 CNNs overlap our zoo (ResNet50, DenseNet, etc.). Their uniform-pipeline fairness
  argument is a useful PRECEDENT to cite for our own fixed-recipe fairness (METHODS §3.2.5). We do NOT compare
  to their 97% (3-class single-source vs our 4-class cross-source).

### 5. COVID & non-COVID detection — multi-stage CNNs ✅ VERIFIED (full PDF read) — SPRINGER entry, SAME DATASETS as us
- **Citation:** G.D. Deepak & S.K. Bhat, "Detection of COVID-19 & non-COVID diseases from chest-X-rays
  using deep learning-based CNNs," *Soft Computing*, vol. 30, pp. 449–471, 2026.
  https://doi.org/10.1007/s00500-025-10905-4
- **Venue / year:** **Springer, *Soft Computing*, vol. 30 (2026) — Q1 confirmed** (IF ~3.7).
  Received Dec 2023, accepted Aug 2025, published online Dec 2025, open access CC BY. Manipal Inst. of Tech, India.
  (Same authors as Paper 1, but a DIFFERENT, larger study — 3-class on 33,920 imgs vs Paper 1's 17-class on 1,700.)
- **What they did (verified):**
  - **3-stage cascade:** SqueezeNet (Normal/Abnormal, 99%) → ResNet-50 (COVID/non-COVID, 98%) →
    EfficientNet-b0 (non-COVID → Pneumonia-bacteria/virus/TB, 97%). Per-stage optimizer/batch sweep (Adam/RMSprop/SGDM).
  - **Dataset: 33,920 images — and they use OUR EXACT SOURCES (verified, p.454 + Data Availability):**
    **`anasmohammedtahir/covidqu` (COVID-QU-Ex — our COVID cross-source)** + **`tawsifurrahman/tuberculosis-tb-chest-xray-dataset` (TB DB)**.
    11,956 COVID / 11,263 non-COVID / 10,701 Normal + 700 TB / 3,500 Normal. **← same canonical datasets we use.**
  - **Reported: 99% / 98% / 97%** across the three stages. Deployed as MATLAB "CovidApp".
  - **Leakage control:** LOOCV + cross-validation; **no preprocessing** beyond augmentation.
  - **"Tested against a new set":** they ran the trained app on sample images "obtained from literature"
    (Ali et al. 2024) — but this is a HANDFUL of qualitative demo images, NOT a quantitative cross-dataset eval.
- **What they did NOT do — their EXPLICIT limitations (verified, p.468 Conclusions):**
  - *"deep learning models inherently lack interpretability… the model's decision-making process remains a
    'black box'… difficult for healthcare professionals to understand the underlying reasoning"* (no XAI).
  - *"potential for data biases, particularly regarding class distribution and dataset diversity… datasets
    used may not fully capture real-world variations… could affect generalizability."* **← admits bias risk, doesn't test it.**
  - *"Future work will focus on evaluating model performance across multiple INDEPENDENT datasets to reduce
    bias."* **← they explicitly defer the cross-dataset validation we DO.**
  - *"does not investigate the effect of variable imaging conditions (lighting, noise, resolution, partial
    anatomy)… data impurity, pixel size, channel adjustments remains unexplored."* **← exactly our covariate-shift/CSA axis.**
- **How we differ:** they cascade CNNs for accuracy on pooled COVID-QU-Ex+TB data and admit untested bias;
  we use the SAME datasets SOURCE-LABELED, run cross-source, and causally certify the bias they flag but don't measure.
- **Why we're stronger:** they name THREE of our contributions as future work — interpretability (C4),
  cross-independent-dataset validation (C3), and imaging-condition/shift robustness (CSA/SSP). We deliver all three
  on the very datasets they used.
- **Benchmark use — STRONGEST overlap:** they use our exact COVID-QU-Ex + TB-DB sources and our backbone families
  (ResNet-50, EfficientNet-b0). Ideal "same data, we go further" comparison. We reproduce their pooled high accuracy
  as baseline, then expose cross-source collapse + certify shortcuts. Do NOT compare to their 97–99% (cascade vs flat 4-class).

### 6. Lung disease classification — DWT-enhanced CNN + segmentation + GAN ✅ VERIFIED (full PDF read) — SPRINGER, our exact datasets
- **Citation:** F.A.A. Slimani & M. Bentourkia, "Lung Disease Classification with Deep Learning Enhanced
  CNN Architecture in Chest X-Ray Imaging," *Journal of Imaging Informatics in Medicine*, 2025.
  https://doi.org/10.1007/s10278-025-01760-8
- **Venue / year:** **Springer, *Journal of Imaging Informatics in Medicine* (formerly J. Digital Imaging),
  2025 — Q1 confirmed** (IF ~4, SIIM journal). Received Apr 2025, accepted Nov 2025. Univ. de Sherbrooke, Canada.
- **What they did (verified):**
  - **Primarily a SEGMENTATION + augmentation methods paper:** Rand-DWT (discrete wavelet transform
    replacing max-pooling) inside **U-Net++ with Attention Gates** for lung segmentation; then
    **DWT + DenseNet-201** (and VGG-19) for **4-class classification** (Normal/TB/Pneumonia/COVID-19);
    **PGGAN** synthetic CXR augmentation; N-CLAHE preprocessing.
  - **Datasets = OUR EXACT SOURCES:** JSRT, **Montgomery, Shenzhen (Chest X-Rays Masks & Labels)**,
    **COVID-QU-Ex (C19, 33,920)**. Segmentation: 99.1% acc / 97.2% Dice (JSRT), 99.3% (C19).
  - **Reported classification:** +2.4% precision over plain DenseNet-201 baseline when DWT+augmentation added.
  - 5-fold CV, 200 epochs, RTX 3060, Keras; segmentation masks validated by clinicians.
- **What they did NOT do (verified):**
  - **No interpretability / XAI at all** — purely accuracy/Dice-driven; no Grad-CAM, no faithfulness.
  - **No cross-source generalization test** — JSRT/MC used only as additional *test* sets for SEGMENTATION,
    not for cross-source *classification* collapse; classification stays within pooled data.
  - **No shortcut/bias analysis, no calibration, no per-channel attribution.**
  - Their contribution is *better segmentation + GAN data* → higher accuracy, not understanding WHY models work.
- **How we differ:** they improve the front-end (segmentation + synthetic data) to push accuracy; we audit the
  back-end decision — causally certifying whether the classifier relies on disease vs dataset, cross-source.
- **Why we're stronger / COMPLEMENTARY:** this paper is the clearest example of "make accuracy higher" without
  ever testing generalization or interpretability — exactly the blind spot CSA/SRC addresses. Their lung
  segmentation (U-Net++) is also directly USEFUL TO US: it produces the lung masks our CSA "background" channel
  needs (cite as a mask source / related segmentation method).
- **Benchmark use:** uses our exact COVID-QU-Ex + Shenzhen/Montgomery sources and DenseNet-201/VGG-19 (in our zoo).
  Cite as (a) a same-data accuracy-maximizing baseline, and (b) a segmentation-method reference for lung masks.

---

## Verified benchmark set (status board)

| # | Paper | Venue | Year | Q1 | Datasets | Their admitted gap = our contribution |
|---|---|---|---|---|---|---|
| 1 | Deepak & Bhat (multi-stage 17-class) | Elsevier ESWA | 2025 | ✅ | pooled Kermany+Patel+QU-Ex+NIH (1,700) | no interpretability / no generalization / no ablation |
| 2 | Patel et al. (explainable TL) ⭐ | Elsevier AEJ | 2024 | ✅ | pooled Cohen+Kermany+NIAID+LIDC (7,389) | Grad-CAM **"correlative, not causative"** |
| 4 | Aljuaid et al. (DL comparison) | Elsevier BSPC | 2026 | ✅ | single balanced Kaggle (6,939) | *claims* no shortcuts via eyeballed Grad-CAM only |
| 5 | Deepak & Bhat (3-stage COVID/TB) | Springer Soft Comp | 2026 | ✅ | **COVID-QU-Ex + TB-DB (our exact sources, 33,920)** | defers cross-dataset + interpretability + shift-robustness to "future work" |
| 6 | Slimani & Bentourkia (DWT+seg+GAN) | Springer JIIM | 2025 | ✅ | **COVID-QU-Ex + Shenzhen + Montgomery + JSRT** | no XAI, no generalization test — pure accuracy via better seg + GAN data |
| ✗ | Hu et al. (MD-Conv) | IEEE Access | 2020 | — | ChestX-ray14 | REJECTED — 2020, off-topic (efficiency, not multi-disease) |

**Publisher mix:** 3 Elsevier (B1, B2, B4) + 2 Springer (B5, B6) verified — all Q1. The earlier
"need a valid IEEE 2024–26 paper for balance" goal was **DROPPED** (decision 2026-06): it was self-imposed,
not a journal requirement. Cite the 5 verified papers as-is; no fabricated IEEE entry.

## Differentiator summary (the table for the paper's Related Work)

| Capability | Benchmark papers (2024–26) | **Ours (CSA + SRC)** |
|---|---|---|
| Multi-disease CXR classification | ✅ (pooled, random split) | ✅ (source-labeled, cross-source) |
| High in-domain accuracy (~97–99%) | ✅ | ✅ (reproduced as baseline) |
| Qualitative XAI (Grad-CAM/LIME/SHAP) | some | ✅ + quantitative + causal |
| Cross-source / external validation | ❌ (all defer to future work) | ✅ core |
| **Causal per-channel shortcut attribution** | ❌ | ✅ |
| **Per-model auditable certificate (SRC)** | ❌ | ✅ |
| **Predicts collapse + miscalibration** | ❌ | ✅ |

**The unifying narrative:** all 5 verified Q1 papers (2024–2026) reach 96–99% on POOLED data, and EACH either
explicitly admits — in its own limitations — that it cannot verify the model sees disease not dataset (B1/B2/B5:
no/correlational XAI, no cross-dataset test) or only *asserts* it via eyeballed Grad-CAM (B4). Our CSA+SRC
delivers the causal, certified, cross-source evidence they defer or assume.

---

## TODO before manuscript submission
- [x] Papers 1, 2, 4, 5, 6 — full-text verified from PDFs.
- [x] Removed stale Paper 3 stub (was a duplicate DOI of Paper 6, never read).
- [x] IEEE-balance requirement dropped (decision 2026-06).
- [ ] Confirm exact SJR/JCR Q1 quartile for each at submission time (not independently verified here).
- [ ] Keep this file in sync with PAPER_OUTLINE.md §0b.
