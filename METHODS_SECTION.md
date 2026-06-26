# Methods

## 3.1 Dataset and Preprocessing

### 3.1.1 Dataset Description

We evaluate our pipeline on a publicly available Kaggle chest X-ray (CXR) dataset comprising approximately 28,000 posteroanterior radiograph images spanning four diagnostic categories: COVID-19 (*n* ≈ 3,616), Normal (*n* ≈ 10,192), Bacterial/Viral Pneumonia (*n* ≈ 10,616), and Tuberculosis (*n* ≈ 3,500). All images are resized to 224 × 224 × 3 pixels, the standard input resolution for ImageNet-pretrained convolutional backbones (He et al., 2016). Data is partitioned into training (70%), validation (15%), and test (15%) splits via stratified random sampling with a fixed random seed (*s* = 42), ensuring class proportions are preserved across all partitions. The identical test partition is reused for every model to guarantee paired-sample comparability.

**Leakage control.** We mitigate optimistic bias from duplicate or near-duplicate radiographs by performing duplicate detection (perceptual hashing) prior to data splitting and removing duplicates. When patient identifiers are available, splits are enforced at the patient level; otherwise, duplicate filtering is used to prevent identical images from appearing across training and evaluation partitions.

### 3.1.2 Image Preprocessing Pipeline

Each image undergoes a five-stage preprocessing pipeline following Mahamud et al. (2024):

1. **Denoising Autoencoder (DAE)**: A U-Net-style convolutional autoencoder with skip connections (noise factor σ = 0.15, 50 epochs, lr = 10⁻³) learns to reconstruct clean images from synthetically corrupted inputs, reducing scanner-specific noise artifacts. Skip connections between encoder and decoder layers preserve fine structural details during reconstruction. The DAE is trained only on the training split and then applied as a fixed operator to validation and test images.

2. **Bilateral Filtering** (*d* = 9, σ_color = 75, σ_space = 75): Edge-preserving smoothing that reduces remaining detector noise while maintaining diagnostically relevant boundaries (e.g., pulmonary infiltrates, consolidation margins).

3. **Morphological Opening** (elliptical kernel, 5 × 5): Removes small bright artifacts (e.g., annotations, electrode markings) without eroding larger pathological opacities.

4. **CLAHE** (clipLimit = 2.0, tileGrid = 8 × 8): Contrast-Limited Adaptive Histogram Equalization normalizes intensity distributions across heterogeneous scanner acquisitions, improving visibility of subtle opacities in low-contrast regions (Pizer et al., 1987).

5. **Gamma Correction** (γ = 1.2): Brightens under-exposed images to standardize overall intensity.

After preprocessing, pixel values are rescaled to [0, 1] by dividing by 255.

## 3.2 Training Protocol

### 3.2.1 Two-Phase Transfer Learning

All pretrained architectures follow a two-phase training protocol designed to leverage ImageNet representations while adapting to CXR-specific features:

**Phase 1 — Feature Extraction** (*E*/2 epochs): The pretrained convolutional base is frozen. Only the custom classification head (Global Average Pooling → Dense(512, ReLU) → Dropout(0.5) → Dense(256, ReLU) → Dropout(0.5) → Dense(4, Softmax)) is trained with Adam optimizer at a learning rate of η₁ = 10⁻⁴. This prevents catastrophic forgetting of general visual features during early gradient updates when the head parameters are randomly initialized.

**Phase 2 — Fine-Tuning** (*E*/2 epochs): All layers are unfrozen and trained end-to-end at a reduced learning rate η₂ = η₁ × 0.01 = 10⁻⁶. The 100× reduction prevents large gradient updates from destabilizing learned convolutional filters while permitting domain-specific adaptation (Tajbakhsh et al., 2016).

Custom architectures (LeNet) bypass Phase 1 and train from scratch for the full *E* epochs at η₁.

### 3.2.2 Training Configuration

All models share the following training recipe:

| Parameter | Value | Justification |
|---|---|---|
| Optimizer | Adam | Adaptive learning rates (Kingma & Ba, 2015) |
| Loss | Categorical cross-entropy | Standard multi-class objective |
| Epochs (*E*) | 100 (Colab) / 20 (local) | With early stopping |
| Batch size | 64 (Colab) / 8 (local) | Memory-constrained |
| Early stopping | patience = 10, monitor = val_loss | Prevents overfitting |
| LR scheduling | ReduceLROnPlateau(factor=0.5, patience=5, min_lr=10⁻⁸) | Adaptive convergence |
| Class weights | Inverse-frequency balanced | Addresses class imbalance |
| Checkpoint | Best val_accuracy restored | Avoids over-trained checkpoints |

### 3.2.3 Reproducibility and Determinism

All experiments are seeded for reproducibility:

- **Python random**: `random.seed(42)`
- **NumPy**: `np.random.seed(42)`
- **TensorFlow**: `tf.random.set_seed(42)`
- **Hash determinism**: `PYTHONHASHSEED=42`
- **CUDA deterministic operations**: `TF_DETERMINISTIC_OPS=1` (TensorFlow ≥ 2.8)

Hardware and software versions are recorded in the experiment log (TensorFlow version, CUDA version, GPU model, Python version). We note that full bitwise reproducibility on GPU is not guaranteed due to non-deterministic cuDNN algorithms; however, the above measures minimize run-to-run variance.

### 3.2.4 Clinically Constrained Data Augmentation

Augmentation parameters are specifically constrained for chest radiograph validity:

| Transform | Range | Clinical Rationale |
|---|---|---|
| Rotation | ±15° | Maximum realistic patient tilt on cassette; >20° produces anatomically implausible CXR geometry |
| Horizontal flip | Yes | CXR exhibits approximate bilateral thoracic symmetry |
| Width/Height shift | ±10% | Simulates patient positioning variation within the X-ray field |
| Shear | 0.05 | Greatly reduced; there is no physiological basis for shearing in chest radiography |
| Zoom | ±15% | Simulates source-to-image distance (SID) variation; >20% risks cropping peripheral pathology |
| Brightness | [0.85, 1.15] | Simulates exposure calibration differences across scanners/sites |
| Fill mode | Constant (black, cval=0) | Regions outside the field of view in CXR are unexposed (black), not mirrored or nearest-neighbor extended |

This design ensures that every augmented image remains a plausible radiograph that a radiologist could encounter, avoiding the introduction of synthetic artifacts that could bias learned representations (Castro et al., 2020).

### 3.2.5 Fairness of Comparison

To ensure fair multi-architecture benchmarking (Bouthillier et al., 2021):

1. **Fixed training recipe**: All 30 architectures use identical hyperparameters (optimizer, learning rate schedule, augmentation, early stopping, class weighting). No per-model hyperparameter tuning was performed.

2. **Identical input pipeline**: All models receive the same 224 × 224 × 3 input resolution, the same preprocessing chain, the same train/val/test split, and the same augmentation policy.

3. **Same evaluation protocol**: All models are evaluated on the identical held-out test partition using the identical metric suite, eliminating split-induced performance variance.

4. **Comparable training budget**: All models train for the same maximum epoch count with the same early stopping policy. The only efficiency difference is architectural — some backbones converge faster due to their inductive biases.

**Known exception**: Vision Transformer (ViT) is a patch-based architecture that may benefit from longer warmup schedules and higher resolution (Dosovitskiy et al., 2021). In our protocol, ViT receives the identical training recipe as CNN baselines. We document this as a potential performance lower-bound for ViT, not a limitation for the remaining 29 models.

## 3.3 Evaluation Framework

### 3.3.1 Standard Classification Metrics

For each model *M_k*, we report:

- **Accuracy**: Overall fraction of correct predictions.
- **Precision / Recall / F1-Score**: Computed per-class and then aggregated via both weighted (accounts for class imbalance) and macro (equal class weight) averaging.
- **Specificity**: Per-class true-negative rate, computed from the confusion matrix.
- **Cohen's Kappa** (κ): Agreement beyond chance; κ > 0.81 indicates near-perfect agreement (Landis & Koch, 1977).
- **Matthews Correlation Coefficient** (MCC): Balanced measure valid even under class imbalance (Chicco & Jurman, 2020).
- **ROC-AUC**: One-vs-rest AUC per class, plus macro-averaged AUC.
- **PR-AUC** (Average Precision): Preferred over ROC-AUC for imbalanced classes (Saito & Rehmsmeier, 2015).
- **Bootstrap 95% Confidence Intervals**: 1,000 bootstrap iterations on test-set accuracy.

### 3.3.2 Calibration Analysis

Well-calibrated probability estimates are critical for clinical decision support (Jiang et al., 2012). We compute:

**Expected Calibration Error (ECE):**

$$\text{ECE} = \sum_{b=1}^{B} \frac{|S_b|}{N} \left| \text{acc}(S_b) - \text{conf}(S_b) \right|$$

where *S_b* is the set of predictions falling in confidence bin *b*, *B* = 15 bins, *N* is the total sample count, acc(*S_b*) is the empirical accuracy, and conf(*S_b*) is the mean confidence within the bin (Naeini et al., 2015). ECE < 0.05 indicates clinically acceptable calibration.

**Brier Score:**

$$\text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} \left( p_{ic} - y_{ic} \right)^2$$

where *p_ic* is the predicted probability and *y_ic* is the one-hot ground truth for sample *i* and class *c*. Lower is better (Brier, 1950).

**Negative Log-Likelihood (NLL):**

$$\text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \log p_{i, y_i^*}$$

where *y_i** is the true class of sample *i*. NLL penalizes confident incorrect predictions more heavily than ECE.

**Post-hoc Temperature Scaling (Guo et al., 2017):** Temperature scaling is fit on the validation split by minimizing NLL and evaluated on the held-out test set, preventing test-set leakage. We learn a scalar temperature *T* ∈ [0.1, 10] via bounded scalar optimization:

$$\hat{q}_c = \frac{\exp(z_c / T)}{\sum_{j} \exp(z_j / T)}$$

where *z_c* are the pre-softmax logits. The optimal *T* is reported alongside the post-calibration ECE.

**Reliability Diagrams**: We visualize calibration via (a) per-class calibration curves (predicted probability vs. fraction of positives) and (b) an overall confidence histogram overlaid with the accuracy-confidence gap, annotated with the ECE value.

### 3.3.3 Threshold Analysis

Standard classification uses argmax (i.e., highest predicted probability). In clinical settings, sensitivity and specificity tradeoffs are disease-specific. For each class *c*, we sweep over 50 decision thresholds *t* ∈ [0.05, 0.95] and compute:

- **Sensitivity**: Se(*t*) = TP / (TP + FN)
- **Specificity**: Sp(*t*) = TN / (TN + FP)
- **Positive Predictive Value**: PPV(*t*) = TP / (TP + FP)
- **Negative Predictive Value**: NPV(*t*) = TN / (TN + FN)

The optimal clinical operating point is selected via **Youden's J statistic** (Youden, 1950):

$$J(t) = \text{Se}(t) + \text{Sp}(t) - 1$$

$$t^* = \arg\max_t J(t)$$

This is reported alongside the achieved sensitivity, specificity, PPV, and NPV at *t**, giving clinicians a per-disease decision rule that balances false-positive and false-negative costs.

### 3.3.4 Statistical Testing

With 30 candidate models, rigorous statistical comparison is essential:

**Friedman Test**: A non-parametric omnibus test for *K* ≥ 3 classifiers evaluated on the same test set (Demšar, 2006). We treat each sample's correctness (binary) as a "block" and compare ranks across models. A significant Friedman statistic (p < 0.05) indicates that at least one model differs significantly.

**Pairwise McNemar's Test**: For each model pair (*M_i*, *M_j*), McNemar's test (with continuity correction) compares the discordant cells of the 2×2 contingency table of correct/incorrect predictions (McNemar, 1947):

$$\chi^2 = \frac{(|b - c| - 1)^2}{b + c}$$

where *b* = samples correctly classified only by *M_j* and *c* = samples correctly classified only by *M_i*.

**Multiple Comparison Correction**: With $\binom{30}{2} = 435$ pairwise tests, uncorrected p-values inflate the family-wise type I error. We apply the **Holm–Bonferroni** step-down procedure (Holm, 1979), which is uniformly more powerful than the classical Bonferroni correction while maintaining the familywise error rate at α = 0.05. We report both uncorrected and corrected significance counts.

## 3.4 Robustness Under Covariate Shift

### 3.4.1 Motivation

Clinical deployment exposes models to distribution shifts absent from the training set — different scanners, compression artifacts, patient positioning errors, and post-processing pipelines (Zech et al., 2018). We systematically evaluate robustness by applying test-time perturbations following the methodology of Hendrycks & Dietterich (2019).

### 3.4.2 Shift Operators

We define seven perturbation families $\{\mathcal{P}_k\}_{k=1}^{7}$, each mapping a clean image *x* to a corrupted version $\tilde{x} = \mathcal{P}_k(x; s)$ at severity level *s*:

| Shift Operator | Severity Levels | Clinical Analogue |
|---|---|---|
| Gaussian noise | σ ∈ {0.01, 0.03, 0.05, 0.08} | Detector electronic noise / low-dose imaging |
| Gaussian blur | σ ∈ {0.5, 1.0, 2.0, 3.0} | Patient motion during exposure |
| Brightness shift | δ ∈ {−0.15, −0.10, +0.10, +0.15} | kVp / mAs exposure calibration |
| Contrast change | α ∈ {0.6, 0.8, 1.2, 1.5} | Post-processing window/level variation |
| JPEG compression | Q ∈ {90, 70, 50, 30} | PACS lossy storage / teleradiology bandwidth |
| Resolution downsample | r ∈ {0.75, 0.50, 0.25} | Mobile / portable X-ray devices |
| Gamma shift | γ ∈ {0.5, 0.75, 1.5, 2.0} | Display calibration / monitor differences |

### 3.4.3 Evaluation Protocol

For the top-*N* models by in-distribution accuracy (*N* = 5), we:

1. Compute **clean accuracy** $A_k^0$ on the unperturbed test set.
2. For each perturbation family $\mathcal{P}_j$ and each severity level *s*, apply $\mathcal{P}_j(\cdot; s)$ to the entire test set and measure the perturbed accuracy $A_k^{j,s}$ and macro F1.
3. Compute **accuracy degradation**: $\Delta_k^{j,s} = A_k^0 - A_k^{j,s}$.
4. Compute **relative robustness**: $\rho_k^{j,s} = A_k^{j,s} / A_k^0$.

### 3.4.4 Aggregate Metrics

- **Mean Relative Robustness (MRR):** $\bar{\rho}_k = \frac{1}{|\mathcal{S}|} \sum_{j,s} \rho_k^{j,s}$ where $\mathcal{S}$ is the set of all (perturbation, severity) pairs. MRR ∈ [0, 1]; higher indicates greater robustness.
- **Robustness-Accuracy Tradeoff (RAT):** $\text{RAT}_k = A_k^0 \times \bar{\rho}_k$. This product penalizes models that achieve high clean accuracy but degrade sharply under shift.

### 3.4.5 Pareto Analysis

We plot each model as a point in the (Accuracy, MRR) plane. The **Pareto front** identifies models that are not dominated on both axes simultaneously — i.e., no other model is both more accurate *and* more robust.

## 3.5 Quantitative Explainability Evaluation

### 3.5.1 Motivation

Qualitative saliency maps (Grad-CAM, LIME, etc.) are necessary but insufficient for scientific claims about model interpretability: they may highlight irrelevant regions despite appearing plausible (Adebayo et al., 2018). We complement visual explanations with three quantitative XAI metrics that measure faithfulness, causality, and stability.

### 3.5.2 Insertion / Deletion AUC (Petsiuk et al., 2018)

Given an attribution map *A*(*x*) and a trained model *f*, we measure how the predicted class probability changes as pixels are progressively revealed (insertion) or occluded (deletion) in order of attributed importance.

**Deletion** (*↓ lower is better*): Starting from the full image, we replace the top-*k*% most-attributed pixels with a black baseline at *n* = 50 uniform steps and record *f*(*x̃*)[*c*] at each step. The area under the resulting curve measures how quickly the model's confidence drops when important regions are removed:

$$\text{Del-AUC} = \int_0^1 f(\tilde{x}_t^{\text{del}})[c] \, dt$$

**Insertion** (*↑ higher is better*): Starting from a black baseline, we add the top-*k*% most-attributed pixels from the true image. A faithful attribution should cause rapid confidence recovery:

$$\text{Ins-AUC} = \int_0^1 f(\tilde{x}_t^{\text{ins}})[c] \, dt$$

The **Ins–Del gap** Δ = Ins-AUC − Del-AUC serves as an overall faithfulness proxy (larger = more faithful).

### 3.5.3 Faithfulness Correlation (Bhatt et al., 2020)

For *S* = 100 random subsets of 10% of the image pixels, we compute (a) the sum of attributions within each subset, and (b) the drop in predicted class probability when that subset is masked. The **Pearson correlation** *r* between (a) and (b) measures whether the attribution map causally explains the model's behavior. *r* ≈ 1 indicates perfect faithfulness; *r* ≈ 0 indicates the attribution map is uninformative.

### 3.5.4 Stability (Arun et al., 2021)

Stability is measured by attribution similarity under small clinically plausible perturbations (e.g., mild noise/brightness), rather than repeated runs on identical inputs. For each test image, we apply *k* = 5 perturbations drawn from the clinical shift set and compute the **pairwise SSIM** between the resulting attribution maps. High stability (SSIM → 1) indicates that the explanation is robust to acquisition variability, a prerequisite for clinical trust.

### 3.5.5 Evaluated Methods

We compute quantitative metrics for four deterministic/near-deterministic methods — Grad-CAM, Grad-CAM++, Vanilla Saliency, and Integrated Gradients — evaluated on *n* = 5 correctly-classified test images stratified across classes.

## 3.6 Ablation Studies

We conduct three targeted ablation experiments to validate individual contributions:

### 3.6.1 Preprocessing Ablation

We compare three preprocessing configurations on the same backbone (best-performing model):

| Variant | Pipeline |
|---|---|
| **Raw** | No preprocessing (resize + normalize only) |
| **Partial** | CLAHE + Gamma correction only |
| **Full** | Complete 5-stage pipeline (DAE → Bilateral → Morphological → CLAHE → Gamma) |

We report accuracy, macro F1, and ECE for each variant to quantify the marginal contribution of each preprocessing stage.

### 3.6.2 Calibration Ablation

For all trained models, we compare ECE and NLL before and after temperature scaling. This validates whether post-hoc calibration consistently improves probability estimates across architectures of varying capacity.

### 3.6.3 XAI Validity Under Distribution Shift

For the top 2–3 models, we measure insertion/deletion AUC and stability on both clean and mildly perturbed (Gaussian noise σ = 0.03) test images. A model whose explanations degrade gracefully under shift provides more clinically trustworthy interpretations.

---

# Experimental Protocol

## 4.1 Implementation Details

The complete pipeline is implemented in Python 3.9 using TensorFlow 2.x / Keras with the following key dependencies: scikit-learn (metrics, splitting), OpenCV (preprocessing), SciPy (statistical tests, optimization), statsmodels (multiple comparison correction), SHAP, LIME, matplotlib/seaborn (visualization), and NumPy. All experiments are conducted on [specify GPU, e.g., NVIDIA A100 / T4 / Apple M4] with CUDA [version] (or Apple Metal for local runs). The complete codebase, configuration, and trained models are available at [repository URL].

## 4.2 Model Zoo

We benchmark 30 architectures spanning seven families:

| Family | Architectures | Source |
|---|---|---|
| VGG | VGG-16, VGG-19 | Simonyan & Zisserman (2015) |
| ResNet | ResNet-50/101/152, ResNetV2-50/101/152 | He et al. (2016) |
| DenseNet | DenseNet-121/169/201 | Huang et al. (2017) |
| MobileNet | MobileNet, MobileNetV2, MobileNetV3-S/L | Howard et al. (2017/2019) |
| EfficientNet | EfficientNet-B0 through B7 | Tan & Le (2019) |
| Inception | InceptionV3, Inception-ResNet-V2 | Szegedy et al. (2016) |
| NASNet | NASNet-Large, NASNet-Mobile | Zoph et al. (2018) |
| Xception | Xception | Chollet (2017) |
| Transformer | ViT (via HuggingFace) | Dosovitskiy et al. (2021) |
| Custom | Modified LeNet-5 (from scratch) | LeCun et al. (1998), adapted |

All models except Modified LeNet-5 use ImageNet-pretrained weights. The classification head is identical across all architectures (Section 3.2.1). Modified LeNet-5 retains the convolutional topology of the original LeNet-5 (6→16→120 filters, 5×5 kernels) but is adapted from 32×32×1 to 224×224×3 input and shares the same classification head as the pretrained models to ensure a fair comparison. It is trained entirely from random initialisation and serves as the **no-transfer-learning baseline**, isolating the contribution of ImageNet pretraining.

## 4.3 Minimum Results Package

The following outputs are generated for every experiment run:

**Tables:**

- **Table A** — In-distribution performance: Accuracy (95% CI), Macro F1, Weighted F1, Macro ROC-AUC, Macro PR-AUC, Cohen's κ, MCC for all 30 models.
- **Table B** — Calibration summary: ECE, Brier Score, NLL (before vs. after temperature scaling) for all models.
- **Table C** — Robustness summary: Clean accuracy, mean accuracy drop, worst-case perturbation drop, MRR, RAT, and rank for top-*N* models.
- **Table D** — Quantitative XAI: Insertion AUC, Deletion AUC, Ins–Del Δ, Faithfulness *r*, and Stability for each XAI method.

**Figures:**

- **Fig. 1** — Reliability diagrams (best- and worst-calibrated model).
- **Fig. 2** — Robustness degradation curves (accuracy vs. severity) for top 3 models.
- **Fig. 3** — Accuracy-drop heatmap (models × perturbation types).
- **Fig. 4** — Pareto front plot (clean accuracy vs. MRR).
- **Fig. 5** — Quantitative XAI: insertion/deletion curves + stability distributions.
- **Fig. 6** — Qualitative XAI panel (curated examples), annotated with quantitative faithfulness scores.

All figures are rendered at 300 DPI with medical-appropriate colormaps (inferno/magma) and are LaTeX-compatible. Tables are exported in both CSV and LaTeX formats.

## 4.4 Reporting Standards

We follow TRIPOD-AI (Collins et al., 2024) and CLAIM (Mongan et al., 2020) reporting guidelines for AI in medical imaging. Hardware specifications, software versions, training time, and all random seeds are recorded in a machine-readable experiment log (JSON).
