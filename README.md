# ECG-Based Heart Failure Detection: 12-Lead and Single-Lead Models

Detecting heart failure with reduced ejection fraction (HFrEF) from ECG signals — two complementary models built for two different deployment realities: a high-performance 12-lead clinical model, and a compact single-lead model engineered to run on real single-channel wearable hardware (MAX30001).

---

## Summary

| | 12-Lead Model | Single-Lead Model |
|---|---|---|
| **Input** | Full 12-lead ECG (250 Hz, 10s) | Lead II only (250 Hz, 10s) — simulates MAX30001 hardware |
| **Architecture** | Fine-tuned ECG-FM (wav2vec2-style transformer, 90.9M params, 25.5M trainable) | CNN-Transformer hybrid + path-signature features, 0.81M params, trained from scratch |
| **Target** | HFrEF (LVEF ≤ 35%) + RV dysfunction, LVH, SHD (multi-task) | HFrEF (LVEF ≤ 35%) + RV dysfunction, LVH, SHD (multi-task) |
| **Test AUROC (HFrEF)** | **0.908** | **0.857** (95% CI: 0.842–0.871) |
| **Test set** | 5,442 patients (4,827 with valid HFrEF labels) | Same held-out patients |

Both models are benchmarked against three independently published sources — see [Benchmarks](#benchmarks-against-published-work) below.

---

## Dataset

Both models are trained and evaluated on **[EchoNext-Mini](https://physionet.org/content/echonext/)** (also referred to as "EchoNext" on PhysioNet), a dataset of 100,000 12-lead ECGs paired with echocardiogram-confirmed structural heart disease labels, collected at Columbia University Irving Medical Center.

- Reference: Hughes, J.W., Jing, L., Finer, J., et al. (2026). *EchoNext-Mini: A Dataset and Baseline AI Model for Detecting Structural Heart Disease from Electrocardiograms.* NEJM AI. https://doi.org/10.1056/AIdbp2500516
- Original large-scale study: Poterucha, T.J., Jing, L., Ricart, R.P., et al. (2025). *Detecting structural heart disease from electrocardiograms using AI.* Nature, 644, 221–230.

**⚠️ Data is not included in this repository.** EchoNext-Mini is distributed under PhysioNet's Restricted Health Data License and requires a signed Data Use Agreement. To reproduce this work:

1. Register at [physionet.org](https://physionet.org) and request access to EchoNext.
2. Download the waveform (`.npy`) and metadata (`.csv`) files per PhysioNet's instructions.
3. Run the scripts in `data_prep/` to reproduce the single-lead extraction and feature engineering described below.

Patient-level splits (train / validation / test / no_split) are used exactly as released, with no re-splitting, to avoid patient leakage across sets.

---

## Repository Structure

```
├── data_prep/              # Preprocessing, lead-II extraction, feature engineering
│   ├── singlelead_prep.py         # Extracts lead II, builds tabular + label arrays
│   ├── prepare_multitarget.py     # Multi-target label construction (12-lead)
│   ├── prepare_full72k.py         # Full 72,475-patient training set builder
│   └── signature_features.py      # Path-signature (log-signature) feature computation
├── models/
│   ├── 12lead_ecgfm/
│   │   ├── model.py                # MultiTargetHFModel (ECG-FM + tabular fusion)
│   │   └── train.py
│   └── singlelead_hybrid/
│       ├── model.py                 # CNN-Transformer hybrid + signature fusion
│       └── train.py
├── evaluation/
│   ├── compute_metrics.py          # AUROC, AUPRC, F1, MCC, Brier score, bootstrap CI
│   ├── generate_roc_pr_plots.py
│   └── compare_benchmarks.py       # Comparison against published papers
├── docs/
│   ├── Final_Comparative_Report.docx
│   ├── SingleLead_Model_Overview.docx
│   └── figures/                     # ROC/PR curves, confusion matrices, flowcharts
├── requirements.txt
└── README.md
```

---

## Methodology Overview

### 12-Lead Model
1. **Preprocessing** — resample 250→500 Hz to match ECG-FM's expected input rate; standardize tabular covariates (train-fit); patient-level split.
2. **Backbone** — [ECG-FM](https://doi.org/10.1093/jamiaopen/ooaf122) (McKeen et al., 2025), a wav2vec2-style transformer pretrained on 1.5M ECGs. 70% of layers frozen; remainder fine-tuned with differential learning rates (2e-6 backbone, 1e-4 new layers).
3. **Fusion** — ECG-FM waveform embedding concatenated with a tabular MLP (7 covariates: sex, ventricular rate, atrial rate, PR interval, QRS duration, QTc, age).
4. **Training** — masked, class-weighted BCE loss (NaN-masked missing labels), gradient clipping, early stopping on validation AUROC.

### Single-Lead Model
1. **Preprocessing** — lead II extracted from the 12-lead waveform; tabular features (heart rate, QRS duration, SDNN, RMSSD, age, sex) recomputed from lead II alone, **not** reused from the original multi-lead-derived values, to honestly simulate single-channel hardware.
2. **Three-branch feature extraction:**
   - **Waveform branch** — 1D CNN stem + 3-layer Transformer encoder (4 heads, 128-dim), trained from scratch.
   - **Tabular branch** — MLP over the 6 lead-II-derived clinical features.
   - **Path-signature branch (novel)** — 23 features from a log-signature transform (rough path theory, depth 3), including a lead-lag self-embedding capturing rhythm regularity. This method had not previously been applied to structural heart disease detection.
3. **Fusion** — concatenation of all three branches, feeding a shared trunk and four multi-task sigmoid heads.
4. **Training** — data augmentation (time-shift, amplitude scaling, Gaussian noise), dropout (0.25–0.35), AdamW + cosine annealing, early stopping.

---

## Results

### Full Test-Set Metrics (HFrEF, primary target)

| Metric | 12-Lead | Single-Lead |
|---|---|---|
| AUROC | 0.908 | 0.857 |
| AUPRC | 0.676 | 0.482–0.505 |
| Sensitivity | 83.3% | 74.6% |
| Specificity | 85.3% | 80.1% |
| Balanced Accuracy | 84.6% | 77.3% |
| F1 Score | 0.580 | 0.473 |
| MCC | 0.540 | 0.407 |
| Brier Score | 0.112 | 0.138 |

### Auxiliary Targets (both models predict these simultaneously)

| Target | 12-Lead AUROC | Single-Lead AUROC |
|---|---|---|
| RV dysfunction | 0.881 | 0.830 |
| SHD (composite) | 0.833 | 0.787 |
| LVH | 0.758 | 0.710 |

---

## Benchmarks Against Published Work

| Source | Dataset | Target | AUROC |
|---|---|---|---|
| Poterucha et al., *Nature* 2025 | Full EchoNext (1.24M pairs, 8 hospitals) | LVEF ≤ 45% | 0.904 |
| Hughes et al., *NEJM AI* 2026 | EchoNext-Mini | Composite SHD | 0.820 |
| Do et al., *arXiv:2604.23385* 2026 | EchoNext-Mini (same split) | LVEF ≤ 45% | 0.904 (b=9) |
| **This work — 12-lead** | EchoNext-Mini | **LVEF ≤ 35%** | **0.908** |
| **This work — single-lead** | EchoNext-Mini | **LVEF ≤ 35%** | **0.857** |

*Note: this project's HFrEF definition (LVEF ≤ 35%, severe) is a stricter threshold than the ≤ 45% used in the three comparators above; see `docs/Final_Comparative_Report.docx` for full discussion.*

---

## Key Findings

- **Path-signature features** are a genuine methodological contribution — never previously applied to this detection task.
- **A/B leakage check**: single-lead tabular features recomputed honestly from lead II alone performed nearly identically to versions using borrowed multi-lead precision, confirming the single-lead result is not inflated.
- **Multi-task vs. single-task ablation**: predicting all four targets together costs a small, precisely quantified ~0.007–0.008 AUROC on the primary HFrEF task, versus training on HFrEF alone.
- **Negative results reported honestly**: signature-feature fusion improved validation AUROC but not test AUROC in one configuration; automatic deflection correction was tested and deliberately not applied after being shown to remove real diagnostic signal (negatively-deflected ECGs had 2.6× higher HFrEF prevalence).

---

## Requirements

See `requirements.txt`. Core dependencies: `torch`, `fairseq-signals` (for ECG-FM), `iisignature` (for path signatures), `scikit-learn`, `numpy`, `pandas`.

---

## License

Code in this repository is released under the MIT License (see `LICENSE`). This does not extend to the EchoNext-Mini dataset, which remains under PhysioNet's Restricted Health Data License and must be obtained separately.

---

## Citation

If you use this code, please cite the original EchoNext-Mini dataset (see [Dataset](#dataset) above) alongside this repository.
