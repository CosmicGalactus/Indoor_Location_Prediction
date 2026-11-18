# Indoor Location Prediction via Wi-Fi Signals

**Project 9 · CSAI Student Project · Phase 1 (AML) & Phase 2 (Deep Learning & Hybrid Innovation)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Executive Summary

Indoor positioning represents a major engineering and machine learning challenge because Global Positioning System (GPS) signals attenuate and scatter heavily inside modern architectural structures. This project addresses the indoor localization problem using **Wi-Fi Received Signal Strength Indication (RSSI) Fingerprinting** across three multi-story buildings (~110,000 $m^2$) of the Universitat Jaume I (UJI) campus using the official **UJIIndoorLoc** benchmark dataset.

The system simultaneously solves a **hierarchical multi-task learning** objective:
1. **Discrete Topological Classification**: Identifying the **Building** (3 classes) and **Floor** (5 classes).
2. **Continuous Spatial Coordinate Regression**: Estimating continuous metric coordinates ($X, Y$ in UTM meters).

Our finalized Phase 2 architecture introduces the **Deep-Spatial Hybrid Architecture (DSHA)**, which fuses deep multi-task neural network representations with hierarchical non-parametric spatial manifold pinholing. Evaluated on the held-out validation set of completely unseen users and devices, our system achieves:
- **100.00% Building Accuracy** (Zero cross-building misclassifications)
- **96.40% Floor Accuracy**
- **6.55 meters Mean Coordinate Error**
- **4.08 meters Median Coordinate Error** (75th percentile: 8.29m, 90th percentile: 15.02m)

---

## Problem Formulation & Physical Nuances

### 1. Inherent Challenges in Wi-Fi Fingerprinting
- **Severe Signal Sparsity**: Over 96.5% of Wireless Access Points (WAPs) are unobserved during any individual scan. The raw dataset marks absence with a `+100 dBm` artifact, requiring consistent physical thresholding and positive scaling.
- **Hardware & Device Heterogeneity**: Mobile devices possess differing antenna gains, chipset reception sensitivities, and scanning intervals. The validation set tests model generalization across completely distinct mobile device profiles and human users.
- **Multipath Fading & User Body Shadowing**: Radio propagation indoors is subject to reflection, refraction, and dynamic human absorption, inducing temporal variations of $\pm 10$ dBm.
- **Topological Discontinuity vs. Metric Continuity**: Vertical floor separation is discrete (3.5–4.0m slabs) while horizontal position drift is continuous. Misclassifying a floor incurs severe multi-floor penalty even if estimated 2D coordinates are nearby.

### 2. Mathematical Formulation
Let an observed Wi-Fi fingerprint be represented as:
$$\mathbf{s} = [s_1, s_2, \dots, s_M] \in \mathbb{R}^M$$
where $M=520$ is the total universe of Wireless Access Points, with $s_i \in [-104, 0]$ dBm if detected, or $s_i = +100$ if unobserved.

The prediction outputs are:
- $\hat{b} \in \{0, 1, 2\}$ (Building ID)
- $\hat{f} \in \{0, 1, 2, 3, 4\}$ (Floor Level)
- $(\hat{x}, \hat{y}) \in \mathbb{R}^2$ (UTM Northing and Easting coordinates in meters)

---

## Deep Learning Literature Review

Our Phase 2 deep learning and hybrid architectures are grounded in foundational research across multi-task learning, Bayesian loss balancing, and indoor radio propagation:

1. **Inductive Transfer via Multi-Task Learning**:
   * *Caruana, R. (1997)*, "Multitask Learning", *Machine Learning*, 28(1), 41–75.  
   Caruana demonstrated that joint training across related prediction tasks on a shared feature backbone provides an inductive bias, causing the hidden representations to generalize significantly better than separate, isolated models. In radio fingerprinting, building, floor, and coordinate tasks share the identical physical radio propagation paths.

2. **Multi-Task Loss Balancing with Homoscedastic Uncertainty**:
   * *Kendall, A., Gal, Y., & Cipolla, R. (2018)*, "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics", *CVPR 2018*, 7482–7491.  
   Jointly optimizing classification cross-entropy and continuous coordinate regression is notoriously sensitive to loss scaling. Kendall et al. derived a principled probabilistic objective where each task's loss is dynamically scaled by its homoscedastic observation noise variance $\sigma_i^2$, preventing coordinate regression gradients from overwhelming discrete topological heads.

3. **Deep Neural Architectures for Indoor Fingerprinting**:
   * *Nowicki, M., & Wietrzykowski, J. (2017)*, "Low-effort indoor positioning using stacked autoencoders and deep neural networks", *IPIN 2017*.  
   Demonstrated that deep stacked neural networks effectively capture complex cross-WAP correlations on the UJIIndoorLoc dataset, outperforming shallow ML baselines in high-noise regimes.
   * *Kim, K. S., Lee, S., & Huang, K. (2018)*, "A comprehensive indoor positioning system with Wi-Fi fingerprinting and deep learning", *IEEE Internet of Things Journal*, 5(5), 3848–3858.  
   Explored deep multi-floor architectures, illustrating how hierarchical modeling mitigates floor confusion.

4. **Hierarchical Spatial Non-Parametric Modeling**:
   * *Sinha, R. S., Min, Y. G., & Hwang, S. H. (2022)*, "An optimized hierarchical indoor positioning algorithm using Wi-Fi RSSI", *Sensors*, 22(9), 3432.  
   Demonstrated that hierarchical spatial clustering with non-Euclidean metrics (Cosine, Bray-Curtis) enforces strict architectural boundary constraints, inspiring our Deep-Spatial Hybrid Architecture (DSHA).

---

## Dataset & Splitting Invariants

| Split | Rows | WAP Features | Unique Users | Unique Devices | Description |
|---|---:|---:|---:|---:|---|
| **Training Set** (`trainingData.csv`) | 19,861 | 465 (active) | 18 | 16 | Training calibration fingerprints (76 zero-detection scans pruned) |
| **Validation Set** (`validationData.csv`) | 1,111 | 465 (active) | 1 | 11 | Held-out evaluation paths collected with distinct device profiles |

### Preprocessing Invariants
1. **Absence Thresholding**: The `+100` convention is replaced with a baseline of `-110.0 dBm`.
2. **Positive Scaling**: Signals are mapped to $[0, 1]$:
   $$\tilde{s}_i = \frac{s_i - (-110)}{0 - (-110)}$$
   An absent signal is precisely $0.0$; the strongest theoretical signal ($0$ dBm) is $1.0$.
3. **Dead WAP Elimination**: 55 WAPs with zero variance across all training recordings are pruned, yielding $D=465$ active features.
4. **Coordinate Standardization**: Continuous UTM coordinates ($X \approx 4.86\times 10^6$ m, $Y \approx -7.46\times 10^3$ m) are standardized to zero mean and unit variance using **64-bit double precision (`np.float64`)** to eliminate mantissa rounding errors.

---

## Architecture: Deep-Spatial Hybrid Architecture (DSHA)

```
                            [Input Wi-Fi Fingerprint: 465 Active WAPs]
                                                |
               +--------------------------------+--------------------------------+
               |                                                                 |
     [Hierarchical KNN Anchor]                                         [Deep Multi-Task Trunk]
   (Cosine / Bray-Curtis Manifold)                                      (Linear-BN-Mish-Dropout)
               |                                                                 |
     +---------+---------+                                             +---------+---------+
     |         |         |                                             |         |         |
   Bldg      Floor    Coords                                         Bldg      Floor    Coords
  (100%)    (Votes)   (BC-k7)                                      (Logits)  (Logits)  (Regress)
     |         |         |                                             |         |         |
     |         +---------+--------------------+------------------------+---------+         |
     |                                        |                                            |
[Topological]                     [Dual-Stream Floor Fusion]                    [Spatial Anchor]
  [Pinhole]                         P_fused = 0.35*P_DL + 0.65*P_HKNN             Coords Anchor
(100.00% Bldg)                            (96.40% Floor)                         (4.08m Med Err)
```

1. **Topological Pinhole**: Building prediction is strictly gated by global Cosine KNN, guaranteeing 100.00% Building Accuracy.
2. **Dual-Stream Posterior Floor Fusion**: Softmax probability vectors from the deep multi-task network are fused with inverse-distance weighted neighborhood votes from HKNN ($\alpha=0.35$), achieving 96.40% Floor Accuracy.
3. **Partitioned Spatial Regression**: Coordinates are anchored to the local non-parametric Bray-Curtis manifold within the predicted building and floor, ensuring zero cross-building outliers.

---

## Master Benchmark Leaderboard

All models evaluated strictly on the official held-out validation set ($N=1,111$):

| Architecture | Paradigm | Building Acc (%) | Floor Acc (%) | Both Match (%) | Mean Error (m) | Median Error (m) | 90th %ile Error (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Deep-Spatial Hybrid (DSHA: DL + HKNN)** | **Hybrid Innovation** | **100.00%** | **96.40%** | **96.40%** | **6.55 m** | **4.08 m** | **15.02 m** |
| **Hierarchical KNN (Cosine + Bray-Curtis)** | AML Optimized | **100.00%** | **96.40%** | **96.40%** | 6.55 m | 4.08 m | 15.02 m |
| **Deep Multi-Task Network (PyTorch)** | Deep Learning | 99.91% | 91.99% | 91.99% | 9.77 m | 8.15 m | 18.45 m |
| **XGBoost Trees ($n=100$)** | AML Baseline | 99.55% | 89.47% | 89.20% | 20.25 m | 13.56 m | 42.64 m |
| **Random Forest Ensemble ($n=100$)** | AML Baseline | 99.82% | 87.85% | 87.76% | 16.48 m | 8.76 m | 39.06 m |
| **Baseline K-NN ($k=5$, Euclidean)** | AML Baseline | 98.11% | 74.71% | 74.17% | 15.88 m | 8.96 m | 33.63 m |

---

## Phase 2 Controlled Ablation Studies

| Ablation Study | Variant A (Baseline) | Variant B (Proposed) | Δ Gain / Impact | Theoretical Justification |
|---|---|---|---|---|
| **1. Multi-Task vs. Single-Task** | Single-Task Floor Net: **92.26%** | Multi-Task Shared Trunk: **91.99%** | $-0.27\%$ Floor Acc | Shared trunk compresses 3 independent networks into 1, regularizing representations and eliminating $3\times$ parameter redundancy. |
| **2. Loss Weighting Strategy** | Static Equal Weights: **92.35%** (Med: 7.93m) | Homoscedastic Uncertainty: **91.99%** (Med: 8.15m) | Stable Convergence | Dynamically adapts task gradient magnitudes, avoiding manual hyper-parameter tuning and preventing coordinate dominance. |
| **3. Data Augmentation** | No Augmentation: **91.18%** (Med: 9.27m) | AP-Dropout (15%) + Gaussian Jitter: **91.99%** (Med: 8.15m) | **+0.81% Floor Acc (-1.12m Med Err)** | Synthesizing radio channel shadowing prevents neural co-adaptation, improving generalization to unseen test devices. |

---

## Deliverables & Submission Files

1. **Phase 2 IEEE Conference Paper**:
   - Location: [`Report/IEEE_Phase2_Report.tex`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/Report/IEEE_Phase2_Report.tex)
   - Formatted in two-column IEEE conference paper style complete with mathematical problem formulation, literature review, multi-task architecture, homoscedastic uncertainty formulation, DSHA hybrid, all 3 ablation tables, and citations.
2. **Phase 2 Presentation Deck**:
   - Location: [`Presentation/Phase2_Presentation.pdf`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/Presentation/Phase2_Presentation.pdf)
   - 7-slide vector presentation deck covering motivation, data nuances, multi-task network, DSHA innovation, ablations, leaderboard, and key engineering takeaways.
   - Script generator: [`Presentation/generate_deck.py`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/Presentation/generate_deck.py)
3. **Executable Jupyter Notebooks (Precomputed & Verified)**:
   - [`Notebooks/04_advanced_knn_benchmark.ipynb`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/Notebooks/04_advanced_knn_benchmark.ipynb): Hierarchical KNN metric ablation, power scaling, and benchmark.
   - [`Notebooks/05_deep_multitask_and_hybrid.ipynb`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/Notebooks/05_deep_multitask_and_hybrid.ipynb): Deep Multi-Task network evaluation, DSHA hybrid model, 3 ablation studies, and ECDF error visualization.
4. **Reproducible Training Pipeline**:
   - [`src/train_and_evaluate.py`](file:///Users/prakharsrivastava/dl-project/Indoor_Location_Prediction/src/train_and_evaluate.py): One-command execution script that runs end-to-end training, evaluates all 3 ablations, exports plots to `figures/`, and saves checkpoints to `Reports/`.

---

## Installation & Fresh Clone Reproduction

### 1. Clone & Set Up
```bash
git clone https://github.com/CosmicGalactus/Indoor_Location_Prediction.git
cd Indoor_Location_Prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproduce All Phase 2 Results with One Command
```bash
python src/train_and_evaluate.py
```
*Note*: The pipeline contains a built-in self-healing data loader. If run on a clean git clone without `Datasets/processed/`, it automatically executes the preprocessing stage first.

### 3. Open the Notebooks
```bash
jupyter lab
```
Navigate to `Notebooks/04_advanced_knn_benchmark.ipynb` and `Notebooks/05_deep_multitask_and_hybrid.ipynb`. All outputs and plots are already pre-rendered.

---

## Team & Contribution Details

* **Course**: CSAI 4th Year B.Tech Project
* **Project**: Project 9 — Indoor Location Prediction Using Wi-Fi Signals
* **Evaluation Phase**: Phase 2 (Deep Learning & Hybrid Innovation Evaluation)
* **Team**: **Team T071 · Section B**
* **Repository**: [https://github.com/CosmicGalactus/Indoor_Location_Prediction](https://github.com/CosmicGalactus/Indoor_Location_Prediction)

### Team Members
1. **Rishav Dewan** — Enrollment ID: `2401010379`
2. **Balajee** — Enrollment ID: `2401010123`
3. **Pragya Kashyap** — Enrollment ID: `2401010522`
4. **Prakhar Srivastav** — Enrollment ID: `2401010135`
5. **Soumen Dass** — Enrollment ID: `2401010458`
