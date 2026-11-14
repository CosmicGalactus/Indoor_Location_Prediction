# Indoor Location Prediction via Wi-Fi Signals

**Project 9 · CSAI Student Project · Phase 1 (AML) & Phase 2 (Deep Learning & Hybrid Innovation)**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Executive Summary

Indoor positioning represents a major engineering and machine learning challenge because Global Positioning System (GPS) signals attenuate and scatter heavily inside modern architectural structures. This project addresses the indoor localization problem using **Wi-Fi Fingerprinting** across three multi-story buildings (~110,000 $m^2$) of the Universitat Jaume I (UJI) campus using the **UJIIndoorLoc** dataset.

The system simultaneously solves a **hierarchical multi-task learning** objective:
1. **Multi-Class Classification**: Identifying the **Building** (0, 1, 2) and **Floor** (0 to 4).
2. **Spatial Coordinate Regression**: Estimating continuous **Latitude** and **Longitude** positions.

The final benchmarked system achieves **99.91% Building Accuracy**, **90.55% Floor Accuracy**, and a **Mean Euclidean Positioning Error of 13.24 meters** (Median: 7.92m) on the completely held-out, unseen-user validation set.

---

## Problem Formulation & Hybrid Logic

### 1. The Real-World Gap in Wi-Fi Fingerprinting
- **Signal Sparsity**: Over 96.5% of Access Points are undetected during any individual scan. The dataset encodes absence with a `+100 dBm` artifact, requiring consistent physical thresholding and positive scaling.
- **Device & Antenna Bias**: Different smartphone models receive signals with differing gain curves. The evaluation benchmark tests generalization to unseen devices and human collectors.
- **Structural Constraints**: Standard spatial regression models operating independently can predict coordinates that violate architectural boundaries or fail to capture the hierarchical relationship between buildings, floors, and spatial rooms.

### 2. Physical & Statistical Formulation
Let an observed Wi-Fi fingerprint be represented as:
$$\mathbf{s} = [s_1, s_2, \dots, s_M] \in \mathbb{R}^M$$
where $M=520$ is the total universe of Wireless Access Points (WAPs), and $s_i \in [-104, 0]$ dBm if detected, or $s_i = +100$ if unobserved.

The mapping targets are:
- $\hat{b} \in \{0, 1, 2\}$ (Building ID)
- $\hat{f} \in \{0, 1, 2, 3, 4\}$ (Floor Level)
- $(\hat{x}, \hat{y}) \in \mathbb{R}^2$ (Latitude and Longitude in meters)

---

## Dataset & Splitting Invariants

The project utilizes the official **UJIIndoorLoc** benchmarks:

| Split | Rows | WAP Features | Unique Users | Unique Devices | Description |
|---|---:|---:|---:|---:|---|
| **Training Set** (`trainingData.csv`) | 19,861 | 465 (active) | 18 | 16 | Training calibration fingerprints (76 zero-detection scans pruned) |
| **Validation Set** (`validationData.csv`) | 1,111 | 465 (active) | 1 | 11 | Held-out test paths collected with distinct device profiles |

### Preprocessing & Normalization Invariants
1. **Absence Thresholding**: The `+100` convention is replaced with a baseline of `-110.0 dBm`.
2. **Positive Scaling**: Signals are mapped to $[0, 1]$:
   $$\tilde{s}_i = \frac{s_i - (-110)}{0 - (-110)}$$
   An absent signal is precisely $0.0$; the strongest theoretical signal ($0$ dBm) is $1.0$.
3. **Dead WAP Elimination**: 55 WAPs that never fired across the entire training dataset are identified and pruned.
4. **Engineered Feature Descriptors**:
   - `signal_count`: Total number of active APs detected in the scan.
   - `max_signal`: Normalized RSSI of the strongest observed AP (direct physical proxy for proximity to closest router).
   - `mean_signal`: Average normalized strength across detected APs.
   - `std_signal`: Variance/spread of visible signals.

---

## Model Benchmark & Evaluation Results

All models were fitted strictly on the training set and evaluated on the official held-out validation set.

### 1. Master Phase 1 & Phase 2 Leaderboard

| Architecture | Paradigm | Building Acc (%) | Floor Acc (%) | Both Match (%) | Mean Error (m) | Median Error (m) | 90th %ile Error (m) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Deep-Spatial Hybrid (DSHA: DL + HKNN)** 🏆 | **Hybrid Innovation** | **100.00%** | **96.40%** | **96.40%** | **9.03m** | **7.66m** | **15.99m** |
| **Hierarchical KNN (Cosine + Bray-Curtis, $\gamma=2.0$)** | AML Optimized | **100.00%** | **96.40%** | **96.40%** | **6.55m** | **4.08m** | **15.02m** |
| **Deep Multi-Task Network (PyTorch)** | Deep Learning | 99.82% | 92.26% | 92.26% | 13.94m | 12.55m | 24.34m |
| **XGBoost (Gradient Boosted Trees)** | AML Baseline | 99.55% | 89.47% | 89.20% | 20.25m | 13.56m | 42.64m |
| **Random Forest Ensemble (n=100)** | AML Baseline | 99.82% | 87.85% | 87.76% | 16.48m | 8.76m | 39.06m |
| **Baseline K-NN ($k=5$, Euclidean)** | AML Baseline | 98.11% | 74.71% | 74.17% | 15.88m | 8.96m | 33.63m |

### 2. Phase 2 Ablation Studies

| Ablation Study | Variant A (Baseline) | Variant B (Proposed) | Δ Gain | Theoretical Justification |
|---|---|---|---|---|
| **Multi-Task vs. Single-Task** | Single-Task Floor Net: **91.54%** | Multi-Task Shared Trunk: **92.26%** | **+0.72%** | Shared spatial manifold regularizes floor classification via inductive transfer. |
| **Loss Weighting Strategy** | Static Weights ($\Sigma w_i \mathcal{L}_i$): **90.01%** | Homoscedastic Uncertainty: **92.26%** | **+2.25%** | Dynamically adapts task gradient magnitudes, avoiding coordinate domination. |
| **Data Augmentation** | Unaugmented Signals | AP-Dropout (15%) + RSSI Jitter | **+1.14%** | Simulates missing APs and cross-device antenna sensitivity shifts. |

### 2. Hierarchical Error Breakdown
Analyzing the physical dependency between structural classification and coordinate regression:

| Operational Condition | Percentage of Tests | Mean Coordinate Error | Median Coordinate Error |
|---|---:|---:|---:|
| **Both Building & Floor Correct** | **90.46%** | **9.62 meters** | **7.41 meters** |
| **Floor Misclassified** | **9.54%** | **47.58 meters** | **38.92 meters** |

*Insight*: When the structural floor classifier is correct, spatial error drops to **7.41 meters**. Misclassified floors introduce vertical and corridor displacement penalties.

---

## Visualizations & Spatial Analysis

All generated high-resolution visualizations are stored under `figures/`:
- **`figures/spatial_distribution.png`**: Campus 2D coordinate footprint colored by building and floor.
- **`figures/class_distribution.png`**: Class proportions across train and validation splits.
- **`figures/rssi_distribution.png`**: Empirical distribution of observed Wi-Fi signal strengths.
- **`figures/ap_visibility_distribution.png`**: Kernel density of detected AP count per fingerprint.
- **`figures/model_error_comparison.png`**: Boxplots & Empirical Cumulative Distribution Functions (ECDF) of Euclidean errors.
- **`figures/floor_confusion_matrix.png`**: Confusion matrix for floor classification.
- **`figures/spatial_error_map.png`**: Heatmap of spatial prediction errors across the UJI campus.

---

## Repository Structure

```text
Indoor_Location_Prediction/
├── Datasets/
│   ├── UJIndoorLoc/
│   │   ├── trainingData.csv           # Official UJI training fingerprints
│   │   └── validationData.csv         # Official UJI evaluation fingerprints
│   └── processed/
│       └── preprocessed_wifi_data.joblib # Serialized clean feature matrices
├── Notebooks/
│   ├── 01_eda.ipynb                   # Exploratory analysis & signal coverage audits
│   ├── 02_preprocessing.ipynb         # Signal transformation & feature engineering
│   ├── 03_model_application.ipynb     # Initial ML models & baseline comparisons
│   ├── 04_advanced_knn_benchmark.ipynb # Metric ablation, power scaling & Hierarchical KNN
│   └── 05_deep_multitask_and_hybrid.ipynb # Phase 2 PyTorch Multi-Task & Hybrid Innovation
├── src/
│   ├── __init__.py
│   ├── dataset.py                     # PyTorch Dataset, DataLoaders & AP-Dropout Augmentation
│   ├── deep_multitask_net.py          # MultiTaskIndoorNet & Homoscedastic Loss Balancing
│   ├── hierarchical_knn.py            # Production HierarchicalKNNPositioner class
│   ├── hybrid_positioner.py           # DeepSpatialHybridPositioner (DSHA Fusion)
│   ├── run_evaluation.py              # Automated Phase 1 evaluation script
│   └── train_and_evaluate.py          # Complete Phase 2 training & ablation pipeline
├── Reports/
│   ├── champion_multitask_dl.pt       # Serialized PyTorch Multi-Task model weights
│   ├── hybrid_champion.joblib         # Serialized DSHA hybrid model & predictions
│   └── hierarchical_knn_champion.joblib # Serialized HKNN model and evaluation tables
├── figures/
│   ├── spatial_distribution.png       # Campus 2D coordinate footprint by building/floor
│   ├── class_distribution.png         # Building and floor class proportions
│   ├── rssi_distribution.png          # Empirical RSSI signal distribution
│   ├── ap_visibility_distribution.png # Active AP counts per fingerprint scan
│   ├── model_error_comparison.png     # Boxplots & ECDF cumulative error curves
│   ├── floor_confusion_matrix.png     # Floor classification confusion matrix
│   ├── spatial_error_map.png          # Spatial heatmap of Euclidean prediction errors
│   ├── advanced_knn_performance.png   # Champion HKNN ECDF and confusion matrix
│   └── phase2_deep_and_hybrid_evaluation.png # Phase 2 loss curves, confusion matrix & spatial error
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/CosmicGalactus/Indoor_Location_Prediction.git
cd Indoor_Location_Prediction
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the End-to-End Pipeline
Execute the Jupyter notebooks in chronological sequence:
1. `Notebooks/01_eda.ipynb` — Inspect dataset distributions and verify signal absence conventions.
2. `Notebooks/02_preprocessing.ipynb` — Prune zero-variance WAPs, transform signals, and serialize features.
3. `Notebooks/03_model_application.ipynb` — Train KNN, Random Forest, and XGBoost models, generate error CDFs, and extract test metrics.

---

## References

1. **UJIIndoorLoc Dataset**: Torres-Sospedra, J., et al. (2014). *UJIIndoorLoc: A New Multi-building and Multi-floor Database for WLAN Fingerprint-based Indoor Localization Problems.* In Proc. IPIN 2014. [DOI: 10.1109/IPIN.2014.7275492](https://doi.org/10.1109/IPIN.2014.7275492)
2. **Wi-Fi Indoor Localization Survey**: He, S., & Chan, S. H. G. (2016). *Wi-Fi Fingerprint-Based Indoor Positioning: Recent Advances and Comparisons.* IEEE Communications Surveys & Tutorials.
3. **Random Forest Ensembles**: Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5-32.
4. **XGBoost**: Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* In Proc. KDD 2016.

---

## Team & Contribution Details

* **Course**: CSAI Student Project
* **Project**: Project 9 — Indoor Location Prediction Using Wi-Fi Signals
* **Evaluation Phase**: Phase 1 (Machine Learning Evaluation)
* **Team**: Team T029 · Section E
* **Repository**: [https://github.com/CosmicGalactus/Indoor_Location_Prediction](https://github.com/CosmicGalactus/Indoor_Location_Prediction)


* **Team name**: T071
* **Project ID**: 9
Rishav Dewan : 2401010379
Balajee : 2401010123
Pragya Kashyap : 2401010511
Prakhar srivastav : 2401010135
Soumen Dass : 2401010458
