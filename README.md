# Indoor Location Prediction via Wi-Fi Signals

**Project 9 · CSAI Student Project · Phase 1: Machine Learning Evaluation**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)

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
- **Device & Antenna Bias**: Different smartphone models receive signals with varying gain curves. The evaluation benchmark tests generalization to unseen devices and human collectors.
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

The project utilizes the official **UJIIndoorLoc** benchmark:

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

### 1. Leaderboard Comparison

| Architecture | Building Acc (%) | Floor Acc (%) | Both Match (%) | Mean Error (m) | Median Error (m) | 75th %ile Error (m) | 90th %ile Error (m) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Random Forest Ensemble (n=100)** | **99.91%** | **90.55%** | **90.46%** | **13.24m** | **7.92m** | **15.42m** | **31.18m** |
| **XGBoost (Gradient Boosted Trees)** | 99.82% | 88.57% | 88.48% | 14.15m | 8.84m | 17.61m | 32.80m |
| **K-Nearest Neighbors ($k=5$)** | 99.82% | 86.86% | 86.77% | 15.68m | 9.47m | 20.31m | 36.45m |

> **Champion Model**: **Random Forest** achieved the best overall performance with a median positioning error under **8 meters** and a 99.91% building classification accuracy.

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
│   └── 03_model_application.ipynb     # Model training, comparisons & error maps
├── Reports/
│   └── champion_models.joblib         # Serialized champion models and evaluation tables
├── figures/
│   ├── spatial_distribution.png       # Campus 2D coordinate footprint by building/floor
│   ├── class_distribution.png         # Building and floor class proportions
│   ├── rssi_distribution.png          # Empirical RSSI signal distribution
│   ├── ap_visibility_distribution.png # Active AP counts per fingerprint scan
│   ├── model_error_comparison.png     # Boxplots & ECDF cumulative error curves
│   ├── floor_confusion_matrix.png     # Champion floor classification confusion matrix
│   └── spatial_error_map.png          # Spatial heatmap of Euclidean prediction errors
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
