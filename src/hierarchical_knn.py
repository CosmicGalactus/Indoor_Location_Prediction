"""
Hierarchical K-Nearest Neighbors (HKNN) for Wi-Fi Indoor Localization
======================================================================
This module implements an advanced multi-stage, physics-informed KNN architecture
tailored for sparse radio frequency fingerprinting (UJIIndoorLoc benchmark).

Key Architectural Innovations:
1. Signal Power Scaling (Gamma Transform):
   Applies a non-linear power transform (gamma=2.0) to normalized RSSI to suppress
   multipath-induced noise in weak signals and emphasize dominant direct-path APs.
2. Cosine Distance Metric for Classification:
   Measures the angular direction and relative signal ratios across WAPs rather
   than absolute magnitudes, providing mathematical invariance to smartphone
   hardware antenna gain bias (addressing the cross-device domain shift).
3. Hierarchical (Cascaded) Topology:
   - Stage 1: Coarse Building Classification via Cosine KNN (k=5).
   - Stage 2: Fine-grained Floor Classification conditioned on predicted Building (k=9, distance-weighted).
   - Stage 3: Spatial Coordinate Regression conditioned on predicted Building via Bray-Curtis metric (k=7).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


class HierarchicalKNNPositioner(BaseEstimator):
    """
    Hierarchical K-Nearest Neighbors Indoor Positioner.
    
    Parameters
    ----------
    gamma : float, default=2.0
        Non-linear power scaling exponent applied to normalized RSSI features.
    k_bldg : int, default=5
        Number of neighbors for Stage 1 (Building classification).
    k_floor : int, default=9
        Number of neighbors for Stage 2 (Floor classification).
    k_coords : int, default=7
        Number of neighbors for Stage 3 (Coordinate regression).
    metric_bldg : str, default='cosine'
        Distance metric for building classification.
    metric_floor : str, default='cosine'
        Distance metric for floor classification.
    metric_coords : str, default='braycurtis'
        Distance metric for spatial coordinate regression.
    weights : str, default='distance'
        Weight function used in prediction ('uniform' or 'distance').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        k_bldg: int = 5,
        k_floor: int = 9,
        k_coords: int = 7,
        metric_bldg: str = "cosine",
        metric_floor: str = "cosine",
        metric_coords: str = "braycurtis",
        weights: str = "distance",
        n_jobs: int = -1,
    ):
        self.gamma = gamma
        self.k_bldg = k_bldg
        self.k_floor = k_floor
        self.k_coords = k_coords
        self.metric_bldg = metric_bldg
        self.metric_floor = metric_floor
        self.metric_coords = metric_coords
        self.weights = weights
        self.n_jobs = n_jobs

        self.bldg_model_ = None
        self.floor_models_ = {}
        self.coord_models_ = {}
        self.buildings_ = None

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Applies power scaling to normalized RSSI features."""
        X_arr = np.asarray(X, dtype=np.float32)
        if self.gamma != 1.0:
            return np.power(X_arr, self.gamma)
        return X_arr

    def fit(
        self,
        X: np.ndarray,
        y_bldg: np.ndarray,
        y_floor: np.ndarray,
        y_coords: np.ndarray,
    ):
        """
        Fit the hierarchical models on training fingerprints and multi-task targets.
        """
        X_trans = self._transform_features(X)
        y_bldg = np.asarray(y_bldg, dtype=np.int64)
        y_floor = np.asarray(y_floor, dtype=np.int64)
        y_coords = np.asarray(y_coords, dtype=np.float64)

        self.buildings_ = np.unique(y_bldg)

        # Stage 1: Global Building Classifier
        self.bldg_model_ = KNeighborsClassifier(
            n_neighbors=self.k_bldg,
            metric=self.metric_bldg,
            weights=self.weights,
            n_jobs=self.n_jobs,
        )
        self.bldg_model_.fit(X_trans, y_bldg)

        # Stage 2 & 3: Per-Building Floor Classifiers and Coordinate Regressors
        self.floor_models_ = {}
        self.coord_models_ = {}

        for b in self.buildings_:
            bldg_mask = y_bldg == b
            X_b = X_trans[bldg_mask]
            y_fl_b = y_floor[bldg_mask]
            y_co_b = y_coords[bldg_mask]

            k_fl = min(self.k_floor, len(X_b))
            fl_clf = KNeighborsClassifier(
                n_neighbors=k_fl,
                metric=self.metric_floor,
                weights=self.weights,
                n_jobs=self.n_jobs,
            )
            fl_clf.fit(X_b, y_fl_b)
            self.floor_models_[b] = fl_clf

            k_co = min(self.k_coords, len(X_b))
            co_reg = KNeighborsRegressor(
                n_neighbors=k_co,
                metric=self.metric_coords,
                weights=self.weights,
                n_jobs=self.n_jobs,
            )
            co_reg.fit(X_b, y_co_b)
            self.coord_models_[b] = co_reg

        return self

    def predict(self, X: np.ndarray):
        """
        Predict Building, Floor, and 2D Coordinates for incoming fingerprints.

        Returns
        -------
        pred_bldg : np.ndarray of shape (N,)
        pred_floor : np.ndarray of shape (N,)
        pred_coords : np.ndarray of shape (N, 2)
        """
        X_trans = self._transform_features(X)
        N = len(X_trans)

        # Stage 1: Predict Building
        pred_bldg = self.bldg_model_.predict(X_trans)

        pred_floor = np.zeros(N, dtype=np.int64)
        pred_coords = np.zeros((N, 2), dtype=np.float64)

        # Stages 2 & 3: Hierarchical inference per building
        for b in self.buildings_:
            mask = pred_bldg == b
            if not np.any(mask):
                continue
            X_sub = X_trans[mask]
            pred_floor[mask] = self.floor_models_[b].predict(X_sub)
            pred_coords[mask] = self.coord_models_[b].predict(X_sub)

        return pred_bldg, pred_floor, pred_coords

