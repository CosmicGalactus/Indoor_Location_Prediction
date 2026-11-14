"""
Deep Multi-Task Neural Network for Indoor Localization
======================================================
Implements a shared representation deep neural network (MultiTaskIndoorNet)
with dedicated classification and regression heads, residual blocks,
and homoscedastic uncertainty loss balancing (Kendall et al., CVPR 2018).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual MLP Block with LayerNorm and Swish (SiLU) activation."""
    def __init__(self, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.ln1(self.fc1(x)))
        out = self.dropout(out)
        out = self.ln2(self.fc2(out))
        out = self.act(out + residual)
        return out


class MultiTaskIndoorNet(nn.Module):
    """
    Shared Deep Multi-Task Neural Network for Wi-Fi Fingerprinting.
    
    Architecture:
    -------------
    Inputs (465 WAPs)
           │
      Input Projection (465 -> 512, LayerNorm, SiLU)
           │
      Residual Block 1 (512 -> 512)
           │
      Trunk Compression (512 -> 256, LayerNorm, SiLU)
           │ (Shared Latent Space)
      ┌────┴───────────────────────────┐
      │                                │
    Building Head (3 cls)          Floor Head (5 cls)       Coordinate Head (2 coords)
    """
    def __init__(
        self,
        input_dim: int = 465,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        n_buildings: int = 3,
        n_floors: int = 5,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim

        # Shared Backbone Trunk
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_block1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.trunk_compression = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
        )

        # Task Head 1: Building Classification
        self.building_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, n_buildings),
        )

        # Task Head 2: Floor Classification
        self.floor_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, n_floors),
        )

        # Task Head 3: Coordinate Regression (Normalized Latitude, Longitude)
        self.coord_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout / 2.0),
            nn.Linear(128, 2),
        )

    def extract_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts the shared 256-dimensional radio embedding."""
        h = self.input_proj(x)
        h = self.res_block1(h)
        latent = self.trunk_compression(h)
        return latent

    def forward(self, x: torch.Tensor):
        latent = self.extract_latent(x)
        bldg_logits = self.building_head(latent)
        floor_logits = self.floor_head(latent)
        coord_preds = self.coord_head(latent)
        return {
            'bldg_logits': bldg_logits,
            'floor_logits': floor_logits,
            'coord_preds': coord_preds,
            'latent': latent,
        }


class UncertaintyMultiTaskLoss(nn.Module):
    """
    Homoscedastic Uncertainty Weighting for Multi-Task Learning
    Reference: Kendall et al. (CVPR 2018) "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    def __init__(self, init_log_var: float = 0.0):
        super().__init__()
        # Learnable log variances for Building, Floor, and Coordinates
        self.log_var_bldg = nn.Parameter(torch.tensor(init_log_var))
        self.log_var_floor = nn.Parameter(torch.tensor(init_log_var))
        self.log_var_coords = nn.Parameter(torch.tensor(init_log_var))

        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, preds: dict, targets: dict) -> dict:
        bldg_loss = self.ce_loss(preds['bldg_logits'], targets['y_bldg'])
        floor_loss = self.ce_loss(preds['floor_logits'], targets['y_floor'])
        coord_loss = self.huber_loss(preds['coord_preds'], targets['y_coords'])

        # Uncertainty-weighted formulation
        precision_bldg = torch.exp(-self.log_var_bldg)
        precision_floor = torch.exp(-self.log_var_floor)
        precision_coords = torch.exp(-self.log_var_coords)

        total_loss = (
            0.5 * precision_bldg * bldg_loss + 0.5 * self.log_var_bldg +
            0.5 * precision_floor * floor_loss + 0.5 * self.log_var_floor +
            0.5 * precision_coords * coord_loss + 0.5 * self.log_var_coords
        )

        return {
            'total_loss': total_loss,
            'bldg_loss': bldg_loss.item(),
            'floor_loss': floor_loss.item(),
            'coord_loss': coord_loss.item(),
            'weights': {
                'w_bldg': precision_bldg.item(),
                'w_floor': precision_floor.item(),
                'w_coords': precision_coords.item(),
            }
        }

