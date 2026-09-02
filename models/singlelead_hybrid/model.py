"""
model.py — Single-lead CNN-Transformer + path-signature hybrid model.

Architecture used in the final reported single-lead result (test AUROC 0.852-0.857
for HFrEF). Extracted verbatim from the working training notebook, with only the
data-path and dependency setup moved to separate files.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CNNTransformerBackbone(nn.Module):
    """1D CNN stem + Transformer encoder for the raw lead-II waveform."""

    def __init__(self, d_model=128, n_heads=4, n_layers=3, out_dim=256):
        super().__init__()
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7), nn.BatchNorm1d(32), nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4), nn.BatchNorm1d(64), nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=7, stride=4, padding=3), nn.BatchNorm1d(d_model), nn.GELU(),
        )
        self.pos_embed = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=0.15, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

    def forward(self, x):
        feats = self.cnn_stem(x.unsqueeze(1))
        feats = feats.transpose(1, 2)
        T = feats.shape[1]
        feats = feats + self.pos_embed[:, :T, :]
        out = self.transformer(feats)
        return self.out_proj(out.mean(dim=1))


class SingleLeadHybridSigModel(nn.Module):
    """Final model: waveform (CNN-Transformer) + tabular + 23 path-signature features,
    fused and passed through 4 multi-task heads (hfref, rv_dysf, lvh, shd)."""

    def __init__(self, backbone, tabular_dim, sig_dim=23, feat_dim=256, hidden=128, n_targets=4):
        super().__init__()
        self.backbone = backbone
        self.tabular_mlp = nn.Sequential(nn.Linear(tabular_dim, 32), nn.ReLU(), nn.Dropout(0.25))
        self.sig_mlp = nn.Sequential(nn.Linear(sig_dim, 32), nn.ReLU(), nn.Dropout(0.25))
        fused_dim = feat_dim + 32 + 32
        self.head_trunk = nn.Sequential(nn.Linear(fused_dim, hidden), nn.ReLU(), nn.Dropout(0.35))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_targets)])

    def forward(self, ecg, tabular, sig):
        feats = self.backbone(ecg)
        tab = self.tabular_mlp(tabular)
        s = self.sig_mlp(sig)
        fused = torch.cat([feats, tab, s], dim=1)
        h = self.head_trunk(fused)
        return torch.cat([head(h) for head in self.heads], dim=1)


def augment_ecg(x, training=True):
    """Training-time-only augmentation: random time-shift, amplitude scaling, Gaussian noise."""
    if not training:
        return x
    B = x.shape[0]
    shift = torch.randint(-20, 21, (1,)).item()
    x = torch.roll(x, shifts=shift, dims=1)
    scale = 0.9 + 0.2 * torch.rand(B, 1, device=x.device)
    x = x * scale
    noise = torch.randn_like(x) * 0.02
    x = x + noise
    return x


def masked_bce_loss(logits, targets, pos_weights):
    """Class-weighted BCE loss with NaN-masking for missing labels (e.g. null LVEF)."""
    losses = []
    for i in range(targets.shape[1]):
        col_t, col_l = targets[:, i], logits[:, i]
        mask = ~torch.isnan(col_t)
        if mask.sum() == 0:
            continue
        losses.append(F.binary_cross_entropy_with_logits(col_l[mask], col_t[mask], pos_weight=pos_weights[i]))
    return torch.stack(losses).mean()
