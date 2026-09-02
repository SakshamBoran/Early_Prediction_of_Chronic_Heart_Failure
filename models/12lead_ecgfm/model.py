"""
model.py — 12-lead ECG-FM fine-tuned model.

Architecture used in the final reported 12-lead result (test AUROC 0.908 for HFrEF,
trained on the full 72,475-patient set). Backbone is ECG-FM (McKeen et al., 2025),
a wav2vec2-style transformer pretrained on 1.5M ECGs; fine-tuned here with partial
layer freezing and tabular fusion.

Requires: fairseq-signals (https://github.com/Jwoo5/fairseq-signals)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import resample_poly


class EchoNextDataset12L(Dataset):
    """Loads 12-lead EchoNext-Mini waveforms + tabular covariates + labels.
    Resamples 250Hz -> 500Hz to match ECG-FM's expected input rate."""

    def __init__(self, x_path, t_path, y_path):
        self.X = np.load(x_path, mmap_mode="r")   # (N, 1, 2500, 12)
        self.T = np.load(t_path)                    # (N, 7)
        self.Y = np.load(y_path)                     # (N, 4)
        assert len(self.X) == len(self.T) == len(self.Y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        ecg = np.array(self.X[idx, 0])               # (2500, 12)
        ecg = resample_poly(ecg, 2, 1, axis=0)        # -> (5000, 12), 250Hz -> 500Hz
        ecg = ecg.T.astype(np.float32)                # (12, 5000) channels-first
        t = self.T[idx].astype(np.float32)
        y = self.Y[idx].astype(np.float32)
        return torch.tensor(ecg), torch.tensor(t), torch.tensor(y)


class MultiTargetHFModel(nn.Module):
    """ECG-FM backbone (partially frozen) + tabular fusion + 4 multi-task heads
    (hfref, rv_dysf, lvh, shd)."""

    def __init__(self, ecgfm_backbone, tabular_dim=7, hidden=256, n_targets=4, freeze_backbone_frac=0.7):
        super().__init__()
        self.backbone = ecgfm_backbone
        params = list(self.backbone.parameters())
        n_freeze = int(len(params) * freeze_backbone_frac)
        for p in params[:n_freeze]:
            p.requires_grad = False
        print(f"Froze {n_freeze}/{len(params)} backbone parameter tensors")

        embed_dim = 768  # ECG-FM's output embedding dimension
        self.tabular_mlp = nn.Sequential(nn.Linear(tabular_dim, 32), nn.ReLU(), nn.Dropout(0.1))
        fused_dim = embed_dim + 32
        self.head_trunk = nn.Sequential(nn.Linear(fused_dim, hidden), nn.ReLU(), nn.Dropout(0.2))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_targets)])

    def forward(self, ecg, tabular):
        out = self.backbone(source=ecg)
        pooled = out["features"].mean(dim=1)
        tab = self.tabular_mlp(tabular)
        fused = torch.cat([pooled, tab], dim=1)
        h = self.head_trunk(fused)
        return torch.cat([head(h) for head in self.heads], dim=1)


def masked_bce_loss(logits, targets, pos_weights):
    """Class-weighted BCE loss with NaN-masking for missing labels (e.g. null LVEF)."""
    losses = []
    for i in range(targets.shape[1]):
        col_t = targets[:, i]
        col_l = logits[:, i]
        mask = ~torch.isnan(col_t)
        if mask.sum() == 0:
            continue
        losses.append(F.binary_cross_entropy_with_logits(col_l[mask], col_t[mask], pos_weight=pos_weights[i]))
    return torch.stack(losses).mean()
