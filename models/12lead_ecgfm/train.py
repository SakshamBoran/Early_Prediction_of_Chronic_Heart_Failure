"""
train.py — Fine-tune ECG-FM on the full 72,475-patient EchoNext-Mini training set.

Reproduces the training run behind the reported 12-lead result
(confirmed test AUROC 0.908 for HFrEF). Requires:
  - fairseq-signals installed from source (github.com/Jwoo5/fairseq-signals)
  - The ECG-FM pretrained checkpoint (mimic_iv_ecg_physionet_pretrained.pt),
    downloadable via huggingface_hub from repo_id="wanglab/ecg-fm"
  - EchoNext-Mini data already prepared (see data_prep/prepare_full72k.py)

Usage:
    python train.py --data_dir /path/to/echo_72k --ckpt_path /path/to/ecgfm_checkpoint.pt
"""

import argparse
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

from model import EchoNextDataset12L, MultiTargetHFModel, masked_bce_loss

TARGET_NAMES = ["hfref", "rv_dysf", "lvh", "shd"]
# Positive-class prevalence in the training set (used to weight the loss)
POS_RATES = torch.tensor([0.1789, 0.1324, 0.2438, 0.5237])


def build_ecgfm_backbone(fairseq_signals_path, ckpt_path):
    """Loads the pretrained ECG-FM backbone via fairseq-signals."""
    sys.path.insert(0, fairseq_signals_path)
    from fairseq_signals.models import build_model_from_checkpoint
    return build_model_from_checkpoint(checkpoint_path=ckpt_path)


def train(data_dir, ckpt_path, fairseq_signals_path="/content/fairseq-signals",
          out_dir=".", epochs=8, patience=3, batch_size=32, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    train_ds = EchoNextDataset12L(f"{data_dir}/X_train_full.npy", f"{data_dir}/T_train_full.npy", f"{data_dir}/Y_train_full.npy")
    val_ds = EchoNextDataset12L(f"{data_dir}/X_val.npy", f"{data_dir}/T_val.npy", f"{data_dir}/Y_val.npy")
    print(f"train={len(train_ds)} val={len(val_ds)}")

    ecgfm_backbone = build_ecgfm_backbone(fairseq_signals_path, ckpt_path)
    net = MultiTargetHFModel(ecgfm_backbone).to(device)
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in net.parameters())
    print(f"Trainable: {n_trainable/1e6:.1f}M / {n_total/1e6:.1f}M")

    pos_weights = ((1 - POS_RATES) / POS_RATES).to(device)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    backbone_params = [p for p in net.backbone.parameters() if p.requires_grad]
    new_params = list(net.tabular_mlp.parameters()) + list(net.head_trunk.parameters()) + list(net.heads.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-6},
        {"params": new_params, "lr": 1e-4},
    ])

    best_val_auroc, patience_ctr = -1, 0
    for epoch in range(epochs):
        net.train()
        train_losses = []
        t0 = time.time()
        for i, (ecg, t, y) in enumerate(train_loader):
            ecg, t, y = ecg.to(device), t.to(device), y.to(device)
            optimizer.zero_grad()
            loss = masked_bce_loss(net(ecg, t), y, pos_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())
            if i % 200 == 0:
                print(f"  epoch {epoch}, batch {i}/{len(train_loader)}, loss={loss.item():.4f}, elapsed={time.time()-t0:.0f}s")

        net.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for ecg, t, y in val_loader:
                logits = net(ecg.to(device), t.to(device))
                all_logits.append(logits.cpu().numpy())
                all_y.append(y.numpy())
        all_logits, all_y = np.concatenate(all_logits), np.concatenate(all_y)
        hfref_mask = ~np.isnan(all_y[:, 0])
        val_auroc = roc_auc_score(all_y[hfref_mask, 0], all_logits[hfref_mask, 0])
        print(f"Epoch {epoch}: train_loss={np.mean(train_losses):.4f}  val_HFrEF_AUROC={val_auroc:.4f}")

        torch.save(net.state_dict(), f"{out_dir}/ecgfm_full72k_epoch{epoch}.pt")
        if val_auroc > best_val_auroc:
            best_val_auroc, patience_ctr = val_auroc, 0
            torch.save(net.state_dict(), f"{out_dir}/ecgfm_full72k_BEST.pt")
            print("  -> saved new best model")
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    print(f"\nBest val HFrEF AUROC: {best_val_auroc:.4f}")
    return net, train_ds, val_ds


def evaluate(net, data_dir, device=None, batch_size=32):
    """Final, single, honest test-set evaluation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    test_ds = EchoNextDataset12L(f"{data_dir}/X_test.npy", f"{data_dir}/T_test.npy", f"{data_dir}/Y_test.npy")
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    net.eval()
    all_logits, all_y = [], []
    with torch.no_grad():
        for ecg, t, y in test_loader:
            logits = net(ecg.to(device), t.to(device))
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())
    all_logits, all_y = np.concatenate(all_logits), np.concatenate(all_y)
    all_probs = 1 / (1 + np.exp(-all_logits))

    print("=== FINAL TEST SET RESULTS ===")
    for i, name in enumerate(TARGET_NAMES):
        mask = ~np.isnan(all_y[:, i])
        y_true = all_y[mask, i]
        y_prob = all_probs[mask, i]
        auroc = roc_auc_score(y_true, y_prob)
        auprc = average_precision_score(y_true, y_prob)
        y_pred = (y_prob > 0.5).astype(int)
        acc = accuracy_score(y_true, y_pred)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sens, spec = tp / (tp + fn), tn / (tn + fp)
        print(f"[{name}] n={mask.sum()} pos_rate={y_true.mean():.3f}")
        print(f"  AUROC={auroc:.4f} AUPRC={auprc:.4f} acc={acc:.4f}")
        print(f"  sensitivity={sens:.4f} specificity={spec:.4f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to prepared 12-lead data (X_train_full.npy etc.)")
    parser.add_argument("--ckpt_path", required=True, help="Path to ECG-FM pretrained checkpoint")
    parser.add_argument("--fairseq_signals_path", default="/content/fairseq-signals")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    trained_model, _, _ = train(
        args.data_dir, args.ckpt_path, args.fairseq_signals_path, args.out_dir,
        args.epochs, args.patience
    )
    evaluate(trained_model, args.data_dir)
