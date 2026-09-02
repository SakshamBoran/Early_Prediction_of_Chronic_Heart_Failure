"""
train.py — Train the final single-lead CNN-Transformer + signature hybrid model.

Reproduces the training run behind the reported single-lead result
(test AUROC 0.852-0.857 for HFrEF). Requires EchoNext-Mini data already
preprocessed via data_prep/singlelead_prep.py and data_prep/signature_features.py.

Usage:
    python train.py --data_dir /path/to/echonext-singlelead
"""

import argparse
import copy
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

from model import CNNTransformerBackbone, SingleLeadHybridSigModel, augment_ecg, masked_bce_loss

TARGET_NAMES = ["hfref", "rv_dysf", "lvh", "shd"]
# Positive-class prevalence in the training set (used to weight the loss)
POS_RATES = torch.tensor([0.1789, 0.1324, 0.2438, 0.5237])


def znorm(x):
    return (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)


def load_split(data_dir, split):
    X = np.load(f"{data_dir}/leadII_{split}.npy").astype(np.float32)
    TB = np.load(f"{data_dir}/T_B_{split}.npy").astype(np.float32)
    Y = np.load(f"{data_dir}/Y_{split}.npy").astype(np.float32)
    return X, TB, Y


def load_signatures(data_dir, split):
    return np.load(f"{data_dir}/sig_{split}.npy").astype(np.float32)


def train(data_dir, out_dir=".", seed=1, epochs=20, patience=6, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    X_train, TB_train, Y_train = load_split(data_dir, "train")
    X_val, TB_val, Y_val = load_split(data_dir, "val")
    X_train, X_val = znorm(X_train), znorm(X_val)

    sig_train = load_signatures(data_dir, "train")
    sig_val = load_signatures(data_dir, "val")

    pos_weights = ((1 - POS_RATES) / POS_RATES).to(device)

    torch.manual_seed(seed)
    backbone = CNNTransformerBackbone(n_layers=3)
    for m in backbone.transformer.layers:
        m.dropout.p = 0.25
    net = SingleLeadHybridSigModel(backbone.to(device), tabular_dim=TB_train.shape[1]).to(device)

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(TB_train), torch.tensor(sig_train), torch.tensor(Y_train))
    val_ds = TensorDataset(torch.tensor(X_val), torch.tensor(TB_val), torch.tensor(sig_val), torch.tensor(Y_val))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

    optimizer = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auroc, patience_ctr, best_state = -1, 0, None
    for epoch in range(epochs):
        net.train()
        for ecg, tab, sig, y in train_loader:
            ecg, tab, sig, y = ecg.to(device), tab.to(device), sig.to(device), y.to(device)
            ecg = augment_ecg(ecg, training=True)
            optimizer.zero_grad()
            loss = masked_bce_loss(net(ecg, tab, sig), y, pos_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        net.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for ecg, tab, sig, y in val_loader:
                logits = net(ecg.to(device), tab.to(device), sig.to(device))
                all_logits.append(logits.cpu().numpy())
                all_y.append(y.numpy())
        all_logits, all_y = np.concatenate(all_logits), np.concatenate(all_y)
        m = ~np.isnan(all_y[:, 0])
        val_auroc = roc_auc_score(all_y[m, 0], all_logits[m, 0])
        print(f"[hybrid-sig] Epoch {epoch}: val_HFrEF_AUROC={val_auroc:.4f}")

        if val_auroc > best_val_auroc:
            best_val_auroc, patience_ctr, best_state = val_auroc, 0, copy.deepcopy(net.state_dict())
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"[hybrid-sig] Early stopping at epoch {epoch}")
                break

    net.load_state_dict(best_state)
    torch.save(best_state, f"{out_dir}/hybrid_sig_model.pt")
    print(f"[hybrid-sig] Saved. Best val AUROC: {best_val_auroc:.4f}")
    return net


def evaluate(net, data_dir, device=None):
    """Final, single, honest test-set evaluation."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X_test, TB_test, Y_test = load_split(data_dir, "test")
    X_test = znorm(X_test)
    sig_test = load_signatures(data_dir, "test")

    net.eval()
    test_ds = TensorDataset(torch.tensor(X_test), torch.tensor(TB_test), torch.tensor(sig_test), torch.tensor(Y_test))
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    all_logits, all_y = [], []
    with torch.no_grad():
        for ecg, tab, sig, y in test_loader:
            logits = net(ecg.to(device), tab.to(device), sig.to(device))
            all_logits.append(logits.cpu().numpy())
            all_y.append(y.numpy())
    all_logits, all_y = np.concatenate(all_logits), np.concatenate(all_y)
    probs = 1 / (1 + np.exp(-all_logits))

    print("=== FINAL TEST RESULTS ===")
    for i, name in enumerate(TARGET_NAMES):
        m = ~np.isnan(all_y[:, i])
        yt, yp = all_y[m, i], probs[m, i]
        auroc = roc_auc_score(yt, yp)
        auprc = average_precision_score(yt, yp)
        pred = (yp > 0.5).astype(int)
        acc = accuracy_score(yt, pred)
        tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
        print(f"  [{name}] AUROC={auroc:.4f} AUPRC={auprc:.4f} acc={acc:.4f} "
              f"sens={tp/(tp+fn):.4f} spec={tn/(tn+fp):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to preprocessed single-lead data")
    parser.add_argument("--out_dir", default=".", help="Where to save the trained checkpoint")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()

    trained_model = train(args.data_dir, args.out_dir, args.seed, args.epochs, args.patience)
    evaluate(trained_model, args.data_dir)
