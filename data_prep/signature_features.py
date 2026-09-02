"""
signature_features.py — Path-signature (log-signature) feature extraction.

Computes the 23 hardware-realistic signature features from a single lead-II
ECG waveform: 5-dim (time, voltage) morphology signature, 14-dim lead-lag
self-embedding (captures rhythm regularity), and 4 simple scalar statistics.

Requires: pip install iisignature
"""

import numpy as np
import iisignature as isig

DEPTH = 3
PREP = isig.prepare(2, DEPTH)   # 2D path (time, voltage) -> 5-dim log-signature
PREP3 = isig.prepare(3, DEPTH)  # 3D path (time, voltage, lagged voltage) -> 14-dim log-signature


def compute_signature_features(x_leadII):
    """
    Compute the 23 signature features for one lead-II ECG segment.

    Parameters
    ----------
    x_leadII : np.ndarray, shape (2500,)
        Raw (not pre-normalized) lead-II waveform, 10s at 250Hz.

    Returns
    -------
    np.ndarray, shape (23,)
        [5 t_II features, 14 leadlag_II features, 4 scalar stats
        (max, min, std, QRS-region range)]
    """
    x = (x_leadII - x_leadII.mean()) / (x_leadII.std() + 1e-6)
    t = np.linspace(0, 1, len(x))

    path_t_II = np.stack([t, x], axis=1)
    feat_t_II = isig.logsig(path_t_II, PREP)

    lag = np.roll(x, 2)
    lag[:2] = x[0]
    path_leadlag = np.stack([t, x, lag], axis=1)
    feat_leadlag = isig.logsig(path_leadlag, PREP3)

    # QRS-region range, with boundary clamping (fixes an edge case where the
    # detected peak sits near the start/end of the window)
    peak_idx = np.argmax(x)
    lo = max(0, peak_idx - 10)
    hi = min(len(x), peak_idx + 10)
    qrs_region = x[lo:hi]
    if len(qrs_region) == 0:
        qrs_region = x

    scalars = np.array([x.max(), x.min(), x.std(), qrs_region.max() - qrs_region.min()])

    return np.concatenate([feat_t_II, feat_leadlag, scalars])


def extract_all(X, tag="split"):
    """Extract signature features for an entire array of ECGs, shape (N, 2500)."""
    import time
    t0 = time.time()
    feats = np.zeros((len(X), 23), dtype=np.float32)
    for i in range(len(X)):
        feats[i] = compute_signature_features(X[i])
        if i % 10000 == 0:
            print(f"  [{tag}] {i}/{len(X)}")
    print(f"[{tag}] done in {time.time()-t0:.1f}s")
    return feats


if __name__ == "__main__":
    # quick smoke test
    dummy_ecg = np.random.randn(2500).astype(np.float32)
    feats = compute_signature_features(dummy_ecg)
    assert feats.shape == (23,), f"Expected shape (23,), got {feats.shape}"
    print("Signature feature extraction OK:", feats.shape)
