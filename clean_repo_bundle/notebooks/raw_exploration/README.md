# Raw Exploration Notebooks

These are the original Kaggle research notebooks, preserved as-is, including:
- Failed cells (session resets during long-running training, mid-experiment)
- Duplicate/repeated setup cells (from recovering after Kaggle session interruptions)
- Multiple architectures tested in sequence before arriving at the final model
  (pure CNN -> CNN-Transformer hybrid -> hybrid + augmentation -> hybrid + signatures)

They are kept here **for transparency and reproducibility of the actual experimental
process**, not as clean, directly-runnable scripts.

For clean, runnable code implementing only the final reported models, see:
- `models/singlelead_hybrid/` — final single-lead model (test AUROC 0.852-0.857)
- `models/12lead_ecgfm/` — final 12-lead model (test AUROC 0.908)

## Files
- `singlelead_exploration.ipynb` — full single-lead architecture search and final training
- `12lead_setup.ipynb` — 12-lead environment setup and connectivity verification
