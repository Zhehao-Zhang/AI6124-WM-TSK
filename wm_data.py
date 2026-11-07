"""Load the prepared WM forecasting folds without rerunning preprocessing."""

import json
from pathlib import Path

import numpy as np


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data/processed/wm_training_data.npz"


def load_all_folds(data_path=None):
    """Return five folds containing X/y arrays, dates, prices and scaling metadata."""
    path = DEFAULT_DATA_PATH if data_path is None else Path(data_path)
    with np.load(path, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata_json"].item())
        folds = {}
        for fold_id in range(1, metadata["n_splits"] + 1):
            prefix = f"fold_{fold_id}_"
            folds[fold_id] = {
                key.removeprefix(prefix): saved[key].copy()
                for key in saved.files if key.startswith(prefix)
            }
            folds[fold_id]["fold_id"] = fold_id
            folds[fold_id]["scaler"] = metadata["scalers"][str(fold_id)]
    return folds


def load_fold(fold_id, data_path=None):
    """Return a single fold, numbered 1 through 5."""
    if isinstance(fold_id, bool) or not isinstance(fold_id, int):
        raise ValueError("fold_id must be an integer from 1 to 5")
    folds = load_all_folds(data_path)
    if fold_id not in folds:
        raise ValueError(f"fold_id must be from 1 to {len(folds)}")
    return folds[fold_id]
