# WM Return Forecasting with TSK and DropRule

AI6124 project on next-trading-day log-return forecasting for Waste Management (NYSE: WM), comparing TSK, TSK with DropRule, and an MLP baseline. Data preparation is complete; model training and comparative evaluation are pending.

## Data

- `data/raw/WM_2000-01-01_2026-09-30_raw.csv`: frozen Yahoo Finance observations.
- `data/processed/wm_samples.csv`: 6,705 supervised samples with unscaled features, targets, dates and price context.
- `data/processed/wm_training_data.npz`: all five prepared training/validation/test folds, feature names, dates, prices and preprocessing metadata.

The seven inputs are five lagged daily log returns, 20-day return volatility and change in log volume. The target is the next trading day's log return. Adjusted closing-price predictions are reconstructed as `price_at_forecast * exp(predicted_log_return)`.

Five expanding-window folds use 252 trading days each for validation and testing. Inputs are standardized using each fold's training statistics; targets remain unscaled. Planned experiments use five random seeds per configuration.

## Usage

Use Python 3.12. Install dependencies with `python -m pip install -r requirements.txt`. The prepared arrays can be loaded directly without rerunning preprocessing:

```python
import numpy as np

with np.load("data/processed/wm_training_data.npz", allow_pickle=False) as saved:
    training_data = {
        i: {key.removeprefix(f"fold_{i}_"): saved[key]
            for key in saved.files if key.startswith(f"fold_{i}_")}
        for i in range(1, 6)
    }

data = training_data[1]  # Example fold; evaluate all five folds.
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
```

## Reproduction

Run `WM_Forecasting.ipynb` in a Jupyter-compatible environment to regenerate the prepared data from the frozen CSV. The final cell exposes the same five folds. Planned evaluation reports log-return RMSE, directional accuracy and reconstructed adjusted-close RMSE against an unchanged-price baseline.
