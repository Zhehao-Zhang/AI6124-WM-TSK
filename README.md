# WM Return Forecasting with TSK and DropRule

AI6124 project on next-trading-day log-return forecasting for Waste Management (NYSE: WM).

## Implementation Status

The notebook implements price inspection, a moving-average trading baseline, supervised feature/target construction, chronological splitting and training-only input standardization. Prepared data are included for immediate model development.

TSK, TSK with DropRule and benchmark for MLP and LSTM training are not implemented yet. Their comparative forecasting and trading results remain to be produced.

## Current Outputs

- 6,726 source observations from 3 January 2000 to 30 September 2026.
- 6,705 supervised samples after excluding 21 initial observations with insufficient history.
- Seven inputs: five lagged daily log returns, 20-day return volatility and change in log volume. The target is the next trading day's log return.
- Five expanding-window folds, each with 252 validation and 252 test observations. Training size increases from 5,193 to 6,201 observations; targets remain unscaled.
- Zero-return naive forecasts have test log-return RMSE of 0.01113–0.01312 and adjusted-close RMSE of USD 1.6852–2.8301 across the five folds. These are baseline results, not trained-model results.

The notebook also contains moving-average trading plots and cumulative wealth for the full source period; these are separate from out-of-sample model evaluation.

## Data and Usage

`data/raw/` contains the frozen source CSV. `data/processed/wm_samples.csv` contains unscaled features, targets and date/price context. `data/processed/wm_training_data.npz` contains all five prepared folds and their preprocessing metadata.

Use Python 3.12 and install dependencies with `python -m pip install -r requirements.txt`.

Continue below the final cell of `WM_Forecasting.ipynb` using `training_data[1]` through `training_data[5]`. Alternatively, develop in a separate Python script or notebook and load the existing arrays:

```python
from wm_data import load_all_folds

training_data = load_all_folds()
data = training_data[1]  # Example fold; evaluate all five folds.
X_train, y_train = data["X_train"], data["y_train"]
X_val, y_val = data["X_val"], data["y_val"]
X_test, y_test = data["X_test"], data["y_test"]
```

`wm_data.py` is optional: it only reads the prepared arrays, dates, prices and scaling metadata. The main notebook does not depend on it; direct NumPy loading is also supported. No preprocessing rerun is required when using the supplied prepared data.

The next implementation step is model definition, training and validation-based selection, followed by evaluation across all five folds and five random seeds. Predicted log returns are converted to adjusted prices using `price_at_forecast * exp(predicted_log_return)`.

## Configuration

Parameters are defined in the notebook's configuration cells:

| Parameters | Default | Effect |
|---|---|---|
| `FAST_WINDOW`, `SLOW_WINDOW` | 20, 100 | Moving-average trading baseline only; does not change model inputs. |
| `TRANSACTION_RATE` | 0.005 | Trading-cost assumption for the baseline. |
| `N_RETURN_LAGS`, `VOLATILITY_WINDOW` | 5, 20 | Feature construction; changing either requires regenerating prepared data. |
| `N_SPLITS`, `VALIDATION_DAYS`, `TEST_DAYS` | 5, 252, 252 | Chronological evaluation boundaries; changing them requires regenerating prepared data. |
| `MODEL_SEEDS` | 0–4 | Recorded for subsequent repeated model training; no models have been trained yet. |
| Ticker, dates, raw filename and source checksum | Frozen WM snapshot | Define the input dataset; changing ticker/date constants alone does not replace the cached CSV. |

Use the supplied folds unchanged for comparable baseline experiments. For feature or split studies, update the relevant settings, rerun the notebook and save a separately named dataset; `load_all_folds(data_path=...)` can load that archive. Reuse each fold's fitted scaler and select model settings on validation data.

A different ticker or date range requires its own raw snapshot, filenames and matching source checksum. To reproduce the current dataset, run the notebook from top to bottom with the included frozen CSV.
