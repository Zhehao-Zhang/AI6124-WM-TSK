# AI6124 Group 31: WM Forecasting and Trading with a Regularised TSK Neuro-Fuzzy System

Next-day forecasting and trading of Waste Management, Inc. (NYSE: WM), January 2000 to September 2026, with a first-order TSK neuro-fuzzy system trained with DropRule and MC-DropRule uncertainty, benchmarked against MLP and LSTM networks.

The main deliverable is **`AI6124_G31_Final.ipynb`**: code, results and the full report, saved with the outputs of a complete run.

## Contents

| Path | Content |
|---|---|
| `AI6124_G31_Final.ipynb` | Final notebook (report, code and outputs) |
| `data/raw/` | Frozen WM price history from Yahoo Finance, with download metadata |
| `data/raw/extra/` | Frozen S&P 500 (^GSPC), Republic Services (RSG) and VIX (^VIX) histories, with download metadata and checksums |
| `data/gdelt/` | GDELT 1.0 daily event aggregates (`gdelt_daily_events.csv`), the BigQuery query that produced them (`02_daily_features.sql`) and a coverage check (`01_coverage.sql`, `01_coverage_result.csv`) |
| `cache/search/` | Hyperparameter search results: every validation run and the configuration selected for each fold, model, input set and task |
| `WM_Forecasting.ipynb`, `data/processed/`, `wm_data.py` | Data-preparation notebook, prepared `base7` folds and a loader for them |
| `benchmarks.py`, `summarize_benchmarks.py`, `test_benchmarks.py`, `preds/`, `results/` | Stand-alone MLP and LSTM benchmark scripts and their outputs |

## Running the final notebook

```bash
python -m pip install -r requirements.txt
jupyter lab AI6124_G31_Final.ipynb
```

Run all cells from top to bottom in a fresh kernel from the repository folder. With `RUN_SEARCH = False` (the default), the search results in `cache/search/` are reloaded and only the final models are trained, which takes about 5 minutes on a 16-core CPU; `RUN_SEARCH = True` repeats the hyperparameter search (about 1–2 hours). `N_JOBS` sets the number of parallel CPU workers. The notebook writes its prepared base dataset to `data/model_ready/`, which is not tracked.

## Stand-alone benchmark scripts

`python benchmarks.py` (MLP) and `python benchmarks.py lstm` (LSTM) select a configuration on each fold's validation block and save five test predictions per fold in `preds/`; `python summarize_benchmarks.py` writes the summary tables and charts in `results/`, and `python -m unittest test_benchmarks` checks the LSTM sequence alignment. The final notebook re-implements the same MLP and LSTM models and selects all models, including TSK, with one common procedure (its section 5.2).

## Data sources

- Prices: Yahoo Finance, downloaded with `yfinance` and frozen for exact reproduction; used for course research only.
- World events: GDELT 1.0 (Leetaru & Schrodt, 2013), queried in Google BigQuery (`gdelt-bq.full.events`).
