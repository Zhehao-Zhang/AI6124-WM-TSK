# Knowing When Not to Predict: A Regularised TSK Neuro-Fuzzy System for WM Stock Forecasting and Trading

**AI6124 Neuro Evolution & Fuzzy Intelligence, Group 31** · Zhang Zhehao, Li Yetong

This repository contains the code, data and results of a study of next-day forecasting and trading for Waste Management, Inc. (NYSE: WM), from January 2000 to September 2026. The main model is a first-order Takagi–Sugeno–Kang (TSK) neuro-fuzzy system trained with DropRule, whose training is regularised for the low signal-to-noise ratio of daily returns and whose MC-DropRule variant provides a measure of model uncertainty. MLP and LSTM networks serve as benchmarks.

The complete report, with all code and the outputs of a full run, is the notebook **[`AI6124_G31_Final.ipynb`](AI6124_G31_Final.ipynb)**.

---

## Research questions

| | Question |
|---|---|
| **RQ1** | Is the regularised TSK model at least as accurate as MLP and LSTM on next-day returns, and how close do the models come to the random walk? |
| **RQ2** | Does additional information (S&P 500, Republic Services, VIX, GDELT world events) improve the forecasts? |
| **RQ3** | Which part of the next-day return is predictable: its sign or its size? |
| **RQ4** | Do the forecasts have economic value in trading, compared with strategies that use no prediction? |

## Method

- **Model.** First-order TSK fuzzy system with Gaussian memberships, product t-norm and linear rule consequents; rules are initialised by k-means on the training inputs and all parameters are trained jointly with Adam.
- **Training under a low signal-to-noise ratio.** Standardised target, L2 penalty on the rule slopes and shrinkage of the rule biases towards their mean, so that the forecast stays close to the training mean unless the data support a departure from it.
- **DropRule and MC-DropRule.** Rules are dropped at random during training; at prediction time, 50 stochastic passes give a mean forecast and a spread that measures the disagreement between rules, used as a confidence filter for trading.
- **Benchmarks.** Two-layer MLP and single-layer LSTM (window of 5–20 days), trained with the same inputs, folds, seeds and selection procedure.
- **Trading.** Multiplicative wealth with 0.5% transaction costs; buy-and-hold and a 20/100-day moving-average crossover without prediction; forecast-driven long/flat rules with prediction; and the forecast-bottleneck-free framework of Ang & Quek (2006) as a supplement.

## Evaluation protocol

- **Walk-forward validation.** Five expanding-window folds, each with a 252-day validation block and a 252-day test block; the five test blocks cover 2021-09-23 to 2026-09-30 (1,260 trading days).
- **Hyperparameter selection.** Grid search; the configuration of fold *k* is chosen by the RMSE pooled over the validation blocks of folds 1 to *k* (seeds 0–2). Test data are never used for selection.
- **Repetitions.** Every selected configuration is trained with five seeds; metrics are averaged over seeds within a fold and reported as mean ± SD over the five folds.
- **Inference.** Diebold–Mariano tests (Newey–West variance, 10 lags) on daily squared errors; block-bootstrap intervals for Sharpe-ratio differences.

## Main results

Test RMSE (mean over folds); lower is better.

| Task | Inputs | TSK (best variant) | MLP | LSTM | Reference |
|---|---|---|---|---|---|
| Next-day return | `base7` | 0.011991 | 0.012019 | 0.012000 | 0.011987 (random walk) |
| Next-day return | `ext15` | **0.011968** | 0.012021 | 0.012034 | 0.011987 (random walk) |
| Next-day \|return\| | `base7` | **0.008305** | 0.008432 | 0.008429 | 0.008449 (EWMA) |
| Next-day \|return\| | `ext15` | **0.008258** | 0.008308 | 0.008275 | 0.008449 (EWMA) |

Trading on the test period after 0.5% transaction costs:

| Strategy | Final wealth | Sharpe ratio |
|---|---|---|
| Buy-and-hold | 1.431 | 0.47 |
| Moving average 20/100 | 0.333 | −1.05 |
| Best forecast rule without confidence filter (LSTM, `base7`) | 1.204 | 0.29 |
| TSK + MC-DropRule confidence rule (`base7`) | 1.370 | 0.42 |

In short: no model beats the random walk with WM's own price history, but TSK ranks above both networks; with market inputs, TSK is the only model ahead of the random walk and is significantly more accurate than MLP and LSTM; GDELT features make every model worse; the size of returns is far more predictable than their sign; and no strategy beats buy-and-hold, although the MC-DropRule confidence filter gives the best prediction-based result. See sections 6 and 7 of the notebook for the full tables, tests and discussion.

## Repository structure

```
.
├── AI6124_G31_Final.ipynb        Final report notebook (code, results, discussion; outputs included)
├── requirements.txt              Package versions of the saved run
├── data/
│   ├── raw/                      Frozen WM daily prices (Yahoo Finance) and download metadata
│   │   └── extra/                Frozen S&P 500, RSG and VIX daily prices, metadata and checksums
│   ├── gdelt/                    GDELT 1.0 daily aggregates and the BigQuery queries that produced them
│   └── processed/                Prepared base7 folds used by the stand-alone benchmark scripts
├── cache/
│   └── search/                   Hyperparameter search results: all validation runs and the
│                                 configuration selected per fold, model, input set and task
├── WM_Forecasting.ipynb          Data-preparation notebook for the prepared base7 folds
├── wm_data.py                    Loader for data/processed/
├── benchmarks.py                 Stand-alone MLP and LSTM benchmarks
├── summarize_benchmarks.py       Summary tables and charts for the stand-alone benchmarks
├── test_benchmarks.py            Unit tests for the LSTM sequence construction
├── preds/                        Test predictions of the stand-alone benchmarks
└── results/                      Selections, summaries and charts of the stand-alone benchmarks
```

## Data

| Dataset | Source | Coverage | Use |
|---|---|---|---|
| WM daily OHLCV, dividends, splits | Yahoo Finance (`yfinance`) | 2000-01-03 – 2026-09-30, 6,726 days | Base inputs and target |
| S&P 500 (^GSPC), Republic Services (RSG), VIX (^VIX) | Yahoo Finance | same period, aligned to WM trading days | Market, peer and implied-volatility inputs (`ext15`) |
| GDELT 1.0 events (`gdelt-bq.full.events`) | Google BigQuery | 1999-11-01 – 2026-09-30, one row per day | Seven world-news indicators (z-scores) |

All price files are frozen snapshots with SHA-256 checksums, so that every run uses identical data; the notebook downloads a missing file again from Yahoo Finance, whose adjusted values may differ slightly. The GDELT file can be regenerated with `data/gdelt/02_daily_features.sql` in the BigQuery sandbox; its feature construction (causal 60-day z-scores, handling of two tone-level breaks) is described in section 3.3.2 of the notebook.

## Reproducing the results

1. Create an environment (tested with Python 3.14.6) and install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

2. Open the notebook from the repository folder and run all cells from top to bottom in a fresh kernel:

   ```bash
   jupyter lab AI6124_G31_Final.ipynb
   ```

Two settings in the first code cell control the run:

| Setting | Default | Effect |
|---|---|---|
| `RUN_SEARCH` | `False` | `False` reloads the hyperparameter search from `cache/search/` and trains only the final models (about 5 minutes on a 16-core CPU); `True` repeats the full grid search (about 1–2 hours). |
| `N_JOBS` | `16` | Number of parallel CPU workers (joblib). |

Training is deterministic (fixed seeds, deterministic PyTorch algorithms), so a rerun reproduces the saved results. The notebook writes its prepared base dataset to `data/model_ready/`, which is not tracked.

## Stand-alone benchmark scripts

The scripts in the repository root are the original MLP and LSTM benchmark implementation on the prepared `base7` folds in `data/processed/`:

```bash
python benchmarks.py              # MLP: select per fold, save five test predictions per fold to preds/
python benchmarks.py lstm         # LSTM (sequence lengths 5, 10 and 20)
python summarize_benchmarks.py    # summary tables and charts in results/
python -m unittest test_benchmarks
```

These scripts select each model on its own fold's validation block. The final notebook re-implements the same MLP and LSTM architectures and training, and selects all models, including TSK, with the common pooled procedure described above.

## Authors

- **Zhang Zhehao**: data pipeline, TSK model, experiments and report.
- **Li Yetong**: MLP and LSTM benchmarks.

## References

The notebook lists all references. The key ones are:

- Ang, K. K., & Quek, C. (2006). Stock trading using RSPOP: A novel rough set-based neuro-fuzzy approach. *IEEE Transactions on Neural Networks*, 17(5), 1301–1315.
- Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control. *IEEE Transactions on Systems, Man, and Cybernetics*, SMC-15(1), 116–132.
- Wu, D., Yuan, Y., Huang, J., & Tan, Y. (2020). Optimize TSK fuzzy systems for regression problems: Minibatch gradient descent with regularization, DropRule, and AdaBound (MBGD-RDA). *IEEE Transactions on Fuzzy Systems*, 28(5), 1003–1015.
- Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location, and tone, 1979–2012. *International Studies Association Annual Convention*.

## Data terms

Price data were obtained from Yahoo Finance and are included for the reproduction of this course project only. GDELT data are openly available; please cite Leetaru & Schrodt (2013) when reusing them.
