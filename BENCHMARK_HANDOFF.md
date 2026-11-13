# WM 预测基准交接

## 已完成

- 完成 proposal 要求的 MLP 基准，以及可选的单层 LSTM 基准。两者预测相同的次日对数收益，使用同一套 7 维特征和 5 个扩展窗口 fold。
- 每个 fold 用验证集选择配置并早停，再以 seed 0–4 各进行一次测试预测。MLP 和 LSTM 各有 25 份逐日预测文件。
- 已计算对数收益 RMSE、方向准确率、调整后收盘价 RMSE 及其相对零收益 naive 的比值，并生成逐次、汇总表和图。

## 共用数据与评估口径

```python
from wm_data import load_all_folds
folds = load_all_folds()  # fold 1–5
```

每个 fold 提供 `X_train/y_train`、`X_val/y_val`、`X_test/y_test`。`X` 是只用该 fold 训练段拟合的标准化特征，`y` 是原尺度对数收益。验证段和测试段各有 252 行；不要用测试指标调参。

预测价格为 `price_at_forecast_test * exp(y_pred)`，其中 `price_at_forecast_test` 是目标日前一交易日的**调整后**收盘价。零收益 naive 预测 `y_pred=0`，因此预测价格等于前一日调整后收盘价。价格 RMSE 比值小于 1 才表示优于 naive。

## 文件与运行

| 文件 | 内容 |
|---|---|
| `benchmarks.py` | MLP/LSTM 训练、验证选参和测试评估；`python benchmarks.py` 跑 MLP，`python benchmarks.py lstm` 跑 LSTM。 |
| `preds/{model}_fold{k}_seed{s}.npz` | 每次运行的 `dates`、`y_true`、`y_pred`、`price`；`dates` 是目标日期，`price` 是预测起点的调整后价格。 |
| `results/{model}_selection.json` | 每个 fold 用 seed 0 选出的配置和验证 RMSE。 |
| `results/benchmark_runs.csv` | 每个模型 × fold × seed 的测试指标。 |
| `results/benchmark_summary.csv` | 每个 fold 的五种子均值±样本标准差，以及跨 fold 均值。 |
| `results/*.png` | 收益 RMSE 和方向准确率对比图。 |

预测文件更新后，运行 `python summarize_benchmarks.py` 刷新逐次指标、汇总表和图。`test_benchmarks.py` 检查 LSTM 窗口对齐。

## 当前结果摘要

| 模型 | 跨 fold 平均收益 RMSE | 方向准确率 | 价格 RMSE／naive |
|---|---:|---:|---:|
| MLP | 0.012019 | 50.22% | 1.0031 |
| LSTM | 0.011999 | 49.56% | 1.0012 |

这两个价格误差比值均略高于 1，应如实报告为未超过零收益 naive。跨 fold 数字是五个 fold 均值的平均；同一 fold 的五个 seed 共用测试日期。

## 接入 TSK／DropRule 结果

使用同一批 fold、测试日期和 seed 0–4。若希望复用现有汇总脚本，按 `preds/tsk_fold{k}_seed{s}.npz` 或 `preds/tsk_droprule_fold{k}_seed{s}.npz` 命名，并保存与上表完全相同的四个字段；`y_pred` 必须是**原尺度对数收益**。`dates`、`y_true`、`price` 直接取该 fold 的 `target_dates_test`、`y_test`、`price_at_forecast_test`，避免重新计算或更改精度。五个 fold × 五个 seed 齐全后，汇总脚本会自动纳入新模型。

