"""Summarize saved benchmark predictions and draw fold comparisons."""

import csv
import os
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wm_mpl_cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "wm_font_cache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from benchmarks import evaluate


ROOT = Path(__file__).resolve().parent
METRICS = ("log_return_rmse", "direction_accuracy", "price_rmse_ratio_to_naive")
RUN_METRICS = (
    "log_return_rmse", "naive_log_return_rmse", "log_return_rmse_ratio_to_naive",
    "direction_accuracy", "price_rmse", "naive_price_rmse",
    "price_rmse_ratio_to_naive",
)
COLORS = ("#536FE5", "#16A9A1", "#A47AE4", "#EEA66A")


def load_scores():
    scores = defaultdict(dict)
    references = {}
    paths = sorted((ROOT / "preds").glob("*.npz"))
    if not paths:
        raise ValueError("No prediction files found in preds/")
    for path in paths:
        match = re.fullmatch(r"(.+)_fold(\d+)_seed(\d+)\.npz", path.name)
        if not match:
            raise ValueError(f"Unexpected prediction filename: {path.name}")
        model, fold, seed = match.group(1), int(match.group(2)), int(match.group(3))
        with np.load(path, allow_pickle=False) as saved:
            if set(saved.files) != {"dates", "y_true", "y_pred", "price"}:
                raise ValueError(f"Unexpected fields in {path.name}")
            dates, y_true, y_pred, price = (saved[key].copy() for key in
                                           ("dates", "y_true", "y_pred", "price"))
        if not len(y_true) or not np.isfinite(y_pred).all():
            raise ValueError(f"Empty or nonfinite predictions in {path.name}")
        reference = (dates, y_true, price)
        if fold in references and any(not np.array_equal(a, b)
                                      for a, b in zip(reference, references[fold])):
            raise ValueError(f"Fold {fold} has inconsistent dates, targets or prices")
        references[fold] = reference
        if seed in scores[(model, fold)]:
            raise ValueError(f"Duplicate seed {seed} for {model} fold {fold}")
        scores[(model, fold)][seed] = evaluate(y_true, y_pred, price)

    models = sorted({model for model, _ in scores})
    folds = sorted(references)
    for model in models:
        for fold in folds:
            if set(scores[(model, fold)]) != set(range(5)):
                raise ValueError(f"Expected seeds 0–4 for {model} fold {fold}")
    return scores, references, models, folds


def summarize(scores, models, folds):
    rows = []
    for model in models:
        fold_rows = []
        for fold in folds:
            row = {"model": model, "fold": str(fold), "sd_basis": "5 seeds"}
            for metric in METRICS:
                values = [scores[(model, fold)][seed][metric] for seed in range(5)]
                row[f"{metric}_mean"] = float(np.mean(values))
                row[f"{metric}_sd"] = float(np.std(values, ddof=1))
            rows.append(row)
            fold_rows.append(row)
        overall = {"model": model, "fold": "overall", "sd_basis": "fold means"}
        for metric in METRICS:
            values = [row[f"{metric}_mean"] for row in fold_rows]
            overall[f"{metric}_mean"] = float(np.mean(values))
            overall[f"{metric}_sd"] = float(np.std(values, ddof=1))
        rows.append(overall)
    return rows


def save_table(rows, output):
    fields = ["model", "fold", "sd_basis"]
    for metric in METRICS:
        fields.extend((f"{metric}_mean", f"{metric}_sd"))
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_runs(scores, references, models, folds, output):
    fields = ["model", "fold", "seed", *RUN_METRICS, "always_up_direction_accuracy"]
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for model in models:
            for fold in folds:
                always_up = float(np.mean(references[fold][1] > 0))
                for seed in range(5):
                    metrics = scores[(model, fold)][seed]
                    writer.writerow({"model": model, "fold": fold, "seed": seed,
                                     **{key: metrics[key] for key in RUN_METRICS},
                                     "always_up_direction_accuracy": always_up})


def plot_scores(scores, references, models, folds, output_dir):
    x = np.arange(len(folds))
    ink, muted, grid = "#25344A", "#64748B", "#E8EDF5"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.titleweight": "semibold", "axes.titlesize": 17,
        "axes.labelcolor": ink, "text.color": ink,
        "xtick.color": muted, "ytick.color": muted,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.spines.bottom": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })

    def style(ax):
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=grid, linewidth=0.9)
        ax.tick_params(length=0, pad=9)

    width = 0.72 / (len(models) + 1)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    style(ax)
    for j, model in enumerate(models):
        means = [np.mean([scores[(model, fold)][s]["log_return_rmse"]
                          for s in range(5)]) for fold in folds]
        sds = [np.std([scores[(model, fold)][s]["log_return_rmse"]
                       for s in range(5)], ddof=1) for fold in folds]
        bars = ax.bar(x - 0.36 + width * (j + 0.5), means, width, yerr=sds,
                      label=model.upper(), color=COLORS[j % len(COLORS)],
                      error_kw={"ecolor": ink, "elinewidth": 1, "capsize": 3})
        ax.bar_label(bars, fmt="%.5f", fontsize=8, color=muted, padding=6)
    naive = [np.sqrt(np.mean(references[fold][1] ** 2)) for fold in folds]
    bars = ax.bar(x - 0.36 + width * (len(models) + 0.5), naive, width,
                  label="Zero-return baseline", color="#CDD7E4")
    ax.bar_label(bars, fmt="%.5f", fontsize=8, color=muted, padding=6)
    ax.set(xticks=x, xticklabels=[f"Fold {fold}" for fold in folds],
           ylabel="Log-return RMSE")
    ax.set_title("Next-day return prediction error", loc="left", pad=24)
    ax.set_ylim(0, max(naive) * 1.3)
    ax.legend(frameon=False, ncol=len(models) + 1, loc="upper right")
    fig.tight_layout(pad=2)
    fig.savefig(output_dir / "rmse_comparison.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    style(ax)
    width = 0.7 / len(models)
    for j, model in enumerate(models):
        means = [np.mean([scores[(model, fold)][s]["direction_accuracy"]
                          for s in range(5)]) for fold in folds]
        sds = [np.std([scores[(model, fold)][s]["direction_accuracy"]
                       for s in range(5)], ddof=1) for fold in folds]
        ax.bar(x - 0.35 + width * (j + 0.5), means, width, yerr=sds,
               label=model.upper(), color=COLORS[j % len(COLORS)],
               error_kw={"ecolor": ink, "elinewidth": 1, "capsize": 3})
    always_up = [np.mean(references[fold][1] > 0) for fold in folds]
    ax.plot(x, always_up, linestyle=(0, (5, 3)), marker="o", linewidth=2,
            color=ink, markerfacecolor="white", markeredgewidth=1.8,
            markersize=7, label="Always up baseline", zorder=5)
    ax.set(xticks=x, xticklabels=[f"Fold {fold}" for fold in folds],
           ylabel="Directional accuracy")
    ax.set_title("Next-day direction accuracy", loc="left", pad=24)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_ylim(0, max(0.68, max(always_up) + 0.08))
    ax.legend(frameon=False, ncol=len(models) + 1, loc="upper right")
    fig.tight_layout(pad=2)
    fig.savefig(output_dir / "direction_accuracy.png", dpi=200)
    plt.close(fig)


def main():
    scores, references, models, folds = load_scores()
    rows = summarize(scores, models, folds)
    output_dir = ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    save_table(rows, output_dir / "benchmark_summary.csv")
    save_runs(scores, references, models, folds, output_dir / "benchmark_runs.csv")
    plot_scores(scores, references, models, folds, output_dir)
    for row in rows:
        print(row["model"], row["fold"], " ".join(
            f"{metric}: {row[f'{metric}_mean']:.6f} ± {row[f'{metric}_sd']:.6f}"
            for metric in METRICS))


if __name__ == "__main__":
    main()
