"""Validation-selected MLP and LSTM benchmarks on the prepared WM folds."""

import argparse
import copy
import json
import random
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch import nn

from wm_data import load_all_folds


PATIENCE = 20
MAX_EPOCHS = 300
GRID = {
    "hidden": ((32, 16), (64, 32)),
    "dropout": (0.0, 0.2),
    "weight_decay": (0.0, 1e-3),
    "lr": (1e-3, 3e-3),
}
LSTM_GRID = {
    "L": (5, 10, 20),
    "hidden": (16, 32),
    "weight_decay": (0.0, 1e-3),
    "lr": (1e-3, 3e-3),
}


class MLP(nn.Module):
    def __init__(self, hidden=(32, 16), dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class LSTM(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=hidden, num_layers=1,
                            batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        steps, _ = self.lstm(x)
        return self.output(steps[:, -1]).squeeze(-1)


def make_sequences(X_prev, X_cur, y_cur, L):
    """Use only prior rows plus each current row to predict its target."""
    X_prev, X_cur, y_cur = map(np.asarray, (X_prev, X_cur, y_cur))
    if (isinstance(L, bool) or not isinstance(L, (int, np.integer)) or L < 1
            or X_prev.ndim != 2 or X_cur.ndim != 2
            or X_prev.shape[1] != X_cur.shape[1]
            or y_cur.ndim != 1 or len(X_cur) != len(y_cur)
            or len(X_prev) < L - 1):
        raise ValueError("Invalid sequence inputs or insufficient preceding rows")
    if not len(X_cur):
        return np.empty((0, L, X_cur.shape[1]), dtype=X_cur.dtype), y_cur
    history = np.concatenate((X_prev[-(L - 1):] if L > 1 else X_prev[:0], X_cur))
    windows = np.stack([history[i:i + L] for i in range(len(X_cur))])
    return windows, y_cur


def make_lstm_fold(data, L):
    X_train, y_train = make_sequences(
        data["X_train"][:L - 1], data["X_train"][L - 1:],
        data["y_train"][L - 1:], L,
    )
    X_val, y_val = make_sequences(data["X_train"], data["X_val"], data["y_val"], L)
    X_test, y_test = make_sequences(data["X_val"], data["X_test"], data["y_test"], L)
    return {**data, "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val, "X_test": X_test, "y_test": y_test}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def predict(model, X, y_std):
    model.eval()
    with torch.no_grad():
        return model(torch.as_tensor(X, dtype=torch.float32)).numpy() * y_std


def train_one(model_cfg, data, seed):
    """Fit on train, stop on validation, and return restored model and target scale."""
    set_seed(seed)
    y_std = float(np.std(data["y_train"]))
    if not np.isfinite(y_std) or y_std <= 0:
        raise ValueError("Training targets must have a positive finite standard deviation")

    X_train = torch.as_tensor(data["X_train"], dtype=torch.float32)
    X_val = torch.as_tensor(data["X_val"], dtype=torch.float32)
    y_train = torch.as_tensor(data["y_train"] / y_std, dtype=torch.float32)
    y_val = torch.as_tensor(data["y_val"] / y_std, dtype=torch.float32)
    model = (LSTM(model_cfg["hidden"]) if model_cfg.get("model") == "lstm"
             else MLP(model_cfg["hidden"], model_cfg["dropout"]))
    optimizer = torch.optim.Adam(model.parameters(), lr=model_cfg["lr"],
                                 weight_decay=model_cfg["weight_decay"])
    loss_fn = nn.MSELoss()
    best_loss = float("inf")
    stale = 0
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val).item()
        if not np.isfinite(val_loss):
            raise ValueError("Validation loss is not finite")
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, y_std, float(np.sqrt(best_loss) * y_std)


def evaluate(y_true, y_pred, price):
    """Return model scores, zero-return baseline scores, and error ratios."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    price = np.asarray(price)
    if y_true.shape != y_pred.shape or y_true.shape != price.shape:
        raise ValueError("Targets, predictions and prices must have matching shapes")
    true_price = price * np.exp(y_true)
    pred_price = price * np.exp(y_pred)
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    naive_rmse = float(np.sqrt(np.mean(y_true ** 2)))
    price_rmse = float(np.sqrt(np.mean((pred_price - true_price) ** 2)))
    naive_price_rmse = float(np.sqrt(np.mean((price - true_price) ** 2)))
    direction_accuracy = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
    naive_direction_accuracy = float(np.mean(y_true == 0))
    return {
        "log_return_rmse": rmse,
        "naive_log_return_rmse": naive_rmse,
        "log_return_rmse_ratio_to_naive": rmse / naive_rmse if naive_rmse else float("nan"),
        "direction_accuracy": direction_accuracy,
        "naive_direction_accuracy": naive_direction_accuracy,
        "direction_accuracy_ratio_to_naive": (
            direction_accuracy / naive_direction_accuracy
            if naive_direction_accuracy else float("nan")
        ),
        "price_rmse": price_rmse,
        "naive_price_rmse": naive_price_rmse,
        "price_rmse_ratio_to_naive": (
            price_rmse / naive_price_rmse if naive_price_rmse else float("nan")
        ),
    }


def select_best(model_name, data):
    sequence_folds = ({L: make_lstm_fold(data, L) for L in LSTM_GRID["L"]}
                      if model_name == "lstm" else {})
    if model_name == "lstm":
        configs = (dict(model="lstm", L=L, hidden=hidden,
                        weight_decay=weight_decay, lr=lr)
                   for L, hidden, weight_decay, lr in product(*LSTM_GRID.values()))
    else:
        configs = (dict(hidden=hidden, dropout=dropout,
                        weight_decay=weight_decay, lr=lr)
                   for hidden, dropout, weight_decay, lr in product(*GRID.values()))
    best = None
    for cfg in configs:
        fit_data = sequence_folds[cfg["L"]] if model_name == "lstm" else data
        model, y_std, val_rmse = train_one(cfg, fit_data, seed=0)
        if best is None or val_rmse < best[0]:
            best = (val_rmse, cfg, model, y_std)
    fit_data = sequence_folds[best[1]["L"]] if model_name == "lstm" else data
    return best, fit_data


def save_selection(model_name, rows):
    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(exist_ok=True)
    (output_dir / f"{model_name}_selection.json").write_text(
        json.dumps({"model": model_name, "selection_seed": 0, "folds": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


def run_benchmarks(model_name="mlp"):
    if model_name not in ("mlp", "lstm"):
        raise ValueError("model_name must be 'mlp' or 'lstm'")
    output_dir = Path(__file__).resolve().parent / "preds"
    output_dir.mkdir(exist_ok=True)
    selections = []
    for fold_id, data in load_all_folds().items():
        if not np.allclose(
            data["price_at_forecast_test"] * np.exp(data["y_test"]),
            data["target_price_test"],
        ):
            raise ValueError(f"Fold {fold_id} has misaligned test prices")
        best, fit_data = select_best(model_name, data)
        _, cfg, selected_model, selected_y_std = best
        selections.append({"fold": fold_id, "config": cfg, "validation_rmse": best[0]})
        print(f"fold {fold_id}: selected {cfg}, validation RMSE={best[0]:.6f}")
        for seed in range(5):
            if seed == 0:
                model, y_std = selected_model, selected_y_std
            else:
                model, y_std, _ = train_one(cfg, fit_data, seed)
            y_pred = predict(model, fit_data["X_test"], y_std)
            y_true = data["y_test"]
            price = data["price_at_forecast_test"]
            np.savez_compressed(
                output_dir / f"{model_name}_fold{fold_id}_seed{seed}.npz",
                dates=data["target_dates_test"], y_true=y_true,
                y_pred=y_pred, price=price,
            )
            scores = evaluate(y_true, y_pred, price)
            print(f"  seed {seed}: {scores}")
    save_selection(model_name, selections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mlp", "lstm"), nargs="?", default="mlp")
    run_benchmarks(parser.parse_args().model)
