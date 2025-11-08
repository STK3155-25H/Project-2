# optimizer_sweep.py
# Scientific comparison of gradient-descent variants on Runge regression
# - Sweeps optimizers and learning rates
# - Compares SGD (mini-batch) vs full-batch ("plain")
# - Saves detailed per-epoch CSV + summary CSV (mean of last N epochs)
# - Ready for later plotting (heatmaps, curves)

from __future__ import annotations
import os
import sys
import json
import math
import time
import argparse
import itertools
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Callable

import pandas as pd
import autograd.numpy as np

# --- Import your provided components ---
# Assumes these modules are discoverable in PYTHONPATH / same repo layout
from src.activation_functions import sigmoid, RELU, LRELU, tanh
from src.cost_functions import CostOLS
from src.scheduler import Constant, Momentum, Adagrad, AdagradMomentum, RMS_prop, Adam
from src.FFNN import FFNN

# ------------------------- Data Generation -------------------------

def runge(x: np.ndarray, noise_std: float = 0.03) -> np.ndarray:
    """Runge function with optional Gaussian noise."""
    noise = np.random.normal(0.0, noise_std, size=x.shape)
    return 1.0 / (1.0 + 25.0 * x**2) + noise

def make_dataset(n: int = 1000,
                 x_low: float = -1.0,
                 x_high: float = 1.0,
                 noise_std: float = 0.03,
                 seed: int | None = 314) -> Tuple[np.ndarray, np.ndarray]:
    if seed is not None:
        np.random.seed(seed)
    X = np.linspace(x_low, x_high, n).reshape(-1, 1)
    y = runge(X, noise_std=noise_std).reshape(-1, 1)
    return X, y

def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int | None = 314):
    n = X.shape[0]
    idx = np.arange(n)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    k = int(math.floor((1.0 - val_frac) * n))
    tr_idx, va_idx = idx[:k], idx[k:]
    return X[tr_idx], y[tr_idx], X[va_idx], y[va_idx]

def standardize(X_train: np.ndarray, X_val: np.ndarray):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-12
    return (X_train - mu) / sd, (X_val - mu) / sd

# ------------------------- Experiment Core -------------------------

def default_arch(input_dim: int = 1, output_dim: int = 1) -> Tuple[int, ...]:
    # A modest architecture that learns Runge well without exploding
    return (input_dim, 64, 64, output_dim)

def build_scheduler(name: str, eta: float):
    """Construct scheduler with sensible defaults per method."""
    name = name.lower()
    if name == "constant":          # Full-batch or SGD w/ fixed LR
        return Constant(eta)
    if name == "momentum":
        return Momentum(eta=eta, momentum=0.9)
    if name == "adagrad":
        return Adagrad(eta=eta)
    if name == "adagradmomentum":
        return AdagradMomentum(eta=eta, momentum=0.9)
    if name == "rmsprop":
        return RMS_prop(eta=eta, rho=0.9)
    if name == "adam":
        return Adam(eta=eta, rho=0.9, rho2=0.999)
    raise ValueError(f"Unknown optimizer: {name}")

def run_one(
    method: str,
    eta: float,
    batch_mode: str,
    seed: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batches_sgd: int,
    lam_l1: float,
    lam_l2: float,
    arch: Tuple[int, ...],
) -> Dict[str, np.ndarray | float | str | int]:
    """
    Trains a single configuration and returns all per-epoch metrics for logging.
    """
    # Model
    model = FFNN(
        dimensions=arch,
        hidden_func=RELU,        # good baseline (you can change to LRELU/tanh if you want)
        output_func=lambda x: x, # regression
        cost_func=CostOLS,
        seed=seed,
    )
    # Scheduler (same instance is deep-copied per layer inside FFNN.fit)
    scheduler = build_scheduler(method, eta)
    # Batches
    if batch_mode == "sgd":
        batches = batches_sgd
    elif batch_mode == "full":
        batches = 1
    else:
        raise ValueError("batch_mode must be 'sgd' or 'full'.")

    # Train
    scores = model.fit(
        X=X_train, t=y_train,
        scheduler=scheduler,
        batches=batches,
        epochs=epochs,
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        X_val=X_val,
        t_val=y_val,
        save_on_interrupt=None,  # could set a path if you want weight snapshots
    )

    # Collect
    result = {
        "train_errors": scores["train_errors"],
        "val_errors": scores.get("val_errors", np.full(epochs, np.nan)),
        "method": method,
        "eta": eta,
        "batch_mode": batch_mode,
        "seed": seed,
    }
    return result

def eta_grid_for_mode(mode: str, etas: list[float]) -> list[float]:
    """
    Per SGD usiamo età più conservative (<= 3e-4).
    Per full-batch lasciamo l'intero grid.
    """
    if mode == "sgd":
        return [e for e in etas if e <= 3e-4]
    return list(etas)


# ------------------------- Orchestration -------------------------

def run_sweep(
    outdir: Path,
    methods: List[str],
    etas: List[float],
    seeds: List[int],
    epochs: int,
    batches_sgd: int,
    lam_l1: float,
    lam_l2: float,
    n_samples: int,
    noise_std: float,
    val_frac: float,
    avg_last_n: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs full sweep and writes CSVs. Returns (per_epoch_df, summary_df)."""

    # Data (same X across runs; different seeds control init/sampling noise)
    # For scientific comparability, fix the dataset for all seeds (seeds apply to NN init).
    X, y = make_dataset(n=n_samples, noise_std=noise_std, seed=1234)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_frac=val_frac, seed=2025)
    Xtr, Xva = standardize(Xtr, Xva)

    arch = default_arch(input_dim=Xtr.shape[1], output_dim=ytr.shape[1])

    # Prepare collectors
    rows: List[Dict] = []

    configs = []
    for method, seed in itertools.product(methods, seeds):
        for mode in ["sgd", "full"]:
            for eta in eta_grid_for_mode(mode, etas):
                configs.append((method, eta, mode, seed))
    
    total = len(configs)
    print(f"Total runs: {total}")

    try:
        for idx, (method, eta, batch_mode, seed) in enumerate(configs, 1):
            print(f"\n[{idx}/{total}] {method} | eta={eta:g} | {batch_mode} | seed={seed}")
            res = run_one(
                method=method,
                eta=eta,
                batch_mode=batch_mode,
                seed=seed,
                X_train=Xtr, y_train=ytr,
                X_val=Xva, y_val=yva,
                epochs=epochs,
                batches_sgd=batches_sgd,
                lam_l1=lam_l1,
                lam_l2=lam_l2,
                arch=arch,
            )
            # Per-epoch records
            for e in range(epochs):
                rows.append({
                    "method": res["method"],
                    "eta": res["eta"],
                    "batch_mode": res["batch_mode"],
                    "seed": res["seed"],
                    "epoch": e + 1,
                    "train_loss": float(res["train_errors"][e]),
                    "val_loss": float(res["val_errors"][e]),
                })

    except KeyboardInterrupt:
        print("\nInterrupted—saving partial results...")

    # Build DataFrame and save
    per_epoch_df = pd.DataFrame(rows)
    per_epoch_path = outdir / "per_epoch_metrics.csv"
    per_epoch_df.to_csv(per_epoch_path, index=False)
    print(f"Saved per-epoch metrics -> {per_epoch_path}")

    # Summary: mean of last N epochs (train & val)
    def mean_last_n(g: pd.DataFrame, col: str, n: int) -> float:
        return g.tail(n)[col].mean()

    summary_records = []
    group_cols = ["method", "eta", "batch_mode", "seed"]
    if per_epoch_df.empty:
        summary_df = pd.DataFrame(columns=group_cols + ["mean_lastN_train", "mean_lastN_val", "N"])
    else:
        for (method, eta, batch_mode, seed), g in per_epoch_df.groupby(group_cols):
            g_sorted = g.sort_values("epoch")
            summary_records.append({
                "method": method,
                "eta": eta,
                "batch_mode": batch_mode,
                "seed": seed,
                "mean_lastN_train": mean_last_n(g_sorted, "train_loss", min(avg_last_n, len(g_sorted))),
                "mean_lastN_val": mean_last_n(g_sorted, "val_loss", min(avg_last_n, len(g_sorted))),
                "N": min(avg_last_n, len(g_sorted)),
            })
        summary_df = pd.DataFrame(summary_records)

    summary_path = outdir / f"summary_last{avg_last_n}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary -> {summary_path}")

    # Also save a heatmap-ready pivot (optional): average over seeds, separate files for SGD and FULL
    for mode in ["sgd", "full"]:
        sub = summary_df[summary_df["batch_mode"] == mode]
        if sub.empty:
            continue
        # average across seeds
        sub_mean = sub.groupby(["method", "eta"], as_index=False)[["mean_lastN_val"]].mean()
        pivot = sub_mean.pivot(index="method", columns="eta", values="mean_lastN_val")
        pivot_path = outdir / f"heatmap_pivot_val_{mode}.csv"
        pivot.to_csv(pivot_path)
        print(f"Saved heatmap pivot ({mode}) -> {pivot_path}")

    # Save config JSON for provenance
    config = {
        "methods": methods,
        "etas": etas,
        "seeds": seeds,
        "epochs": epochs,
        "batches_sgd": batches_sgd,
        "lam_l1": lam_l1,
        "lam_l2": lam_l2,
        "n_samples": n_samples,
        "noise_std": noise_std,
        "val_frac": val_frac,
        "avg_last_n": avg_last_n,
        "arch": arch,
        "timestamp": datetime.now().isoformat(),
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    return per_epoch_df, summary_df

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Optimizer sweep for FFNN on Runge function (SGD vs Full-batch)."
    )
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--avg-last-n", type=int, default=50, help="Average of the last N epochs for summary.")
    p.add_argument("--n-samples", type=int, default=1200)
    p.add_argument("--noise-std", type=float, default=0.03)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batches-sgd", type=int, default=10, help="Number of batches for SGD mode.")
    p.add_argument("--lam-l1", type=float, default=0.0)
    p.add_argument("--lam-l2", type=float, default=0.0)

    p.add_argument(
        "--methods", type=str, nargs="*", default=[
            "constant", "momentum", "adagrad", "adagradmomentum", "rmsprop", "adam"
        ],
        help="Optimizers to include."
    )
    p.add_argument(
        "--etas", type=float, nargs="*",
        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        help="Learning rates to sweep."
    )
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 1337, 31415])

    p.add_argument("--outdir", type=str, default="output/OptimizerSweep")
    return p.parse_args()

def main():
    args = parse_args()
    # timestamped subfolder
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path(args.outdir) / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {outdir}")

    run_sweep(
        outdir=outdir,
        methods=args.methods,
        etas=args.etas,
        seeds=args.seeds,
        epochs=args.epochs,
        batches_sgd=args.batches_sgd,
        lam_l1=args.lam_l1,
        lam_l2=args.lam_l2,
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        val_frac=args.val_frac,
        avg_last_n=args.avg_last_n,
    )

if __name__ == "__main__":
    main()
