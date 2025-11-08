# plot_from_csvs_globals.py
# -----------------------------------------------------------------------------
# Re-render heatmaps and training curves from CSV logs produced by
# gd_sweep_and_plots.py. All configuration is done via GLOBAL VARIABLES below.
# -----------------------------------------------------------------------------

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
#                           GLOBAL CONFIGURATION
# =============================================================================

# Folder containing CSV logs produced by gd_sweep_and_plots.py
INDIR = Path("output/OptimizerSweep/20251108_153012")

# (optional) specify exact summary csv, or leave None to auto-detect
SUMMARY_CSV = None

# Output directory for the plots
FIGDIR = INDIR / "figures_from_csv"

# What to plot: "sgd", "full", or ["sgd","full"]
MODES = ["sgd", "full"]

# Heatmap visualization limits (set None to auto scale)
VMIN = None           # e.g., 0.001
VMAX = None           # e.g., 0.05

# Filter: show only learning rates between these limits (None disables)
ETA_MIN = None        # e.g., 1e-6
ETA_MAX = None        # e.g., 1e-2

# Show only specific optimizers (case-insensitive); None means "all"
ONLY_METHODS = None   # e.g., ["adam", "rmsprop"]

# Indicate cells where at least one seed exploded
ANNOTATE_EXPLODED = True

# Limit curves to first N epochs (None means full training)
EPOCH_MAX = None

# heatmap colormap
CMAP = "viridis"

# image DPI
DPI = 160

# =============================================================================
#                          INTERNAL IMPLEMENTATION
# =============================================================================


def discover_summary(indir: Path, summary_path: Optional[Path]) -> Path:
    if summary_path is not None:
        return summary_path
    candidates = sorted(indir.glob("summary_last*.csv"))
    if not candidates:
        raise FileNotFoundError("No summary_last*.csv found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_data(indir: Path, summary_path: Optional[Path]):
    per_epoch = pd.read_csv(indir / "per_epoch_metrics.csv")
    summary = pd.read_csv(discover_summary(indir, summary_path))

    failures = None
    if (indir / "exploded_runs.csv").exists():
        failures = pd.read_csv(indir / "exploded_runs.csv")

    return per_epoch, summary, failures


def filter_eta(df, eta_min, eta_max):
    if eta_min is not None:
        df = df[df["eta"] >= float(eta_min)]
    if eta_max is not None:
        df = df[df["eta"] <= float(eta_max)]
    return df


def filter_methods(df, methods):
    if not methods:
        return df
    return df[df["method"].str.lower().isin([m.lower() for m in methods])]


def sorted_unique_etas(df: pd.DataFrame):
    return sorted(df["eta"].astype(float).unique().tolist(), key=float)


def best_eta_per_optimizer(summary: pd.DataFrame, mode: str):
    sub = summary[summary["batch_mode"] == mode]
    if sub.empty:
        return pd.DataFrame(columns=["method", "eta"])
    grouped = sub.groupby(["method", "eta"], as_index=False)["mean_lastN_val"].mean()
    idx = grouped.groupby("method")["mean_lastN_val"].idxmin()
    return grouped.loc[idx, ["method", "eta"]]


def plot_heatmap(figdir, summary, failures, mode: str):
    sub = summary[summary["batch_mode"] == mode].copy()
    sub = filter_eta(sub, ETA_MIN, ETA_MAX)
    sub = filter_methods(sub, ONLY_METHODS)
    if sub.empty:
        print(f"[heatmap] Skipping mode={mode}, dataset empty.")
        return

    grouped = sub.groupby(["method", "eta"], as_index=False)["mean_lastN_val"].mean()
    methods = sorted(grouped["method"].unique().tolist())
    etas = sorted_unique_etas(grouped)

    mat = np.full((len(methods), len(etas)), np.nan)
    for i, m in enumerate(methods):
        for j, e in enumerate(etas):
            row = grouped[(grouped["method"] == m) & (grouped["eta"] == e)]
            if not row.empty:
                mat[i, j] = float(row["mean_lastN_val"].values[0])

    explode_mask = np.zeros_like(mat, dtype=bool)
    if ANNOTATE_EXPLODED and failures is not None:
        f = failures[failures["batch_mode"] == mode]
        exploded_pairs = {(row["method"], float(row["eta"])) for _, row in f.iterrows()}
        for i, m in enumerate(methods):
            for j, e in enumerate(etas):
                if (m, e) in exploded_pairs:
                    explode_mask[i, j] = True

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", origin="upper", vmin=VMIN, vmax=VMAX, cmap=CMAP)
    ax.set_xticks(np.arange(len(etas)))
    ax.set_xticklabels([f"{e:g}" for e in etas], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"VALIDATION LOSS — {mode.upper()}")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("mean_lastN_val")

    for i in range(len(methods)):
        for j in range(len(etas)):
            if explode_mask[i, j]:
                ax.text(j, i, "×", ha="center", va="center", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(figdir / f"heatmap_{mode}.png", dpi=DPI)
    plt.close(fig)


def plot_curves(figdir, per_epoch, summary, mode):
    best = best_eta_per_optimizer(summary, mode)
    if best.empty:
        print(f"[curves] No best LR found for {mode}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for _, row in best.iterrows():
        m, lr = row["method"], float(row["eta"])
        sub = per_epoch[(per_epoch["batch_mode"] == mode) &
                        (per_epoch["method"] == m) &
                        (per_epoch["eta"] == lr)]
        if EPOCH_MAX is not None:
            sub = sub[sub["epoch"] <= EPOCH_MAX]

        g = sub.groupby("epoch")["val_loss"]
        mean, std = g.mean(), g.std()
        ax.plot(mean.index.values, mean.values, label=f"{m} (η={lr:g})")
        if np.isfinite(std.values).all():
            ax.fill_between(mean.index.values, mean - std, mean + std, alpha=0.2)

    ax.set_title(f"BEST η CURVES — {mode.upper()}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / f"curves_{mode}.png", dpi=DPI)
    plt.close(fig)


# =============================================================================
#                                RUN
# =============================================================================

FIGDIR.mkdir(parents=True, exist_ok=True)
per_epoch, summary, failures = load_data(INDIR, SUMMARY_CSV)

for m in MODES:
    plot_heatmap(FIGDIR, summary, failures, m)
    plot_curves(FIGDIR, per_epoch, summary, m)

print("✅ All plots saved to:", FIGDIR)
