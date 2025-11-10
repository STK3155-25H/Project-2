# plot_from_csvs_globals.py
# -----------------------------------------------------------------------------
# Re-render heatmaps e curve (best η per optimizer) dai CSV prodotti da
# gd_sweep_and_plots.py. Tutta la configurazione è fatta via variabili GLOBALI.
# - Limiti separati per SGD e FULL (heatmap vmin/vmax + curve x/y + epoch-max).
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

# Cartella che contiene i CSV (per_epoch_metrics.csv, summary_last*.csv, exploded_runs.csv?)
INDIR = Path("output/OptimizerSweep/20251108_122801")

# (Opzionale) path esplicito al summary_last*.csv (se None lo scopre da solo)
SUMMARY_CSV = None

# Dove salvare le figure
FIGDIR = INDIR / "figures_from_csv"

# Quali modalità plottare (metti ["sgd"], ["full"] o ["sgd","full"])
MODES = ["sgd", "full"]

# ---------------- Heatmap: limiti colore separati per SGD e FULL ----------------
# Metti None per autoscale
SGD_VMIN = None
SGD_VMAX = 0.8
FULL_VMIN = None
FULL_VMAX = 0.8

# Colormap (uguale per entrambi i mode)
CMAP = "viridis"

# ---------------- Curve: limiti assi separati per SGD e FULL ----------------
# Se None, non impone il limite
SGD_CURVE_XMIN = 0.0
SGD_CURVE_XMAX = 400
SGD_CURVE_YMIN = 0.0
SGD_CURVE_YMAX = 0.2

FULL_CURVE_XMIN = 0.0
FULL_CURVE_XMAX = 30
FULL_CURVE_YMIN = 0.0
FULL_CURVE_YMAX = 1.5

# Taglia le curve ai primi N epoch (se None non taglia).
# Puoi specificare override per mode; se None usa EPOCH_MAX globale.
EPOCH_MAX = None
SGD_EPOCH_MAX = None
FULL_EPOCH_MAX = None

# ---------------- Filtri e opzioni comuni ----------------
# Filtra i learning rate mostrati (None = nessun filtro)
ETA_MIN = None      # es. 1e-6
ETA_MAX = None      # es. 1e-2

# Mostra solo questi optimizer (case-insensitive); None = tutti
ONLY_METHODS: Optional[List[str]] = None  # es. ["adam", "rmsprop"]

# Annotare con "×" le celle in cui almeno un seed è esploso
ANNOTATE_EXPLODED = True

# DPI immagini
DPI = 160

# =============================================================================
#                          INTERNAL IMPLEMENTATION
# =============================================================================

def discover_summary(indir: Path, summary_path: Optional[Path]) -> Path:
    if summary_path is not None:
        return summary_path
    candidates = sorted(indir.glob("summary_last*.csv"))
    if not candidates:
        raise FileNotFoundError("Nessun summary_last*.csv trovato.")
    return max(candidates, key=lambda p: p.stat().st_mtime)

def load_data(indir: Path, summary_path: Optional[Path]):
    per_epoch = pd.read_csv(indir / "per_epoch_metrics.csv")
    summary = pd.read_csv(discover_summary(indir, summary_path))
    failures = None
    if (indir / "exploded_runs.csv").exists():
        failures = pd.read_csv(indir / "exploded_runs.csv")
    return per_epoch, summary, failures

def filter_eta(df: pd.DataFrame, eta_min, eta_max):
    if eta_min is not None:
        df = df[df["eta"] >= float(eta_min)]
    if eta_max is not None:
        df = df[df["eta"] <= float(eta_max)]
    return df

def filter_methods(df: pd.DataFrame, methods: Optional[List[str]]):
    if not methods:
        return df
    lower = {m.lower() for m in methods}
    return df[df["method"].str.lower().isin(lower)]

def sorted_unique_etas(df: pd.DataFrame):
    return sorted(df["eta"].astype(float).unique().tolist(), key=float)

def best_eta_per_optimizer(summary: pd.DataFrame, mode: str):
    sub = summary[summary["batch_mode"] == mode]
    if sub.empty:
        return pd.DataFrame(columns=["method", "eta"])
    grouped = sub.groupby(["method", "eta"], as_index=False)["mean_lastN_val"].mean()
    idx = grouped.groupby("method")["mean_lastN_val"].idxmin()
    return grouped.loc[idx, ["method", "eta"]]

def mode_limits_for_heatmap(mode: str):
    if mode == "sgd":
        return SGD_VMIN, SGD_VMAX
    return FULL_VMIN, FULL_VMAX

def mode_limits_for_curve(mode: str):
    if mode == "sgd":
        return (SGD_CURVE_XMIN, SGD_CURVE_XMAX, SGD_CURVE_YMIN, SGD_CURVE_YMAX)
    return (FULL_CURVE_XMIN, FULL_CURVE_XMAX, FULL_CURVE_YMIN, FULL_CURVE_YMAX)

def mode_epoch_max(mode: str):
    if mode == "sgd":
        return SGD_EPOCH_MAX if SGD_EPOCH_MAX is not None else EPOCH_MAX
    return FULL_EPOCH_MAX if FULL_EPOCH_MAX is not None else EPOCH_MAX

# -------------------- Plot: Heatmap --------------------

def plot_heatmap(figdir: Path, summary: pd.DataFrame, failures: Optional[pd.DataFrame], mode: str):
    sub = summary[summary["batch_mode"] == mode].copy()
    sub = filter_eta(sub, ETA_MIN, ETA_MAX)
    sub = filter_methods(sub, ONLY_METHODS)
    if sub.empty:
        print(f"[heatmap] Nessun dato per mode={mode} (dopo filtri). Skipping.")
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
    if ANNOTATE_EXPLODED and failures is not None and not failures.empty:
        fsub = failures[failures["batch_mode"] == mode]
        fsub = filter_eta(fsub, ETA_MIN, ETA_MAX)
        fsub = filter_methods(fsub, ONLY_METHODS)
        exploded_pairs = {(r["method"], float(r["eta"])) for _, r in fsub.iterrows()}
        for i, m in enumerate(methods):
            for j, e in enumerate(etas):
                if (m, e) in exploded_pairs:
                    explode_mask[i, j] = True

    vmin, vmax = mode_limits_for_heatmap(mode)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(mat, aspect="auto", origin="upper", vmin=vmin, vmax=vmax, cmap=CMAP)
    ax.set_xticks(np.arange(len(etas)))
    ax.set_xticklabels([f"{e:g}" for e in etas], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels(methods)
    ax.set_title(f"Validation loss heatmap (avg last N) — {mode.upper()}")
    ax.set_xlabel("learning rate (eta)")
    ax.set_ylabel("optimizer")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("mean_lastN_val")

    for i in range(len(methods)):
        for j in range(len(etas)):
            if explode_mask[i, j]:
                ax.text(j, i, "×", ha="center", va="center", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(figdir / f"heatmap_{mode}.png", dpi=DPI)
    plt.close(fig)
    print(f"[saved] {figdir / f'heatmap_{mode}.png'}")

# -------------------- Plot: Curves (best η per optimizer) --------------------

def plot_curves(figdir: Path, per_epoch: pd.DataFrame, summary: pd.DataFrame, mode: str):
    sub_sum = summary[summary["batch_mode"] == mode].copy()
    sub_sum = filter_eta(sub_sum, ETA_MIN, ETA_MAX)
    sub_sum = filter_methods(sub_sum, ONLY_METHODS)
    if sub_sum.empty:
        print(f"[curves] Nessun dato summary per mode={mode} dopo filtri. Skipping.")
        return

    best = best_eta_per_optimizer(sub_sum, mode)
    if best.empty:
        print(f"[curves] Nessun best η per mode={mode}.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    x_min, x_max, y_min, y_max = mode_limits_for_curve(mode)
    emax = mode_epoch_max(mode)

    legends = []
    for _, row in best.iterrows():
        method, eta = row["method"], float(row["eta"])
        sub = per_epoch[
            (per_epoch["batch_mode"] == mode) &
            (per_epoch["method"] == method) &
            (per_epoch["eta"].astype(float) == eta)
        ].copy()

        if sub.empty:
            continue

        if emax is not None:
            sub = sub[sub["epoch"] <= int(emax)]

        g = sub.groupby("epoch")["val_loss"]
        mean, std = g.mean(), g.std()
        x, y = mean.index.values, mean.values
        ax.plot(x, y)
        if np.isfinite(std.values).all():
            ax.fill_between(x, mean - std, mean + std, alpha=0.2)
        legends.append(f"{method} (η={eta:g})")

    # Limiti assi dedicati per mode
    if (x_min is not None) or (x_max is not None):
        ax.set_xlim(x_min, x_max)
    if (y_min is not None) or (y_max is not None):
        ax.set_ylim(y_min, y_max)

    ax.set_title(f"Validation loss vs epoch — best η per optimizer ({mode.upper()})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss")
    if legends:
        ax.legend(legends, loc="best")
    fig.tight_layout()
    out = figdir / f"curves_{mode}.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"[saved] {out}")

# =============================================================================
#                                 RUN
# =============================================================================

def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    per_epoch, summary, failures = load_data(INDIR, Path(SUMMARY_CSV) if SUMMARY_CSV else None)

    for m in MODES:
        plot_heatmap(FIGDIR, summary, failures, m)
        plot_curves(FIGDIR, per_epoch, summary, m)

    print("✅ All plots saved to:", FIGDIR)

if __name__ == "__main__":
    main()
