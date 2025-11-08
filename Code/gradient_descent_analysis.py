# gd_sweep_and_plots.py
# Sweep optimizer + LR (molto ampio) su Runge, con plots e gestione "explosion-safe".
# - CSV completi (per-epoch + summary + pivots) + PNG (heatmap + curve)
# - LR log-spaced configurabili via esponenti (e.g., 10**[-8..0])
# - Se un run esplode/va NaN/Inf: non blocca il resto; marcato in CSV e "×" in heatmap.

from __future__ import annotations
import argparse, itertools, json, math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- Project modules (devono essere importabili) ---
from src.activation_functions import LRELU
from src.cost_functions import CostOLS
from src.scheduler import Constant, Momentum, Adagrad, AdagradMomentum, RMS_prop, Adam
from src.FFNN import FFNN

# -------------------- DATA --------------------

def runge(x: np.ndarray, noise_std: float = 0.03) -> np.ndarray:
    noise = np.random.normal(0.0, noise_std, size=x.shape)
    return 1.0 / (1.0 + 25.0 * x**2) + noise

def make_dataset(n: int = 1500, seed: int | None = 1234):
    if seed is not None:
        np.random.seed(seed)
    X = np.linspace(-1.0, 1.0, n).reshape(-1, 1)
    y = runge(X, noise_std=0.03).reshape(-1, 1)
    return X, y

def train_val_split(X: np.ndarray, y: np.ndarray, val_frac: float = 0.2, seed: int | None = 2025):
    n = X.shape[0]
    idx = np.arange(n)
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(idx)
    k = int(math.floor((1.0 - val_frac) * n))
    tr, va = idx[:k], idx[k:]
    return X[tr], y[tr], X[va], y[va]

def standardize(Xtr, Xva):
    mu = Xtr.mean(axis=0, keepdims=True)
    sd = Xtr.std(axis=0, keepdims=True) + 1e-12
    return (Xtr - mu) / sd, (Xva - mu) / sd

# -------------------- MODEL --------------------

def default_arch(input_dim: int = 1, output_dim: int = 1) -> Tuple[int, ...]:
    return (input_dim, 30, 30, output_dim)

def build_scheduler(name: str, eta: float):
    name = name.lower()
    if name == "constant":          return Constant(eta)
    if name == "momentum":          return Momentum(eta=eta, momentum=0.9)
    if name == "adagrad":           return Adagrad(eta=eta)
    if name == "adagradmomentum":   return AdagradMomentum(eta=eta, momentum=0.9)
    if name == "rmsprop":           return RMS_prop(eta=eta, rho=0.9)
    if name == "adam":              return Adam(eta=eta, rho=0.9, rho2=0.999)
    raise ValueError(f"Unknown optimizer: {name}")

# -------------------- SINGLE RUN (safe) --------------------

def run_one_safe(
    method: str,
    eta: float,
    mode: str,
    seed: int,
    Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray,
    arch: Tuple[int, ...],
    epochs: int,
    batches_sgd: int,
    lam_l1: float, lam_l2: float,
) -> Dict:
    """
    Esegue un run. Se esplode (overflow/NaN/Inf/Exception), ritorna 'status=exploded'
    e metriche NaN; altrimenti status 'ok' con le serie complete.
    """
    try:
        model = FFNN(
            dimensions=arch,
            hidden_func=LRELU,
            output_func=lambda x: x,
            cost_func=CostOLS,
            seed=seed,
        )
        scheduler = build_scheduler(method, eta)
        batches = (batches_sgd if mode == "sgd" else 1)

        scores = model.fit(
            X=Xtr, t=ytr,
            scheduler=scheduler,
            batches=batches,
            epochs=epochs,
            lam_l1=lam_l1, lam_l2=lam_l2,
            X_val=Xva, t_val=yva,
            save_on_interrupt=None,
        )

        tr = np.asarray(scores["train_errors"])
        va = np.asarray(scores.get("val_errors", np.full(epochs, np.nan)))

        # Se si sono generati NaN/Inf lungo il training, marchiamo come exploded
        if not np.all(np.isfinite(tr)) or not np.all(np.isfinite(va)):
            return {
                "status": "exploded",
                "reason": "non-finite losses",
                "epoch_of_failure": int(np.where(~np.isfinite(va))[0][0]) if np.any(~np.isfinite(va)) else -1,
                "train_errors": tr, "val_errors": va
            }

        return {"status": "ok", "train_errors": tr, "val_errors": va}

    except Exception as e:
        return {
            "status": "exploded",
            "reason": repr(e),
            "epoch_of_failure": -1,
            "train_errors": np.full(epochs, np.nan),
            "val_errors":   np.full(epochs, np.nan),
        }

# -------------------- SWEEP + LOG --------------------

def logspace_from_exponents(exp_min: float, exp_max: float, n: int) -> List[float]:
    # 10**[exp_min .. exp_max] con n punti
    return list(np.logspace(exp_min, exp_max, int(n)))

def run_sweep(
    outdir: Path,
    methods: List[str],
    etas_full: List[float],
    etas_sgd: List[float],
    seeds: List[int],
    epochs: int,
    batches_sgd: int,
    lam_l1: float, lam_l2: float,
    n_samples: int,
    noise_std: float,
    val_frac: float,
    avg_last_n: int,
):
    # Dataset fisso per comparabilità
    X, y = make_dataset(n=n_samples, seed=1234)
    Xtr, ytr, Xva, yva = train_val_split(X, y, val_frac=val_frac, seed=2025)
    Xtr, Xva = standardize(Xtr, Xva)
    arch = default_arch(Xtr.shape[1], ytr.shape[1])

    configs = []
    for method, seed in itertools.product(methods, seeds):
        for mode in ["sgd", "full"]:
            grid = (etas_sgd if mode == "sgd" else etas_full)
            for eta in grid:
                configs.append((method, float(eta), mode, int(seed)))

    rows = []
    failures = []
    print(f"Total runs: {len(configs)}")

    for i, (method, eta, mode, seed) in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] {method} | eta={eta:g} | {mode} | seed={seed}")
        res = run_one_safe(
            method, eta, mode, seed,
            Xtr, ytr, Xva, yva,
            arch, epochs, batches_sgd, lam_l1, lam_l2
        )

        if res["status"] != "ok":
            failures.append({
                "method": method, "eta": eta, "batch_mode": mode, "seed": seed,
                "status": res["status"], "reason": res.get("reason", ""),
                "epoch_of_failure": res.get("epoch_of_failure", -1)
            })

        tr = res["train_errors"]; va = res["val_errors"]
        for e in range(epochs):
            rows.append({
                "method": method, "eta": eta, "batch_mode": mode, "seed": seed, "epoch": e+1,
                "train_loss": float(tr[e]) if np.isfinite(tr[e]) else np.nan,
                "val_loss":   float(va[e]) if np.isfinite(va[e]) else np.nan,
                "status": res["status"]
            })

    per_epoch_df = pd.DataFrame(rows)
    per_epoch_path = outdir / "per_epoch_metrics.csv"
    per_epoch_df.to_csv(per_epoch_path, index=False)
    print(f"[saved] {per_epoch_path}")

    failures_df = pd.DataFrame(failures)
    if not failures_df.empty:
        failures_path = outdir / "exploded_runs.csv"
        failures_df.to_csv(failures_path, index=False)
        print(f"[saved] {failures_path}")

    # Summary: mean last N (ignora NaN)
    summary_records = []
    group_cols = ["method", "eta", "batch_mode", "seed"]
    for key, g in per_epoch_df.groupby(group_cols):
        g = g.sort_values("epoch")
        n = min(avg_last_n, len(g))
        summary_records.append({
            "method": key[0], "eta": float(key[1]), "batch_mode": key[2], "seed": int(key[3]),
            "mean_lastN_train": g.tail(n)["train_loss"].mean(skipna=True),
            "mean_lastN_val":   g.tail(n)["val_loss"].mean(skipna=True),
            "N": n,
            "all_nan": bool(g["val_loss"].tail(n).isna().all())
        })
    summary_df = pd.DataFrame(summary_records)
    summary_path = outdir / f"summary_last{avg_last_n}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # Pivots per heatmap (media su seeds); i punti "all_nan" restano NaN
    for mode in ["sgd", "full"]:
        sub = summary_df[summary_df["batch_mode"] == mode]
        if sub.empty: continue
        mean_by_eta = sub.groupby(["method", "eta"], as_index=False)["mean_lastN_val"].mean()
        pivot = mean_by_eta.pivot(index="method", columns="eta", values="mean_lastN_val")
        pivot_path = outdir / f"heatmap_pivot_val_{mode}.csv"
        pivot.to_csv(pivot_path)
        print(f"[saved] {pivot_path}")

    cfg = {
        "methods": methods,
        "etas_full": list(map(float, etas_full)),
        "etas_sgd":  list(map(float, etas_sgd)),
        "seeds": seeds,
        "epochs": epochs, "batches_sgd": batches_sgd,
        "lam_l1": lam_l1, "lam_l2": lam_l2,
        "n_samples": n_samples, "noise_std": noise_std, "val_frac": val_frac,
        "avg_last_n": avg_last_n, "arch": arch, "timestamp": datetime.now().isoformat(),
    }
    with open(outdir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # Plots
    figdir = outdir / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    make_heatmaps(figdir, summary_df, failures_df if not failures_df.empty else None)
    make_best_lr_curves(figdir, per_epoch_df, summary_df)

    print("Done.")

# -------------------- PLOTS --------------------

def sorted_unique_etas(df: pd.DataFrame) -> List[float]:
    return sorted(df["eta"].unique().tolist(), key=float)

def plot_heatmap(ax, data_2d: np.ndarray, row_labels: List[str], col_labels: List[float], title: str):
    im = ax.imshow(data_2d, aspect="auto", origin="upper")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([f"{e:g}" for e in col_labels], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    ax.set_xlabel("learning rate (eta)")
    ax.set_ylabel("optimizer")
    cb = plt.colorbar(im, ax=ax)
    cb.set_label("mean_lastN_val")

def make_heatmaps(figdir: Path, summary: pd.DataFrame, failures: pd.DataFrame | None, dpi: int = 160):
    grouped = summary.groupby(["batch_mode", "method", "eta"], as_index=False)["mean_lastN_val"].mean()
    exploded_pairs = set()
    if failures is not None and not failures.empty:
        # Mark a (method, eta, mode) as exploded if QUALSIASI seed è esploso
        for (m, e, mode), g in failures.groupby(["method", "eta", "batch_mode"]):
            exploded_pairs.add((mode, m, float(e)))

    for mode in ["sgd", "full"]:
        sub = grouped[grouped["batch_mode"] == mode]
        if sub.empty: continue
        methods = sorted(sub["method"].unique().tolist())
        etas = sorted_unique_etas(sub)

        # matrix per heatmap
        mat = np.full((len(methods), len(etas)), np.nan)
        explode_mask = np.zeros_like(mat, dtype=bool)

        for i, m in enumerate(methods):
            for j, e in enumerate(etas):
                row = sub[(sub["method"] == m) & (sub["eta"] == e)]
                if not row.empty:
                    mat[i, j] = row["mean_lastN_val"].values[0]
                if (mode, m, float(e)) in exploded_pairs:
                    explode_mask[i, j] = True

        fig, ax = plt.subplots(figsize=(12, 6))
        plot_heatmap(ax, mat, methods, etas, f"Validation loss heatmap (avg last N) — {mode.upper()}")

        # overlay “×” dove è esploso almeno un seed per quella cella
        for i in range(explode_mask.shape[0]):
            for j in range(explode_mask.shape[1]):
                if explode_mask[i, j]:
                    ax.text(j, i, "×", ha="center", va="center")

        fig.tight_layout()
        out = figdir / f"heatmap_val_lastN_{mode}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)

def pick_best_eta_per_method(summary: pd.DataFrame, mode: str) -> pd.DataFrame:
    sub = summary[summary["batch_mode"] == mode]
    if sub.empty: return pd.DataFrame(columns=["method", "eta"])
    mean_by_eta = sub.groupby(["method", "eta"], as_index=False)["mean_lastN_val"].mean()
    idx = mean_by_eta.groupby("method")["mean_lastN_val"].idxmin()
    return mean_by_eta.loc[idx, ["method", "eta"]].reset_index(drop=True)

def make_best_lr_curves(figdir: Path, per_epoch: pd.DataFrame, summary: pd.DataFrame, dpi: int = 160, show_seed_curves: bool = True):
    for mode in ["sgd", "full"]:
        best = pick_best_eta_per_method(summary, mode)
        if best.empty: continue

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Validation loss vs epoch — best η per optimizer ({mode.upper()})")
        ax.set_xlabel("epoch"); ax.set_ylabel("val_loss")
        legends = []
        for _, row in best.iterrows():
            method, eta = row["method"], row["eta"]
            sub = per_epoch[(per_epoch["batch_mode"] == mode) & (per_epoch["method"] == method) & (per_epoch["eta"] == eta)]
            if sub.empty: continue
            g = sub.groupby("epoch")["val_loss"]
            mean, std = g.mean(), g.std()
            x, y = mean.index.values, mean.values
            ax.plot(x, y)
            if np.isfinite(std.values).all():
                ax.fill_between(x, mean - std, mean + std, alpha=0.2)
            legends.append(f"{method} (η={eta:g})")
        ax.legend(legends, loc="best")
        fig.tight_layout()
        out = figdir / f"curves_best_eta_{mode}.png"
        fig.savefig(out, dpi=dpi)
        plt.close(fig)

# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Optimizer sweep (SGD vs FULL) su Runge + plots + explosion catching.")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--avg-last-n", type=int, default=50)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batches-sgd", type=int, default=30)
    p.add_argument("--lam-l1", type=float, default=0.0)
    p.add_argument("--lam-l2", type=float, default=0.0)
    p.add_argument("--methods", type=str, nargs="*", default=[
        "constant", "momentum", "adagrad", "adagradmomentum", "rmsprop", "adam"
    ])
    p.add_argument("--seeds", type=int, nargs="*", default=[42, 1337, 31415])
    p.add_argument("--outdir", type=str, default="output/OptimizerSweep")

    # ---- LR grids super-ampi, generati via esponenti ----
    p.add_argument("--full-exp-min", type=float, default=-8, help="log10 min per FULL (10**value)")
    p.add_argument("--full-exp-max", type=float, default=0,  help="log10 max per FULL (10**value)")
    p.add_argument("--full-n",      type=int,   default=33,  help="# punti FULL")
    p.add_argument("--sgd-exp-min", type=float, default=-8,  help="log10 min per SGD (10**value)")
    p.add_argument("--sgd-exp-max", type=float, default=-2,  help="log10 max per SGD (10**value)")
    p.add_argument("--sgd-n",       type=int,   default=25,  help="# punti SGD")
    return p.parse_args()

def main():
    args = parse_args()
    outdir = Path(args.outdir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Writing results to: {outdir}")

    etas_full = logspace_from_exponents(args.full_exp_min, args.full_exp_max, args.full_n)
    etas_sgd  = logspace_from_exponents(args.sgd_exp_min,  args.sgd_exp_max,  args.sgd_n)

    run_sweep(
        outdir=outdir,
        methods=args.methods,
        etas_full=etas_full,
        etas_sgd=etas_sgd,
        seeds=args.seeds,
        epochs=args.epochs,
        batches_sgd=args.batches_sgd,
        lam_l1=args.lam_l1, lam_l2=args.lam_l2,
        n_samples=args.n_samples, noise_std=0.02, val_frac=args.val_frac,
        avg_last_n=args.avg_last_n,
    )

if __name__ == "__main__":
    main()
