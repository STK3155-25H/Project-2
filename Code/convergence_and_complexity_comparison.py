import os
from config import OUTPUT_DIR, MODELS_DIR
BASE_DIR = MODELS_DIR
BENCHMARK_OUTPUT_DIR = OUTPUT_DIR 
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch

# ---- Imports from your codebase ----
from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import RELU, identity
from Code.src.torch_utils import train_one_model



# ======================================================================
# Experiment constants (required parameters)
# ======================================================================
N_POINTS = 200 # linspace of 200 points in [-1, 1]
EPOCHS = 1500 # number of training epochs
LR = 1e-3 # learning rate
LAM_L1 = 0.0 # L1 regularization
LAM_L2 = 0.0 # L2 regularization
BATCHES = 100 # number of mini-batches
NOISE_STD = 0.03 # Gaussian noise on targets
TEST_SIZE = 0.25 # 75% train, 25% validation
SEED = 314 # for reproducibility
# Layouts to sweep (n_hidden, width). Edit as desired.
LAYOUTS: List[Tuple[int, int]] = [
        (1, 8),
        (1, 16),
        (1, 32),
        (2, 16),
        (2, 32),
        (2, 64),
        (3, 32),
        (3, 64),
        (3, 128),
        (4, 64),
        (4, 128),
        (5, 128)
]
# ======================================================================
# I/O setup
# ======================================================================
OUT_ROOT = Path(BENCHMARK_OUTPUT_DIR)
CSV_DIR = OUT_ROOT / "csv"
FIG_DIR = OUT_ROOT / "figs"
for p in (CSV_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)


# ======================================================================
# Utils: dataset, layout, parameters
# ======================================================================

def runge(x: np.ndarray, noise_std: float = 0.0, rng=None) -> np.ndarray:
    """Runge function: 1 / (1 + 25 x^2) with optional Gaussian noise."""
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, noise_std, size=x.shape) if noise_std > 0 else 0.0
    return 1.0 / (1.0 + 25.0 * x**2) + noise


def build_layout(n_hidden: int, width: int) -> List[int]:
    """Builds a layout [1, width, ..., width, 1]."""
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]


def count_params_from_layout(layout: List[int]) -> int:
    """
    Number of parameters for a fully connected MLP with bias:
    sum_l ( (n_l + 1) * n_{l+1} ).
    """
    return int(sum((layout[i] + 1) * layout[i + 1] for i in range(len(layout) - 1)))


@dataclass
class ExperimentResult:
    impl: str # "ffnn" or "torch"
    n_hidden: int
    width: int
    params: int
    epochs: int
    train_time: float
    final_train_loss: float
    final_val_loss_noisy: float
    final_val_loss_clean: float


# ======================================================================
# Convergence checks
# ======================================================================

def assert_convergence(losses: np.ndarray, name: str, min_drop: float = 0.2) -> None:
    """
    Checks that the loss decreases "on average":
    - computes the mean over the first quarter of epochs and over the last quarter
    - requires that the last quarter is at least min_drop*100 % lower.

    min_drop = 0.2 => at least 20% mean reduction.
    """
    losses = np.asarray(losses, dtype=float)
    assert losses.ndim == 1 and len(losses) >= 10, \
        f"[{name}] Too few epochs ({len(losses)}) to evaluate convergence"

    q = max(1, len(losses) // 4)

    first_mean = losses[:q].mean()
    last_mean = losses[-q:].mean()

    msg = (
        f"[{name}] Insufficient convergence: "
        f"first_mean={first_mean:.4e}, last_mean={last_mean:.4e}"
    )
    assert last_mean < (1.0 - min_drop) * first_mean, msg


def assert_similar_performance(val1: float, val2: float,
                               rel_tol: float = 0.5,
                               name1: str = "ffnn",
                               name2: str = "torch") -> None:
    """
    Checks whether the two implementations have similar final MSE on clean validation data.

    If the relative difference is > (1 + rel_tol),
    it DOES NOT raise an AssertionError, only prints a warning.
    """
    v1, v2 = float(val1), float(val2)
    if v1 == 0 and v2 == 0:
        return

    ratio = max(v1, v2) / (min(v1, v2) + 1e-12)

    if ratio > 1.0 + rel_tol:
        msg = (f"[{name1} vs {name2}] WARNING: very different performance "
               f"(val1={v1:.4e}, val2={v2:.4e}, ratio={ratio:.3f})")
        print(msg)


# ======================================================================
# Training helpers
# ======================================================================

def train_ffnn_single_layout(
    layout: List[int],
    X_train: np.ndarray,
    y_train_noisy: np.ndarray,
    X_val: np.ndarray,
    y_val_noisy: np.ndarray,
    y_val_clean: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
    lam_l1: float = LAM_L1,
    lam_l2: float = LAM_L2,
    rho: float = 0.9,
    rho2: float = 0.999,
    batches: int = BATCHES,
    seed: int = SEED,
) -> Tuple[FFNN, Dict[str, np.ndarray], float, float, float, float]:
    """
    Trains a custom FFNN on a single layout and returns:
    - net: FFNN model
    - history: dict with train/val (noisy) curves
    - train_time: training time (seconds)
    - final_train_loss (noisy)
    - final_val_loss_noisy
    - final_val_loss_clean
    """
    np.random.seed(seed)

    net = FFNN(
        dimensions=layout,
        hidden_func=RELU,
        output_func=identity,
        cost_func=CostOLS,
        seed=seed,
    )
    scheduler = Adam(lr, rho, rho2)

    t0 = time.perf_counter()
    history_raw = net.fit(
        X=X_train,
        t=y_train_noisy,
        scheduler=scheduler,
        batches=batches,
        epochs=epochs,
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        X_val=X_val,
        t_val=y_val_noisy,
    )
    t1 = time.perf_counter()
    train_time = t1 - t0

    train_losses = np.asarray(
        history_raw.get("train_errors", history_raw.get("train_loss", [])),
        dtype=float,
    )
    val_losses_noisy = np.asarray(
        history_raw.get("val_errors", history_raw.get("val_loss", [])),
        dtype=float,
    )

    # Final evaluation (noisy & clean) on validation data
    y_pred_val = net.predict(X_val)
    val_loss_noisy = float(CostOLS(y_val_noisy)(y_pred_val))
    val_loss_clean = float(CostOLS(y_val_clean)(y_pred_val))

    # Final training loss (noisy)
    final_train_loss = float(train_losses[-1]) if len(train_losses) else float("nan")

    history = {
        "train_loss": train_losses,
        "val_loss_noisy": val_losses_noisy,
    }

    return net, history, train_time, final_train_loss, val_loss_noisy, val_loss_clean


def train_torch_single_layout(
    layout: List[int],
    X_train: np.ndarray,
    y_train_noisy: np.ndarray,
    X_val: np.ndarray,
    y_val_noisy: np.ndarray,
    y_val_clean: np.ndarray,
    epochs: int = EPOCHS,
    lr: float = LR,
    lam_l1: float = LAM_L1,
    lam_l2: float = LAM_L2,
    batches: int = BATCHES,
    seed: int = SEED,
) -> Tuple[torch.nn.Module, Dict[str, np.ndarray], float, float, float, float]:
    """
    Trains a PyTorch MLP with the same layout and returns:
    - model: PyTorch network
    - history: dict with train/val (noisy) curves
    - train_time: training time (seconds)
    - final_train_loss (noisy)
    - final_val_loss_noisy
    - final_val_loss_clean
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_noisy_t = torch.from_numpy(y_train_noisy.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_noisy_t = torch.from_numpy(y_val_noisy.astype(np.float32))
    y_val_clean_t = torch.from_numpy(y_val_clean.astype(np.float32))

    t0 = time.perf_counter()
    model, history_raw, final_noisy, final_clean = train_one_model(
        layout=layout,
        act_name="RELU",
        device=device,
        X_train_t=X_train_t,
        y_train_noisy_t=y_train_noisy_t,
        X_val_t=X_val_t,
        y_val_noisy_t=y_val_noisy_t,
        y_val_clean_t=y_val_clean_t,
        epochs=epochs,
        lr=lr,
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        batches=batches,
        betas=(0.9, 0.999),
        seed=seed,
    )
    t1 = time.perf_counter()
    train_time = t1 - t0

    train_losses = np.asarray(history_raw["train_loss"], dtype=float)
    val_losses = np.asarray(history_raw["val_loss"], dtype=float)

    final_train_loss = float(train_losses[-1]) if len(train_losses) else float("nan")

    history = {
        "train_loss": train_losses,
        "val_loss_noisy": val_losses,
    }

    return model, history, train_time, final_train_loss, float(final_noisy), float(final_clean)


# ======================================================================
# Runge dataset
# ======================================================================

def generate_runge_dataset(
    N: int = N_POINTS,
    test_size: float = TEST_SIZE,
    noise_std: float = NOISE_STD,
    seed: int = SEED,
):
    """
    Generates a dataset for Runge:
    - X in [-1, 1], N equally spaced points
    - y_clean = f(x)
    - y_noisy = f(x) + Gaussian noise N(0, noise_std^2)
    """
    rng = np.random.default_rng(seed)
    X = np.linspace(-1, 1, N).reshape(-1, 1)

    y_clean = runge(X, noise_std=0.0, rng=rng).reshape(-1, 1)
    y_noisy = runge(X, noise_std=noise_std, rng=rng).reshape(-1, 1)

    X_train, X_val, y_train_clean, y_val_clean, y_train_noisy, y_val_noisy = train_test_split(
        X, y_clean, y_noisy, test_size=test_size, random_state=seed
    )

    return X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean


# ======================================================================
# Orchestration
# ======================================================================

def save_history_csv(impl: str, n_hidden: int, width: int, history: Dict[str, np.ndarray]) -> Path:
    """Save per-epoch curves used for convergence plots."""
    df = pd.DataFrame({
        "epoch": np.arange(1, len(history["train_loss"]) + 1, dtype=int),
        "train_loss": history["train_loss"],
        "val_loss_noisy": history["val_loss_noisy"],
    })
    out = CSV_DIR / f"histories_{impl}_h{n_hidden}_w{width}.csv"
    df.to_csv(out, index=False)
    return out


def save_dataset_csv(X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean) -> Path:
    """Save the exact dataset splits used by the experiment."""
    df_train = pd.DataFrame({
        "split": "train",
        "X": X_train.flatten(),
        "y_noisy": y_train_noisy.flatten(),
        # we do not train on clean labels, but store them for completeness:
    })
    df_val = pd.DataFrame({
        "split": "val",
        "X": X_val.flatten(),
        "y_noisy": y_val_noisy.flatten(),
        "y_clean": y_val_clean.flatten(),
    })
    df = pd.concat([df_train, df_val], ignore_index=True)
    out = CSV_DIR / "dataset.csv"
    df.to_csv(out, index=False)
    return out


def plot_and_save_convergence(repr_files: Dict[str, Path], n_hidden: int, width: int) -> Path:
    """Plot convergence curves for a representative layout, using the saved CSVs."""
    plt.figure(figsize=(7.5, 4.5))
    for impl, path in repr_files.items():
        df = pd.read_csv(path)
        plt.plot(df["epoch"], df["train_loss"], label=f"{impl} train")
        plt.plot(df["epoch"], df["val_loss_noisy"], linestyle="--", label=f"{impl} val (noisy)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Convergence curves — layout [1, {width}×{n_hidden}, 1]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    out = FIG_DIR / "convergence_curves_repr.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_and_save_time_vs_params(summary_df: pd.DataFrame) -> Path:
    """Plot training time vs parameter count (log-log), and save CSV for plotted data."""
    df = summary_df.copy()
    # data used for the plot:
    tvp = df[["impl", "params", "train_time"]].sort_values(["impl", "params"])
    tvp.to_csv(CSV_DIR / "time_vs_params.csv", index=False)

    plt.figure(figsize=(6.5, 4.8))
    for impl, g in tvp.groupby("impl"):
        plt.plot(g["params"], g["train_time"], marker="o", label=impl)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("# parameters (log)")
    plt.ylabel("Training time [s] (log)")
    plt.title("Training time vs #parameters (log-log)")
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend()
    out = FIG_DIR / "time_vs_params_loglog.png"
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def plot_and_save_final_mse(summary_df: pd.DataFrame) -> Path:
    """Plot final validation MSE on clean data per layout & impl; also save CSV used."""
    df = summary_df.copy()
    # Build a human-friendly layout label and save the exact plotted table
    df["layout"] = df.apply(lambda r: f"h{int(r['n_hidden'])}_w{int(r['width'])}", axis=1)
    fm = df[["impl", "layout", "params", "final_val_loss_clean"]].sort_values(["layout", "impl"])
    fm.to_csv(CSV_DIR / "final_mse_clean.csv", index=False)

    # Wide table for grouped bars
    pivot = fm.pivot(index="layout", columns="impl", values="final_val_loss_clean").fillna(np.nan)
    ax = pivot.plot(kind="bar", figsize=(8.5, 4.8))
    ax.set_ylabel("Final MSE (validation clean)")
    ax.set_title("Final validation MSE (clean) per layout/implementation")
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=0)
    plt.tight_layout()
    out = FIG_DIR / "final_mse_clean.png"
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main():
    # ---------------- Dataset ----------------
    X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean = generate_runge_dataset(
        N_POINTS, TEST_SIZE, NOISE_STD, SEED
    )
    save_dataset_csv(X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean)

    results: List[ExperimentResult] = []
    history_index_rows = [] # to track where each history is saved
    # Optional: pick a representative layout for the convergence-curve figure.
    # If the desired (2, 50) is absent, we fall back to the first layout.
    repr_choice = (2, 50) if (2, 50) in LAYOUTS else LAYOUTS[0]
    repr_hist_files: Dict[str, Path] = {}

    # ------------- Sweep layouts -------------
    for (n_hidden, width) in LAYOUTS:
        layout = build_layout(n_hidden, width)
        params = count_params_from_layout(layout)

        # --- Custom FFNN ---
        net, hist_f, t_ffnn, tr_ffnn, vnoisy_ffnn, vclean_ffnn = train_ffnn_single_layout(
            layout, X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean,
            epochs=EPOCHS, lr=LR, lam_l1=LAM_L1, lam_l2=LAM_L2, batches=BATCHES, seed=SEED
        )

        # Convergence check (won't raise if very noisy; feel free to relax/remove if needed)
        try:
            assert_convergence(hist_f["train_loss"], f"ffnn h{n_hidden} w{width}", min_drop=0.2)
        except AssertionError as e:
            print(str(e))

        # Save per-epoch curves for FFNN
        ffnn_hist_path = save_history_csv("ffnn", n_hidden, width, hist_f)
        history_index_rows.append({
            "impl": "ffnn",
            "n_hidden": n_hidden,
            "width": width,
            "params": params,
            "epochs": len(hist_f["train_loss"]),
            "csv_path": str(ffnn_hist_path),
        })
        if (n_hidden, width) == repr_choice:
            repr_hist_files["ffnn"] = ffnn_hist_path

        results.append(ExperimentResult(
            impl="ffnn",
            n_hidden=n_hidden,
            width=width,
            params=params,
            epochs=EPOCHS,
            train_time=t_ffnn,
            final_train_loss=tr_ffnn,
            final_val_loss_noisy=vnoisy_ffnn,
            final_val_loss_clean=vclean_ffnn,
        ))

        # --- PyTorch ---
        model_t, hist_t, t_torch, tr_torch, vnoisy_t, vclean_t = train_torch_single_layout(
            layout, X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean,
            epochs=EPOCHS, lr=LR, lam_l1=LAM_L1, lam_l2=LAM_L2, batches=BATCHES, seed=SEED
        )

        try:
            assert_convergence(hist_t["train_loss"], f"torch h{n_hidden} w{width}", min_drop=0.2)
        except AssertionError as e:
            print(str(e))

        torch_hist_path = save_history_csv("torch", n_hidden, width, hist_t)
        history_index_rows.append({
            "impl": "torch",
            "n_hidden": n_hidden,
            "width": width,
            "params": params,
            "epochs": len(hist_t["train_loss"]),
            "csv_path": str(torch_hist_path),
        })
        if (n_hidden, width) == repr_choice:
            repr_hist_files["torch"] = torch_hist_path

        results.append(ExperimentResult(
            impl="torch",
            n_hidden=n_hidden,
            width=width,
            params=params,
            epochs=EPOCHS,
            train_time=t_torch,
            final_train_loss=tr_torch,
            final_val_loss_noisy=vnoisy_t,
            final_val_loss_clean=vclean_t,
        ))

        # Cross-impl performance sanity note
        assert_similar_performance(vclean_ffnn, vclean_t, name1="ffnn", name2="torch")

    # ---------------- Summary CSV ----------------
    summary_df = pd.DataFrame([asdict(r) for r in results])
    summary_path = CSV_DIR / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Also save a convenient index of where per-epoch histories are stored
    hist_index_df = pd.DataFrame(history_index_rows)
    hist_index_df.to_csv(CSV_DIR / "histories_index.csv", index=False)

    # ---------------- Plots (+ CSVs for plotted data) ----------------
    # 1) Convergence curves for representative layout (reads from the saved history CSVs)
    if len(repr_hist_files) == 2: # both ffnn & torch available
        plot_and_save_convergence(repr_hist_files, n_hidden=repr_choice[0], width=repr_choice[1])

    # 2) Training time vs #params (and save CSV used)
    plot_and_save_time_vs_params(summary_df)

    # 3) Final MSE (validation clean) per layout & impl (and save CSV used)
    plot_and_save_final_mse(summary_df)

    print(f"[OK] CSVs saved under: {CSV_DIR.resolve()}")
    print(f"[OK] Figures saved under: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()