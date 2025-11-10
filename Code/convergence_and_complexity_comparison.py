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

from complexity_analysis_TORCH import train_one_model


# ======================================================================
#  Experiment constants (required parameters)
# ======================================================================

N_POINTS = 200
EPOCHS = 1500
LR = 1e-3
LAM_L1 = 0.0
LAM_L2 = 0.0
BATCHES = 100
NOISE_STD = 0.03
TEST_SIZE = 0.25
SEED = 314

# Output directories
BASE_DIR = Path("output/benchmark")
CSV_DIR = BASE_DIR / "csv"
FIG_DIR = BASE_DIR / "figs"
CSV_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ======================================================================
#  Utils: dataset, layout, parameters
# ======================================================================

def runge(x: np.ndarray, noise_std: float = 0.0, rng=None) -> np.ndarray:
    rng = rng or np.random.default_rng()
    noise = rng.normal(0.0, noise_std, size=x.shape) if noise_std > 0 else 0.0
    return 1.0 / (1.0 + 25.0 * x**2) + noise


def build_layout(n_hidden: int, width: int) -> List[int]:
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]


def count_params_from_layout(layout: List[int]) -> int:
    return int(sum((layout[i] + 1) * layout[i + 1] for i in range(len(layout) - 1)))


@dataclass
class ExperimentResult:
    impl: str                # "ffnn" or "torch"
    n_hidden: int
    width: int
    params: int
    epochs: int
    train_time: float
    final_train_loss: float
    final_val_loss_noisy: float
    final_val_loss_clean: float


# ======================================================================
#  Convergence checks
# ======================================================================

def assert_convergence(losses: np.ndarray, name: str, min_drop: float = 0.2) -> None:
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
    v1, v2 = float(val1), float(val2)
    if v1 == 0 and v2 == 0:
        return

    ratio = max(v1, v2) / (min(v1, v2) + 1e-12)

    if ratio > 1.0 + rel_tol:
        msg = (f"[{name1} vs {name2}] WARNING: very different performance "
               f"(val1={v1:.4e}, val2={v2:.4e}, ratio={ratio:.3f})")
        print(msg)


# ======================================================================
#  Training helpers
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

    # Standardize naming (train_errors / train_loss, etc.)
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

    final_train_loss = float(train_losses[-1]) if len(train_losses) > 0 else float("nan")

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_noisy_t = torch.from_numpy(y_train_noisy.astype(np.float32))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_noisy_t = torch.from_numpy(y_val_noisy.astype(np.float32))
    y_val_clean_t = torch.from_numpy(y_val_clean.astype(np.float32))

    t0 = time.perf_counter()
    model, history_raw, final_noisy, final_clean = train_one_model(
        layout=layout,
        act_name="RELU",   # same activation as FFNN
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

    final_train_loss = float(train_losses[-1]) if len(train_losses) > 0 else float("nan")

    history = {
        "train_loss": train_losses,
        "val_loss_noisy": val_losses,
    }

    return model, history, train_time, final_train_loss, float(final_noisy), float(final_clean)


# ======================================================================
#  Runge dataset
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
#  Main experiment
# ======================================================================

def run_experiments():
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Shared dataset for all layouts
    X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean = generate_runge_dataset()

    # Save dataset splits as CSV
    ds_train = pd.DataFrame({"x": X_train.ravel(), "y_noisy": y_train_noisy.ravel()})
    ds_val = pd.DataFrame({
        "x": X_val.ravel(),
        "y_noisy": y_val_noisy.ravel(),
        "y_clean": y_val_clean.ravel(),
    })
    ds_train.to_csv(CSV_DIR / "dataset_train.csv", index=False)
    ds_val.to_csv(CSV_DIR / "dataset_val.csv", index=False)

    # Larger / more varied layouts:
    settings = [
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
        (5, 128),
    ]

    all_results: List[ExperimentResult] = []

    # Layout used for detailed convergence plots
    layout_for_detailed_plots = (3, 64)
    stored_histories: Dict[str, Dict[str, np.ndarray]] = {}

    for idx, (n_hidden, width) in enumerate(settings, 1):
        layout = build_layout(n_hidden, width)
        params = count_params_from_layout(layout)
        print(f"\n=== [{idx}/{len(settings)}] Layout: hidden={n_hidden}, width={width}, params~{params} ===")

        # --------------------------------------------------------------
        #  Custom FFNN
        # --------------------------------------------------------------
        net, hist_ffnn, time_ffnn, tr_ffnn, val_noisy_ffnn, val_clean_ffnn = train_ffnn_single_layout(
            layout,
            X_train,
            y_train_noisy,
            X_val,
            y_val_noisy,
            y_val_clean,
        )

        # Convergence assertion for FFNN (slightly stricter)
        assert_convergence(hist_ffnn["train_loss"], f"FFNN (hidden={n_hidden}, width={width})", min_drop=0.2)

        # Save per-epoch history CSV
        df_hist_ff = pd.DataFrame({
            "epoch": np.arange(1, len(hist_ffnn["train_loss"]) + 1),
            "train_loss": hist_ffnn["train_loss"],
            "val_loss_noisy": hist_ffnn["val_loss_noisy"] if len(hist_ffnn["val_loss_noisy"]) else np.nan,
        })
        df_hist_ff.to_csv(CSV_DIR / f"histories_ffnn_h{n_hidden}_w{width}.csv", index=False)

        all_results.append(
            ExperimentResult(
                impl="ffnn",
                n_hidden=n_hidden,
                width=width,
                params=params,
                epochs=EPOCHS,
                train_time=time_ffnn,
                final_train_loss=tr_ffnn,
                final_val_loss_noisy=val_noisy_ffnn,
                final_val_loss_clean=val_clean_ffnn,
            )
        )

        if (n_hidden, width) == layout_for_detailed_plots:
            stored_histories["ffnn"] = hist_ffnn

        # --------------------------------------------------------------
        #  PyTorch MLP
        # --------------------------------------------------------------
        model_t, hist_torch, time_torch, tr_t, val_noisy_t, val_clean_t = train_torch_single_layout(
            layout,
            X_train,
            y_train_noisy,
            X_val,
            y_val_noisy,
            y_val_clean,
        )

        # Convergence assertion for Torch (slightly looser)
        assert_convergence(hist_torch["train_loss"], f"PyTorch (hidden={n_hidden}, width={width})", min_drop=0.1)

        # Save per-epoch history CSV
        df_hist_t = pd.DataFrame({
            "epoch": np.arange(1, len(hist_torch["train_loss"]) + 1),
            "train_loss": hist_torch["train_loss"],
            "val_loss_noisy": hist_torch["val_loss_noisy"] if len(hist_torch["val_loss_noisy"]) else np.nan,
        })
        df_hist_t.to_csv(CSV_DIR / f"histories_torch_h{n_hidden}_w{width}.csv", index=False)

        # Performance comparison (only warning, no blocking assert)
        assert_similar_performance(
            val_clean_ffnn, val_clean_t,
            rel_tol=0.5,
            name1="ffnn",
            name2="torch",
        )

        all_results.append(
            ExperimentResult(
                impl="torch",
                n_hidden=n_hidden,
                width=width,
                params=params,
                epochs=EPOCHS,
                train_time=time_torch,
                final_train_loss=tr_t,
                final_val_loss_noisy=val_noisy_t,
                final_val_loss_clean=val_clean_t,
            )
        )

        if (n_hidden, width) == layout_for_detailed_plots:
            stored_histories["torch"] = hist_torch

    # ==================================================================
    #   Scientific analysis of results + CSV dumps
    # ==================================================================
    df = pd.DataFrame([asdict(r) for r in all_results])
    print("\n=== SUMMARY RESULTS ===")
    print(df)

    # Save summary CSV
    df.to_csv(CSV_DIR / "summary.csv", index=False)

    # Average time per epoch
    df["time_per_epoch"] = df["train_time"] / df["epochs"]
    df[["impl", "n_hidden", "width", "params", "epochs", "train_time", "time_per_epoch",
        "final_train_loss", "final_val_loss_noisy", "final_val_loss_clean"]].to_csv(
        CSV_DIR / "time_vs_params.csv", index=False
    )

    # Final MSE (clean & noisy) per layout/impl
    df_final = df[["impl", "n_hidden", "width", "final_val_loss_clean", "final_val_loss_noisy", "final_train_loss"]]
    df_final.to_csv(CSV_DIR / "final_mse_clean.csv", index=False)

    # Torch/FFNN time ratio for the same layout (print only)
    print("\nTime_per_epoch ratio (torch / ffnn) for each layout:")
    for (n_hidden, width) in settings:
        df_sub = df[(df["n_hidden"] == n_hidden) & (df["width"] == width)]
        if len(df_sub) == 2:
            t_ffnn = df_sub[df_sub["impl"] == "ffnn"]["time_per_epoch"].iloc[0]
            t_torch = df_sub[df_sub["impl"] == "torch"]["time_per_epoch"].iloc[0]
            ratio = t_torch / t_ffnn
            print(f"  hidden={n_hidden}, width={width}: ratio={ratio:.3f}")

    # ==================================================================
    #   PLOT 1: train/val convergence curves (noisy) for one layout
    # ==================================================================
    if "ffnn" in stored_histories and "torch" in stored_histories:
        plt.figure(figsize=(8, 5))
        ep_ff = np.arange(1, len(stored_histories["ffnn"]["train_loss"]) + 1)
        ep_t = np.arange(1, len(stored_histories["torch"]["train_loss"]) + 1)

        plt.plot(ep_ff, stored_histories["ffnn"]["train_loss"], label="FFNN train")
        if len(stored_histories["ffnn"]["val_loss_noisy"]) == len(ep_ff):
            plt.plot(ep_ff, stored_histories["ffnn"]["val_loss_noisy"], "--", label="FFNN val (noisy)")

        plt.plot(ep_t, stored_histories["torch"]["train_loss"], label="Torch train")
        if len(stored_histories["torch"]["val_loss_noisy"]) == len(ep_t):
            plt.plot(ep_t, stored_histories["torch"]["val_loss_noisy"], "--", label="Torch val (noisy)")

        n_hidden, width = (3, 64)
        plt.title(f"Train/Validation convergence — hidden={n_hidden}, width={width}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE")
        plt.yscale("log")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"convergence_curves_h{n_hidden}_w{width}.png", dpi=200)
        plt.close()

    # ==================================================================
    #   PLOT 2: training time vs number of parameters
    # ==================================================================
    plt.figure(figsize=(7, 5))
    for impl, marker in [("ffnn", "o"), ("torch", "s")]:
        df_impl = df[df["impl"] == impl].sort_values("params")
        plt.plot(
            df_impl["params"].to_numpy(),
            df_impl["train_time"].to_numpy(),
            marker,
            label=impl.upper(),
        )
    plt.xlabel("Number of parameters (estimate)")
    plt.ylabel("Training time (s)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Empirical analysis of time complexity (1500 epochs)")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "time_vs_params.png", dpi=200)
    plt.close()

    # ==================================================================
    #   PLOT 3: final (clean) MSE per implementation and layout
    # ==================================================================
    plt.figure(figsize=(9, 5))
    x_labels = [f"h={h},w={w}" for (h, w) in settings]
    x = np.arange(len(settings))
    width_bar = 0.35

    mse_ffnn = []
    mse_torch = []
    for (h, w) in settings:
        df_sub = df[(df["n_hidden"] == h) & (df["width"] == w)]
        mse_ffnn.append(df_sub[df_sub["impl"] == "ffnn"]["final_val_loss_clean"].iloc[0])
        mse_torch.append(df_sub[df_sub["impl"] == "torch"]["final_val_loss_clean"].iloc[0])

    mse_ffnn = np.array(mse_ffnn)
    mse_torch = np.array(mse_torch)

    plt.bar(x - width_bar / 2, mse_ffnn, width_bar, label="FFNN")
    plt.bar(x + width_bar / 2, mse_torch, width_bar, label="Torch")

    plt.xticks(x, x_labels, rotation=30, ha="right")
    plt.ylabel("Final MSE (validation clean)")
    plt.title("Comparison of final accuracy (1500 epochs, Runge)")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "final_mse_clean.png", dpi=200)
    plt.close()

    print(f"\nCSV saved under: {CSV_DIR.resolve()}")
    print(f"Figures saved under: {FIG_DIR.resolve()}")


if __name__ == "__main__":
    run_experiments()
