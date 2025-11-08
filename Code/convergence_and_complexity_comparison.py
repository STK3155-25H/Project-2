
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch

# ---- Imports from codebase ----
from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import RELU, identity

from complexity_analysis_TORCH import train_one_model


# ======================================================================
#  Experiment constants
# ======================================================================

N_POINTS = 200           # linspace with 200 points in [-1, 1]
EPOCHS = 1500            # training epochs
LR = 1e-3                # learning rate
LAM_L1 = 0.0             # L1 regularization
LAM_L2 = 0.0             # L2 regularization
BATCHES = 100            # number of mini-batches
NOISE_STD = 0.03         # Gaussian noise level
TEST_SIZE = 0.25         # 75% train, 25% validation
SEED = 314               # reproducibility seed


# ======================================================================
#  Utilities: dataset, layout, parameter counting
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
#  Convergence and consistency checks
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
        msg = (f"[{name1} vs {name2}] WARNING: performances differ significantly "
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

    train_losses = np.asarray(
        history_raw.get("train_errors", history_raw.get("train_loss", [])),
        dtype=float,
    )
    val_losses_noisy = np.asarray(
        history_raw.get("val_errors", history_raw.get("val_loss", [])),
        dtype=float,
    )

    # Final evaluation (noisy & clean validation)
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

    final_train_loss = float(train_losses[-1]) if len(train_losses) > 0 else float("nan")

    history = {"train_loss": train_losses, "val_loss_noisy": val_losses}

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
  
    rng = np.random.default_rng(seed)
    X = np.linspace(-1, 1, N).reshape(-1, 1)

    y_clean = runge(X, noise_std=0.0, rng=rng).reshape(-1, 1)
    y_noisy = runge(X, noise_std=noise_std, rng=rng).reshape(-1, 1)

    X_train, X_val, y_train_clean, y_val_clean, y_train_noisy, y_val_noisy = train_test_split(
        X, y_clean, y_noisy, test_size=test_size, random_state=seed
    )

    return X_train, y_train_noisy, X_val, y_val_noisy, y_val_clean
