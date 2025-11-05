# -----------------------------------------------------------------------------------------
# Part B — Comparing the best OLS configuration with a FFNN on the Runge function
# -----------------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from src.OLS_utils import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE, R2_score, seed
)
from tqdm import tqdm

# Assume the following are imported from their respective files
from src.FFNN import FFNN
from src.scheduler import Adam  # or another scheduler
from src.activation_functions import sigmoid, identity, LRELU
from src.cost_functions import CostOLS

# -----------------------------
# Settings
# -----------------------------
# NOTE: rimpiazza questi con i best effettivi letti dal CSV della Part A quando disponibile
best_n = 1500
best_degree = 10

noise = 0.03
N_RUNS = 30

# NN hyperparams
epochs = 1500
batches = 32
lam_l1 = 0.0
lam_l2 = 0.00
eta = 0.01
rho = 0.9
rho2 = 0.999
dimensions = (1, 30, 30, 1)  # 1 input, 50 hidden, 1 output

# -----------------------------
# Output dirs
# -----------------------------
OUT = Path("output/OLS_vs_FFNN")
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Storage
# -----------------------------
mse_train_runs_ols = np.zeros(N_RUNS)
mse_test_runs_ols  = np.zeros(N_RUNS)
R2_train_runs_ols  = np.zeros(N_RUNS)
R2_test_runs_ols   = np.zeros(N_RUNS)

mse_train_runs_nn = np.zeros(N_RUNS)
mse_test_runs_nn  = np.zeros(N_RUNS)
R2_train_runs_nn  = np.zeros(N_RUNS)
R2_test_runs_nn   = np.zeros(N_RUNS)

train_losses = np.full((epochs, N_RUNS), np.nan)
val_losses   = np.full((epochs, N_RUNS), np.nan)

# For plotting fits from last run
x_plot = None
y_plot = None
y_ols_plot = None
y_nn_plot = None

print(
    f">>> Starting Part B | best_n={best_n} | best_degree={best_degree} | "
    f"runs={N_RUNS} | noise={noise} | seed_base={seed}"
)

# -----------------------------
# Experiment
# -----------------------------
for r in tqdm(range(N_RUNS), desc="Runs", unit="run"):
    # Seed per run
    np.random.seed(seed + r)

    # Generate Runge data
    x = np.linspace(-1, 1, best_n)
    y = runge_function(x, noise=noise)

    # Split & scale (si assume che split_scale ritorni già x_train/x_test e y_train/y_test nelle corrette scale)
    x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)

    # -------------------
    # OLS part
    # -------------------
    # Costruzione feature polinomiali + scaling colonne (basato sul train)
    X_train, col_means, col_stds = polynomial_features_scaled(
        x_train.flatten(), best_degree, return_stats=True
    )
    X_test = polynomial_features_scaled(
        x_test.flatten(), best_degree, col_means=col_means, col_stds=col_stds
    )

    theta = OLS_parameters(X_train, y_train)

    y_train_pred_ols = (X_train @ theta).reshape(-1)
    y_test_pred_ols  = (X_test  @ theta).reshape(-1)

    mse_train_runs_ols[r] = MSE(y_train, y_train_pred_ols)
    mse_test_runs_ols[r]  = MSE(y_test,  y_test_pred_ols)
    R2_train_runs_ols[r]  = R2_score(y_train, y_train_pred_ols)
    R2_test_runs_ols[r]   = R2_score(y_test,  y_test_pred_ols)

    # -------------------
    # FFNN part
    # -------------------
    nn = FFNN(
        dimensions,
        hidden_func=LRELU,
        output_func=identity,
        cost_func=CostOLS,
        seed=seed + r
    )
    scheduler = Adam(eta=eta, rho=rho, rho2=rho2)

    scores = nn.fit(
        X=x_train.reshape(-1, 1),
        t=y_train.reshape(-1, 1),
        scheduler=scheduler,
        batches=batches,
        epochs=epochs,
        lam_l1=lam_l1,
        lam_l2=lam_l2,
        X_val=x_test.reshape(-1, 1),
        t_val=y_test.reshape(-1, 1)
    )

    # Salvataggio learning curves
    # (ci si aspetta che scores contenga "train_errors" e "val_errors" di lunghezza = epochs)
    tr = np.asarray(scores.get("train_errors", []), dtype=float).reshape(-1)
    vl = np.asarray(scores.get("val_errors",   []), dtype=float).reshape(-1)
    L = min(len(tr), epochs)
    train_losses[:L, r] = tr[:L]
    val_losses[:L, r]   = vl[:L]

    # Predizioni
    y_train_pred_nn = nn.predict(x_train.reshape(-1, 1)).reshape(-1)
    y_test_pred_nn  = nn.predict(x_test.reshape(-1, 1)).reshape(-1)

    mse_train_runs_nn[r] = MSE(y_train, y_train_pred_nn)
    mse_test_runs_nn[r]  = MSE(y_test,  y_test_pred_nn)
    R2_train_runs_nn[r]  = R2_score(y_train, y_train_pred_nn)
    R2_test_runs_nn[r]   = R2_score(y_test,  y_test_pred_nn)

    # Salvataggio per l'ultimo run (plot)
    if r == N_RUNS - 1:
        x_plot     = x_test.copy()
        y_plot     = y_test.copy()
        y_ols_plot = y_test_pred_ols.copy()
        y_nn_plot  = y_test_pred_nn.copy()

# -----------------------------
# Aggregazione metriche
# -----------------------------
def mean_std(a):
    return float(np.nanmean(a)), float(np.nanstd(a, ddof=1))

mse_train_ols, mse_train_std_ols = mean_std(mse_train_runs_ols)
mse_test_ols,  mse_test_std_ols  = mean_std(mse_test_runs_ols)
R2_train_ols,  R2_train_std_ols  = mean_std(R2_train_runs_ols)
R2_test_ols,   R2_test_std_ols   = mean_std(R2_test_runs_ols)

mse_train_nn, mse_train_std_nn = mean_std(mse_train_runs_nn)
mse_test_nn,  mse_test_std_nn  = mean_std(mse_test_runs_nn)
R2_train_nn,  R2_train_std_nn  = mean_std(R2_train_runs_nn)
R2_test_nn,   R2_test_std_nn   = mean_std(R2_test_runs_nn)

# Learning curves medie (+ std)
avg_train_loss = np.nanmean(train_losses, axis=1)
avg_val_loss   = np.nanmean(val_losses,   axis=1)
std_train_loss = np.nanstd(train_losses,  axis=1, ddof=1)
std_val_loss   = np.nanstd(val_losses,    axis=1, ddof=1)

# -----------------------------
# Print risultati in console
# -----------------------------
print("\nOLS Results:")
print(f"Train MSE: {mse_train_ols:.4f} ± {mse_train_std_ols:.4f}")
print(f"Test  MSE: {mse_test_ols:.4f} ± {mse_test_std_ols:.4f}")
print(f"Train  R2: {R2_train_ols:.4f} ± {R2_train_std_ols:.4f}")
print(f"Test   R2: {R2_test_ols:.4f} ± {R2_test_std_ols:.4f}")

print("\nFFNN Results:")
print(f"Train MSE: {mse_train_nn:.4f} ± {mse_train_std_nn:.4f}")
print(f"Test  MSE: {mse_test_nn:.4f} ± {mse_test_std_nn:.4f}")
print(f"Train  R2: {R2_train_nn:.4f} ± {R2_train_std_nn:.4f}")
print(f"Test   R2: {R2_test_nn:.4f} ± {R2_test_std_nn:.4f}")

# -----------------------------
# Salvataggi tabelle
# -----------------------------
summary = pd.DataFrame({
    "model": ["OLS", "FFNN"],
    "train_mse_mean": [mse_train_ols, mse_train_nn],
    "train_mse_std":  [mse_train_std_ols, mse_train_std_nn],
    "test_mse_mean":  [mse_test_ols, mse_test_nn],
    "test_mse_std":   [mse_test_std_ols, mse_test_std_nn],
    "train_R2_mean":  [R2_train_ols, R2_train_nn],
    "train_R2_std":   [R2_train_std_ols, R2_train_std_nn],
    "test_R2_mean":   [R2_test_ols, R2_test_nn],
    "test_R2_std":    [R2_test_std_ols, R2_test_std_nn],
})
summary.to_csv(TAB / "part_b_summary.csv", index=False)

# -----------------------------
# Plots
# -----------------------------
# Plot 1: Average loss vs epochs for FFNN
fig, ax = plt.subplots(figsize=(8, 6))
epoch_range = np.arange(len(avg_train_loss))
ax.plot(epoch_range, avg_train_loss, label='Average Train MSE')
ax.plot(epoch_range, avg_val_loss,   label='Average Val MSE')
ax.fill_between(epoch_range,
                avg_train_loss - std_train_loss,
                avg_train_loss + std_train_loss,
                alpha=0.2, label='Train std')
ax.fill_between(epoch_range,
                avg_val_loss - std_val_loss,
                avg_val_loss + std_val_loss,
                alpha=0.2, label='Val std')
ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')
ax.set_title('FFNN Learning Curves (Averaged over Runs)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "ffnn_loss_epochs.png", dpi=150)
plt.close(fig)

# Plot 2: Runge function fits from last run on test set
# (FIX: forza 1D per evitare fancy indexing (n,1,1))
if x_plot is not None:
    x1d   = np.asarray(x_plot).reshape(-1)
    y1d   = np.asarray(y_plot).reshape(-1)
    yols1 = np.asarray(y_ols_plot).reshape(-1)
    ynn1  = np.asarray(y_nn_plot).reshape(-1)

    order = np.argsort(x1d)
    x_sorted      = x1d[order]
    y_sorted      = y1d[order]
    y_ols_sorted  = yols1[order]
    y_nn_sorted   = ynn1[order]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x_sorted, y_sorted, label='Test Data (with noise)', alpha=0.5)
    ax.plot(x_sorted, y_ols_sorted, label=f'OLS degree {best_degree}', linewidth=2)
    ax.plot(x_sorted, y_nn_sorted, label='FFNN', linewidth=2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Runge Function — Fits on Test Set (Last Run)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "runge_fits_test.png", dpi=150)
    plt.close(fig)

# (Opzionale) True function su griglia densa — solo come riferimento visuale
# ATTENZIONE: se split_scale scala/centra x e y, questa curva NON è direttamente comparabile
# con le predizioni (che sono nello spazio scalato/centrato). La lasciamo come figura separata.
dense_x = np.linspace(-1, 1, 1000)
y_true_dense = runge_function(dense_x, noise=False)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(dense_x, y_true_dense, linewidth=2, label='True Runge (noise-free)')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('True Runge Function (Reference)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "runge_true_reference.png", dpi=150)
plt.close(fig)

print(f"\nPart B done. Aggregated over {N_RUNS} runs.")
print(f"Figures -> {FIG}")
print(f"Tables  -> {TAB}")
print("Note: run Part A first and load the actual best (n, degree) from CSV quando disponibile.")
