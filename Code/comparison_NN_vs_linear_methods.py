# -----------------------------------------------------------------------------------------
# Part B — Comparing the best OLS configuration with Ridge, Lasso, and a FFNN on Runge
# -----------------------------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from config import MODELS_DIR, OLS_VS_FFNN_OUTPUT_DIR
# ======= Your utils (as in your file) =======
from src.OLS_utils import (
    runge_function, split_scale, polynomial_features_scaled,
    OLS_parameters, MSE, R2_score, seed,
    Gradient_descent_advanced, Ridge_parameters, MSE_Bias_Variance
)

# NN bits
from src.FFNN import FFNN
from src.scheduler import Adam
from src.activation_functions import sigmoid, identity, LRELU
from src.cost_functions import CostOLS

# -----------------------------
# Settings
# -----------------------------
# NOTE: replace these with the actual bests read from the Part A CSV when available
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
dimensions = (1, 30, 30, 1)

# Ridge / Lasso hyperparam search (log-spaced, reasonably wide)
RIDGE_LAMS = np.logspace(-6, 2, 9) # 1e-6 ... 1e+2
LASSO_LAMS = np.logspace(-6, -1, 6) # 1e-6 ... 1e-1 (subgradient is happier with smaller l1)
# Optimizer settings for Lasso subgradient
LASSO_OPT = dict(method='adam', lr=0.01, n_iter=4000, beta=0.9, epsilon=1e-8, use_sgd=False)

# Inner validation split from the training set (for Ridge/Lasso model selection)
VAL_FRAC = 0.2 # 20% of train goes to val for picking lam
# -----------------------------
# Output dirs
# -----------------------------
OUT = Path(OLS_VS_FFNN_OUTPUT_DIR)
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Extra metrics
# -----------------------------

def mae(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(MSE(y_true, y_pred)))

def explained_variance(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    num = np.var(y_true - y_pred)
    den = np.var(y_true)
    return float(1.0 - num/den) if den > 0 else np.nan

def adjusted_R2(y_true, y_pred, p):
    """
    Adjusted R^2 for linear models (p = number of regressors incl. intercept).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    n = y_true.shape[0]
    if n <= p + 1:
        return np.nan
    r2 = R2_score(y_true, y_pred)
    return float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))

def collect_all_metrics(y_tr, y_tr_pred, y_te, y_te_pred, p_linear=None):
    """
    Returns a dict of metrics for train and test.
    p_linear: number of linear regressors (incl. intercept), used only for adjR2 on linear models.
    """
    d = dict(
        train_MSE=float(MSE(y_tr, y_tr_pred)),
        test_MSE=float(MSE(y_te, y_te_pred)),
        train_R2=float(R2_score(y_tr, y_tr_pred)),
        test_R2=float(R2_score(y_te, y_te_pred)),
        train_MAE=mae(y_tr, y_tr_pred),
        test_MAE=mae(y_te, y_te_pred),
        train_RMSE=rmse(y_tr, y_tr_pred),
        test_RMSE=rmse(y_te, y_te_pred),
        train_EVS=explained_variance(y_tr, y_tr_pred),
        test_EVS=explained_variance(y_te, y_te_pred),
    )
    if p_linear is not None:
        d["train_adjR2"] = adjusted_R2(y_tr, y_tr_pred, p_linear)
        d["test_adjR2"] = adjusted_R2(y_te, y_te_pred, p_linear)
    else:
        d["train_adjR2"] = np.nan
        d["test_adjR2"] = np.nan
    return d

# -----------------------------
# Storage (per-run arrays)
# -----------------------------
MODELS = ["OLS", "Ridge", "Lasso", "FFNN"]

per_run = {
    m: {
        "mse_train": np.zeros(N_RUNS),
        "mse_test": np.zeros(N_RUNS),
        "r2_train": np.zeros(N_RUNS),
        "r2_test": np.zeros(N_RUNS),
        "mae_train": np.zeros(N_RUNS),
        "mae_test": np.zeros(N_RUNS),
        "rmse_train":np.zeros(N_RUNS),
        "rmse_test": np.zeros(N_RUNS),
        "evs_train": np.zeros(N_RUNS),
        "evs_test": np.zeros(N_RUNS),
        "adjr2_train":np.full(N_RUNS, np.nan),
        "adjr2_test": np.full(N_RUNS, np.nan),
    } for m in MODELS
}

# track best lambdas per run
ridge_best_lams = np.zeros(N_RUNS)
lasso_best_lams = np.zeros(N_RUNS)

# Learning curves (FFNN)
train_losses = np.full((epochs, N_RUNS), np.nan)
val_losses = np.full((epochs, N_RUNS), np.nan)
# For plotting fits from last run
x_plot = y_plot = None
y_ols_plot = y_nn_plot = None
y_ridge_plot = y_lasso_plot = None

print(
    f">>> Starting Part B | best_n={best_n} | best_degree={best_degree} | "
    f"runs={N_RUNS} | noise={noise} | seed_base={seed}"
)

# -----------------------------
# Experiment
# -----------------------------
for r in tqdm(range(N_RUNS), desc="Runs", unit="run"):
    np.random.seed(seed + r)

    # Generate Runge data
    x = np.linspace(-1, 1, best_n)
    y = runge_function(x, noise=noise)

    # Primary split+scale (X_train/X_test and centered y)
    x_train, x_test, y_train, y_test = split_scale(x, y, random_state=seed + r)

    # ========== OLS ==========
    # Poly features using train stats
    X_train, col_means, col_stds = polynomial_features_scaled(
        x_train.flatten(), best_degree, return_stats=True
    )
    X_test = polynomial_features_scaled(
        x_test.flatten(), best_degree, col_means=col_means, col_stds=col_stds
    )

    theta_ols = OLS_parameters(X_train, y_train)
    y_tr_pred_ols = (X_train @ theta_ols).ravel()
    y_te_pred_ols = (X_test @ theta_ols).ravel()
    p_lin = X_train.shape[1] # for adjR2 in linear models
    met = collect_all_metrics(y_train, y_tr_pred_ols, y_test, y_te_pred_ols, p_linear=p_lin)
    per_run["OLS"]["mse_train"][r] = met["train_MSE"]
    per_run["OLS"]["mse_test"][r] = met["test_MSE"]
    per_run["OLS"]["r2_train"][r] = met["train_R2"]
    per_run["OLS"]["r2_test"][r] = met["test_R2"]
    per_run["OLS"]["mae_train"][r] = met["train_MAE"]
    per_run["OLS"]["mae_test"][r] = met["test_MAE"]
    per_run["OLS"]["rmse_train"][r] = met["train_RMSE"]
    per_run["OLS"]["rmse_test"][r] = met["test_RMSE"]
    per_run["OLS"]["evs_train"][r] = met["train_EVS"]
    per_run["OLS"]["evs_test"][r] = met["test_EVS"]
    per_run["OLS"]["adjr2_train"][r]= met["train_adjR2"]
    per_run["OLS"]["adjr2_test"][r] = met["test_adjR2"]

    # ========== Inner train/val split for Ridge & Lasso ==========
    # Split current train into (train2, val2), then build poly (with stats from train2)
    n_tr = x_train.shape[0]
    idx = np.arange(n_tr)
    np.random.shuffle(idx)
    cut = int((1.0 - VAL_FRAC) * n_tr)
    idx_tr2 = idx[:cut]
    idx_val2 = idx[cut:]

    x_tr2, x_val2 = x_train[idx_tr2].ravel(), x_train[idx_val2].ravel()
    y_tr2, y_val2 = y_train[idx_tr2].ravel(), y_train[idx_val2].ravel()

    X_tr2, cmean2, cstd2 = polynomial_features_scaled(x_tr2, best_degree, return_stats=True)
    X_val2 = polynomial_features_scaled(x_val2, best_degree, col_means=cmean2, col_stds=cstd2)

    # Also prepare (full) train/test design matrices (already computed above: X_train/X_test)

    # ========== Ridge (model selection on val) ==========
    best_lam_ridge = None
    best_val_mse = np.inf
    for lam in RIDGE_LAMS:
        theta = Ridge_parameters(X_tr2, y_tr2, lam=lam, intercept=True)
        val_pred = (X_val2 @ theta).ravel()
        val_mse = MSE(y_val2, val_pred)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lam_ridge = lam

    ridge_best_lams[r] = best_lam_ridge if best_lam_ridge is not None else np.nan
    # Refit ridge on full train with best lambda
    theta_ridge = Ridge_parameters(X_train, y_train, lam=float(best_lam_ridge), intercept=True)
    y_tr_pred_ridge = (X_train @ theta_ridge).ravel()
    y_te_pred_ridge = (X_test @ theta_ridge).ravel()
    met = collect_all_metrics(y_train, y_tr_pred_ridge, y_test, y_te_pred_ridge, p_linear=p_lin)
    for k_old, k_new in [
        ("mse_train","train_MSE"),("mse_test","test_MSE"),
        ("r2_train","train_R2"),("r2_test","test_R2"),
        ("mae_train","train_MAE"),("mae_test","test_MAE"),
        ("rmse_train","train_RMSE"),("rmse_test","test_RMSE"),
        ("evs_train","train_EVS"),("evs_test","test_EVS"),
        ("adjr2_train","train_adjR2"),("adjr2_test","test_adjR2"),
    ]:
        per_run["Ridge"][k_old][r] = met[k_new]

    # ========== Lasso (model selection on val) ==========
    # NOTE: we use your Gradient_descent_advanced(Type=2) subgradient L1.
    # To avoid penalizing intercept, we rely on your implementation that keeps the intercept column in X.
    # (If your gradient_Lasso penalizes intercept, consider zeroing its L1 component there.)
    best_lam_lasso = None
    best_val_mse = np.inf

    for lam in LASSO_LAMS:
        theta_est = Gradient_descent_advanced(
            X_tr2, y_tr2, Type=2, lam=float(lam), **LASSO_OPT, batch_size=X_tr2.shape[0], theta_history=False
        )
        val_pred = (X_val2 @ theta_est).ravel()
        val_mse = MSE(y_val2, val_pred)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_lam_lasso = lam

    lasso_best_lams[r] = best_lam_lasso if best_lam_lasso is not None else np.nan
    # Refit on full train
    theta_lasso = Gradient_descent_advanced(
        X_train, y_train, Type=2, lam=float(best_lam_lasso), **LASSO_OPT, batch_size=X_train.shape[0], theta_history=False
    )
    y_tr_pred_lasso = (X_train @ theta_lasso).ravel()
    y_te_pred_lasso = (X_test @ theta_lasso).ravel()
    met = collect_all_metrics(y_train, y_tr_pred_lasso, y_test, y_te_pred_lasso, p_linear=p_lin)
    for k_old, k_new in [
        ("mse_train","train_MSE"),("mse_test","test_MSE"),
        ("r2_train","train_R2"),("r2_test","test_R2"),
        ("mae_train","train_MAE"),("mae_test","test_MAE"),
        ("rmse_train","train_RMSE"),("rmse_test","test_RMSE"),
        ("evs_train","train_EVS"),("evs_test","test_EVS"),
        ("adjr2_train","train_adjR2"),("adjr2_test","test_adjR2"),
    ]:
        per_run["Lasso"][k_old][r] = met[k_new]

    # ========== FFNN ==========
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

    tr = np.asarray(scores.get("train_errors", []), dtype=float).reshape(-1)
    vl = np.asarray(scores.get("val_errors", []), dtype=float).reshape(-1)
    L = min(len(tr), epochs)
    train_losses[:L, r] = tr[:L]
    val_losses[:L, r] = vl[:L]
    y_tr_pred_nn = nn.predict(x_train.reshape(-1, 1)).reshape(-1)
    y_te_pred_nn = nn.predict(x_test.reshape(-1, 1)).reshape(-1)
    met = collect_all_metrics(y_train, y_tr_pred_nn, y_test, y_te_pred_nn, p_linear=None)
    per_run["FFNN"]["mse_train"][r] = met["train_MSE"]
    per_run["FFNN"]["mse_test"][r] = met["test_MSE"]
    per_run["FFNN"]["r2_train"][r] = met["train_R2"]
    per_run["FFNN"]["r2_test"][r] = met["test_R2"]
    per_run["FFNN"]["mae_train"][r] = met["train_MAE"]
    per_run["FFNN"]["mae_test"][r] = met["test_MAE"]
    per_run["FFNN"]["rmse_train"][r] = met["train_RMSE"]
    per_run["FFNN"]["rmse_test"][r] = met["test_RMSE"]
    per_run["FFNN"]["evs_train"][r] = met["train_EVS"]
    per_run["FFNN"]["evs_test"][r] = met["test_EVS"]
    # adjR2 stays NaN for NN

    # Save for last-run fits
    if r == N_RUNS - 1:
        x_plot = x_test.copy()
        y_plot = y_test.copy()
        y_ols_plot = y_te_pred_ols.copy()
        y_ridge_plot = y_te_pred_ridge.copy()
        y_lasso_plot = y_te_pred_lasso.copy()
        y_nn_plot = y_te_pred_nn.copy()
# -----------------------------
# Aggregation
# -----------------------------

def mean_std(a):
    """
    Safe mean/std that:
      - returns (nan, nan) if all values are NaN
      - uses population if only 1 valid sample (std = 0.0 to avoid ddof issues)
      - never emits warnings
    """
    a = np.asarray(a, dtype=float).ravel()
    mask = ~np.isnan(a)
    n_valid = int(mask.sum())
    if n_valid == 0:
        return (float('nan'), float('nan'))
    m = float(a[mask].mean())
    if n_valid <= 1:
        s = 0.0
    else:
        s = float(a[mask].std(ddof=1))
    return m, s


def summarize_model(stats_dict):
    """
    stats_dict: {metric_name: array_like over runs}
    Produces {metric_mean, metric_std} pairs.
    If an array is all-NaN, the pair becomes (nan, nan) without warnings.
    """
    rows = {}
    for k, v in stats_dict.items():
        mu, sd = mean_std(v)
        rows[k + "_mean"] = mu
        rows[k + "_std"] = sd
    return rows

summary_rows = []
for m in MODELS:
    stats = { # train/test metrics we recorded
        "train_mse": per_run[m]["mse_train"],
        "test_mse": per_run[m]["mse_test"],
        "train_r2": per_run[m]["r2_train"],
        "test_r2": per_run[m]["r2_test"],
        "train_mae": per_run[m]["mae_train"],
        "test_mae": per_run[m]["mae_test"],
        "train_rmse":per_run[m]["rmse_train"],
        "test_rmse": per_run[m]["rmse_test"],
        "train_evs": per_run[m]["evs_train"],
        "test_evs": per_run[m]["evs_test"],
        "train_adjR2": per_run[m]["adjr2_train"],
        "test_adjR2": per_run[m]["adjr2_test"],
    }
    row = {"model": m}
    row.update(summarize_model(stats))
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TAB / "part_b_summary.csv", index=False)

# Save chosen lambdas per run
pd.DataFrame({
    "run": np.arange(N_RUNS),
    "ridge_best_lambda": ridge_best_lams,
    "lasso_best_lambda": lasso_best_lams,
}).to_csv(TAB / "part_b_best_lams_per_run.csv", index=False)
# Learning curves averages (+ std)
avg_train_loss = np.nanmean(train_losses, axis=1)
avg_val_loss = np.nanmean(val_losses, axis=1)
std_train_loss = np.nanstd(train_losses, axis=1, ddof=1)
std_val_loss = np.nanstd(val_losses, axis=1, ddof=1)
# Export learning curves used in the plot
pd.DataFrame({
    "epoch": np.arange(len(avg_train_loss)),
    "avg_train_mse": avg_train_loss,
    "std_train_mse": std_train_loss,
    "avg_val_mse": avg_val_loss,
    "std_val_mse": std_val_loss,
}).to_csv(TAB / "ffnn_learning_curves_avg.csv", index=False)
# Also export per-run test MSEs (used by boxplot & bars)
rows = []
for i, m in enumerate(MODELS):
    for r in range(N_RUNS):
        rows.append({"run": r, "model": m, "test_mse": per_run[m]["mse_test"][r]})
(pd.DataFrame(rows)).to_csv(TAB / "test_mse_per_run.csv", index=False)

# -----------------------------
# Console print (compact)
# -----------------------------
print("\n=== Aggregated metrics (mean ± std) ===")
for m in MODELS:
    mtr, mte = mean_std(per_run[m]["mse_train"]), mean_std(per_run[m]["mse_test"])
    rtr, rte = mean_std(per_run[m]["r2_train"]), mean_std(per_run[m]["r2_test"])
    print(f"{m:6s} | Train MSE {mtr[0]:.4f} ± {mtr[1]:.4f} | Test MSE {mte[0]:.4f} ± {mte[1]:.4f} | "
          f"Train R2 {rtr[0]:.4f} ± {rtr[1]:.4f} | Test R2 {rte[0]:.4f} ± {rte[1]:.4f}")

# -----------------------------
# Plots
# -----------------------------
# 1) FFNN average learning curves (high-contrast colors)
fig, ax = plt.subplots(figsize=(8, 6))
epoch_range = np.arange(len(avg_train_loss))
ax.plot(epoch_range, avg_train_loss, label='Average Train MSE', linewidth=2.0, color='red')
ax.plot(epoch_range, avg_val_loss, label='Average Val MSE', linewidth=2.0, color='blue')
ax.fill_between(epoch_range, avg_train_loss - std_train_loss, avg_train_loss + std_train_loss,
                alpha=0.15, label='Train std', color='red')
ax.fill_between(epoch_range, avg_val_loss - std_val_loss, avg_val_loss + std_val_loss,
                alpha=0.15, label='Val std', color='blue')
ax.set_xlabel('Epochs')
ax.set_ylabel('MSE')
ax.set_title('FFNN Learning Curves (Averaged over Runs)')
ax.legend()
plt.tight_layout()
plt.savefig(FIG / "ffnn_loss_epochs.png", dpi=200)
plt.close(fig)
# 2) Fits on the last run (test set) + CSV EXPORT with gray datapoints & high-contrast lines
if x_plot is not None:
    x1d = np.asarray(x_plot).reshape(-1)
    y1d = np.asarray(y_plot).reshape(-1)
    yols1 = np.asarray(y_ols_plot).reshape(-1)
    yrid1 = np.asarray(y_ridge_plot).reshape(-1)
    ylas1 = np.asarray(y_lasso_plot).reshape(-1)
    ynn1 = np.asarray(y_nn_plot).reshape(-1)
    order = np.argsort(x1d)
    x_sorted = x1d[order]
    y_sorted = y1d[order]
    y_ols_sorted = yols1[order]
    y_ridge_sorted= yrid1[order]
    y_lasso_sorted= ylas1[order]
    y_nn_sorted = ynn1[order]
    # -------- CSV EXPORT --------
    df_plot = pd.DataFrame({
        "x": x_sorted,
        "y_true": y_sorted,
        "OLS": y_ols_sorted,
        "Ridge": y_ridge_sorted,
        "Lasso": y_lasso_sorted,
        "FFNN": y_nn_sorted,
    })
    df_plot.to_csv(TAB / "runge_fits_last_run.csv", index=False)
    print(f"Saved CSV of plotted data -> {TAB / 'runge_fits_last_run.csv'}")
    # -------- PLOT --------
    fig, ax = plt.subplots(figsize=(8, 6))
    # Scatter datapoints in gray
    ax.scatter(x_sorted, y_sorted, label="Test Data (with noise)",
               alpha=0.4, color="gray")
    # High-contrast plot colors
    ax.plot(x_sorted, y_ols_sorted, label=f"OLS deg {best_degree}",
            linewidth=2.5, color="red")
    ax.plot(x_sorted, y_ridge_sorted, label="Ridge (best λ)",
            linewidth=2.5, color="blue")
    ax.plot(x_sorted, y_lasso_sorted, label="Lasso (best λ)",
            linewidth=2.5, color="orange")
    ax.plot(x_sorted, y_nn_sorted, label="FFNN",
            linewidth=2.5, color="green")

    ax.set_xlabel('x'); ax.set_ylabel('y')
    ax.set_title('Runge — Fits on Test Set (Last Run)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG / "runge_fits_test.png", dpi=200)
    plt.close(fig)

# 3) Boxplot of Test MSE across runs (fix labels arg)
fig, ax = plt.subplots(figsize=(8, 6))
data = [per_run[m]["mse_test"] for m in MODELS]
ax.boxplot(data, tick_labels=MODELS, showmeans=True)
ax.set_ylabel("Test MSE")
ax.set_title("Test MSE distribution across runs")
plt.tight_layout()
plt.savefig(FIG / "test_mse_boxplot.png", dpi=200)
plt.close(fig)

# 4) Bar chart of mean±std Test MSE (high-contrast bars: mapped colors)
means = [np.nanmean(per_run[m]["mse_test"]) for m in MODELS]
stds = [np.nanstd(per_run[m]["mse_test"], ddof=1) for m in MODELS]
colors = ["red", "blue", "orange", "green"]
fig, ax = plt.subplots(figsize=(8, 6))
pos = np.arange(len(MODELS))
ax.bar(pos, means, yerr=stds, capsize=5, color=colors)
ax.set_xticks(pos); ax.set_xticklabels(MODELS)
ax.set_ylabel("Test MSE (mean ± std)")
ax.set_title("Average generalization error")
plt.tight_layout()
plt.savefig(FIG / "test_mse_bar_mean_std.png", dpi=200)
plt.close(fig)

print(f"\nPart B done. Aggregated over {N_RUNS} runs.")
print(f"Figures -> {FIG}")
print(f"Tables -> {TAB}")
print("Note: run Part A first and load the actual best (n, degree) from CSV when available.")