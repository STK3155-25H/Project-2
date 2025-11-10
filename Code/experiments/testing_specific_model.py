import os
from ..config import OUTPUT_DIR, MODELS_DIR
BASE_DIR = MODELS_DIR
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "complexity_analysis")
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
from src.FFNN import FFNN
from src.cost_functions import CostOLS
from src.activation_functions import identity, LRELU, RELU, sigmoid
from ..config import SPECIFIC_MODEL_EVAL_OUTPUT_DIR
# -----------------------------
# Reproducibility
# -----------------------------
SEED = 314
np.random.seed(SEED)
# -----------------------------
# Data generators
# -----------------------------
def runge_noisy(x: np.ndarray, noise_std: float = 0.05) -> np.ndarray:
    """Runge function with optional Gaussian noise."""
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1.0 / (1.0 + 25.0 * x**2) + noise
def runge_true(x: np.ndarray) -> np.ndarray:
    """Noise-free Runge function."""
    return 1.0 / (1.0 + 25.0 * x**2)
# Reference noisy sample (for background in plots)
X_REF = np.linspace(-1, 1, 200).reshape(-1, 1)
Y_REF_NOISY = runge_noisy(X_REF, noise_std=0.03)
# -----------------------------
# Activation helpers
# -----------------------------
def normalize_act_name(name: str) -> str:
    """Canonicalize an activation name to lowercase alphanumerics."""
    if name is None:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", name).lower()
def get_activation_function(act_name: str):
    """Return activation callable from a name (case-insensitive, with aliases)."""
    key = normalize_act_name(act_name)
    mapping = {
        "relu": RELU,
        "lrelu": LRELU,
        "leakyrelu": LRELU,
        "leaky": LRELU,
        "sigmoid": sigmoid,
        "identity": identity, # supported if you ever train with it
    }
    func = mapping.get(key)
    if func is None:
        raise ValueError(f"Unknown activation function: {act_name!r}")
    return func
def canonical_file_token(act_name: str) -> str:
    """Token as it appears in filenames for a given activation."""
    key = normalize_act_name(act_name)
    token_map = {
        "relu": "RELU",
        "lrelu": "LRELU",
        "leakyrelu": "LRELU",
        "leaky": "LRELU",
        "sigmoid": "sigmoid",
        "identity": "identity",
    }
    return token_map.get(key, act_name)
# -----------------------------
# Filenames and folders
# -----------------------------
def model_filename(n_hidden: int, width: int, act_name: str) -> str:
    """Expected model filename for a layout + activation."""
    token = canonical_file_token(act_name)
    return f"model_hidden_{n_hidden}_width_{width}_act_{token}.npz"
def layout_dir_name(n_hidden: int, width: int) -> str:
    """Folder name for a specific layout."""
    return f"hidden{n_hidden}_width{width}"
def run_layout_output_dir(base_out: str, run_dir: str, n_hidden: int, width: int) -> str:
    """
    Output directory:
      base_out / <run_name> / hidden<N>_width<W> /
    """
    run_name = os.path.basename(os.path.normpath(run_dir))
    return os.path.join(base_out, run_name, layout_dir_name(n_hidden, width))
# -----------------------------
# Core evaluation
# -----------------------------
def evaluate_one_model(
    model_path: str,
    n_hidden: int,
    width: int,
    hidden_activation_name: str,
    save_plot: bool,
    layout_out_dir: str,
    save_curve_csv: bool = True,
):
    """
    Load one model, predict on [-1,1], compute MSE vs true Runge,
    and optionally save a per-activation plot & CSV to the layout-specific directory.
    Returns:
        mse (float), X_eval (ndarray shape [N,1]), y_pred (ndarray shape [N,1])
    Raises:
        FileNotFoundError if model_path doesn't exist.
    """
    hidden_act = get_activation_function(hidden_activation_name)
    dims = [1] + [width] * n_hidden + [1]
    net = FFNN(
        dimensions=tuple(dims),
        hidden_func=hidden_act,
        output_func=identity,
        cost_func=CostOLS,
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    net.load_weights(model_path)
    X_eval = np.linspace(-1, 1, 200).reshape(-1, 1)
    y_true = runge_true(X_eval)
    y_pred = net.predict(X_eval)
    mse = float(np.mean((y_pred - y_true) ** 2))
    # Per-activation plot
    plt.figure(figsize=(8, 6))
    plt.plot(X_REF, Y_REF_NOISY, label="Noisy sample", linewidth=1, alpha=0.7)
    plt.plot(X_eval, y_true, label="True Runge", linewidth=2)
    plt.plot(X_eval, y_pred, label="Model prediction", linestyle="--", linewidth=2)
    plt.title(f"{os.path.basename(model_path)}\nMSE: {mse:.6f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    os.makedirs(layout_out_dir, exist_ok=True)
    if save_plot:
        png_name = f"eval_{model_filename(n_hidden, width, hidden_activation_name)[:-4]}.png"
        out_png = os.path.join(layout_out_dir, png_name)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"[Saved] {out_png}")
    else:
        plt.show()
    plt.close()
    # === CSV SAVE (per-activation curve) ===
    if save_curve_csv:
        # Align noisy reference to the same grid
        x = X_eval.ravel()
        y_t = y_true.ravel()
        y_p = y_pred.ravel()
        y_ref = Y_REF_NOISY.ravel()
        csv_name = f"data_{model_filename(n_hidden, width, hidden_activation_name)[:-4]}.csv"
        out_csv = os.path.join(layout_out_dir, csv_name)
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y_true", "y_pred", "y_ref_noisy"])
            for xi, yt, yp, yr in zip(x, y_t, y_p, y_ref):
                writer.writerow([f"{xi:.10f}", f"{yt:.10f}", f"{yp:.10f}", f"{yr:.10f}"])
        print(f"[Saved] per-activation curve CSV -> {out_csv}")
    return mse, X_eval, y_pred
# -----------------------------
# Batch runner (run + layout)
# -----------------------------
def evaluate_all_activations_for_layout(
    run_dir: str,
    n_hidden: int,
    width: int,
    activations: list | None = None,
    save_plot: bool = True,
    base_plot_dir: str = SPECIFIC_MODEL_EVAL_OUTPUT_DIR,
    save_csv: bool = True,
    save_curve_csv: bool = True,
) -> dict:
    """
    Evaluate a set of activations for a given run folder and layout.
    Creates a run-specific + layout-specific output folder:
        base_plot_dir/<run_name>/hidden<N>_width<W>/
    Missing models are skipped without raising.
    Returns:
        results dict: activation -> {
            "mse": float,
            "model_path": str,
            "X_eval": ndarray,
            "y_pred": ndarray
        }
    """
    if activations is None:
        activations = ["RELU", "LRELU", "sigmoid"]
    run_dir = os.path.normpath(run_dir)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(f"Run directory not found: {run_dir}")
    layout_out_dir = run_layout_output_dir(base_plot_dir, run_dir, n_hidden, width)
    os.makedirs(layout_out_dir, exist_ok=True)
    print(f"== Evaluating run='{run_dir}', layout=hidden{n_hidden}, width={width} ==")
    print(f" Output -> {layout_out_dir}")
    results = {}
    for act in activations:
        fname = model_filename(n_hidden, width, act)
        model_path = os.path.join(run_dir, fname)
        try:
            mse, X_eval, y_pred = evaluate_one_model(
                model_path=model_path,
                n_hidden=n_hidden,
                width=width,
                hidden_activation_name=act,
                save_plot=save_plot,
                layout_out_dir=layout_out_dir,
                save_curve_csv=save_curve_csv,
            )
            results[act] = {
                "mse": mse,
                "model_path": model_path,
                "X_eval": X_eval,
                "y_pred": y_pred,
            }
            print(f"[OK] {act:<8} MSE={mse:.6e}")
        except FileNotFoundError:
            print(f"[Skip] {act:<8} (file not found)")
        except ValueError as e:
            print(f"[Skip] {act:<8} ({e})")
        except Exception as e:
            print(f"[Skip] {act:<8} (error: {type(e).__name__}: {e})")
    # Write CSV summary (meta)
    if save_csv and results:
        csv_path = os.path.join(layout_out_dir, "summary.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["activation", "mse", "model_path"])
            for act, info in sorted(results.items(), key=lambda kv: kv[1]["mse"]):
                writer.writerow([act, f"{info['mse']:.8f}", info["model_path"]])
        print(f"[Saved] summary CSV -> {csv_path}")
    # Unified overlay plot with all available predictions
    if results:
        # Use X from any entry (all identical)
        any_key = next(iter(results.keys()))
        X_overlay = results[any_key]["X_eval"]
        y_true = runge_true(X_overlay)
        plt.figure(figsize=(10, 7))
        # Noisy and clean references
        plt.plot(X_REF, Y_REF_NOISY, label="Noisy sample", linewidth=1, alpha=0.7)
        plt.plot(X_overlay, y_true, label="True Runge", linewidth=2)
        # Plot each activation's prediction; sort legend by MSE
        results_sorted = sorted(results.items(), key=lambda kv: kv[1]["mse"])
        for act, info in results_sorted:
            y_pred = info["y_pred"]
            mse = info["mse"]
            plt.plot(
                X_overlay,
                y_pred,
                linestyle="--",
                linewidth=2,
                label=f"{act} (MSE {mse:.3e})",
            )
        plt.title(
            f"Overlay predictions — run '{os.path.basename(os.path.normpath(run_dir))}'\n"
            f"layout: hidden{n_hidden}, width={width}"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True)
        overlay_png = os.path.join(layout_out_dir, "overlay_predictions.png")
        overlay_pdf = os.path.join(layout_out_dir, "overlay_predictions.pdf")
        plt.savefig(overlay_png, dpi=150, bbox_inches="tight")
        plt.savefig(overlay_pdf, bbox_inches="tight")
        plt.close()
        print(f"[Saved] overlay overlay plot -> {overlay_png}")
        print(f"[Saved] overlay plot -> {overlay_pdf}")
        # === CSV SAVE (overlay, long format) ===
        if save_curve_csv:
            overlay_csv = os.path.join(layout_out_dir, "overlay_data.csv")
            with open(overlay_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "x", "y_true", "y_pred", "y_ref_noisy",
                    "activation", "mse", "model_path"
                ])
                x = X_overlay.ravel()
                y_t = y_true.ravel()
                y_ref = Y_REF_NOISY.ravel()
                for act, info in results_sorted:
                    y_p = info["y_pred"].ravel()
                    mse = info["mse"]
                    mpath = info["model_path"]
                    for xi, yt, yp, yr in zip(x, y_t, y_p, y_ref):
                        writer.writerow([
                            f"{xi:.10f}", f"{yt:.10f}", f"{yp:.10f}", f"{yr:.10f}",
                            act, f"{mse:.10f}", mpath
                        ])
            print(f"[Saved] overlay data CSV -> {overlay_csv}")
            # === CSV SAVE (overlay, wide format) ===
            # One row per x; columns: x, y_true, y_ref_noisy, y_pred_<ACT...> (ordered by MSE)
            overlay_wide_csv = os.path.join(layout_out_dir, "overlay_wide.csv")
            x = X_overlay.ravel()
            y_t = y_true.ravel()
            y_ref = Y_REF_NOISY.ravel()
            # Collect y_pred columns in MSE order
            act_names = [act for act, _ in results_sorted]
            ypred_cols = [info["y_pred"].ravel() for _, info in results_sorted]
            header = ["x", "y_true", "y_ref_noisy"] + [f"y_pred_{act}" for act in act_names]
            with open(overlay_wide_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                for i in range(len(x)):
                    row = [f"{x[i]:.10f}", f"{y_t[i]:.10f}", f"{y_ref[i]:.10f}"]
                    row += [f"{yp[i]:.10f}" for yp in ypred_cols]
                    writer.writerow(row)
            print(f"[Saved] overlay wide CSV -> {overlay_wide_csv}")
    else:
        print("[Info] No models were evaluated (overlay not created).")
    return results
# -----------------------------
# CLI example
# -----------------------------
if __name__ == "__main__":
    # Choose the specific run folder and desired layout
    RUN_DIR = "Models/run_20251105_110618"
    N_HIDDEN = 3
    WIDTH = 30
    # None -> tries default set; or pass your own list:
    # ACTIVATIONS = ["relu", "lrelu", "sigmoid", "identity"]
    ACTIVATIONS = None
    evaluate_all_activations_for_layout(
        run_dir=RUN_DIR,
        n_hidden=N_HIDDEN,
        width=WIDTH,
        activations=ACTIVATIONS,
        save_plot=True,
        base_plot_dir=SPECIFIC_MODEL_EVAL_OUTPUT_DIR,
        save_csv=True,
        save_curve_csv=True,
    )