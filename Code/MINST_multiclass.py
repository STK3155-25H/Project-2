"""
mnist_trainer.py
MNIST multiclass FFNN + resume + checkpoint + scientific plots
"""

from __future__ import annotations
import os
import json
from pathlib import Path
import time
from typing import Dict, Tuple

import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder

# === your modules ===
from src.FFNN import FFNN
from src.scheduler import Adam
from src.activation_functions import RELU, softmax
from src.cost_functions import CostCrossEntropy


# ============================================================
# Utility helpers
# ============================================================
def one_hot(y, n_classes):
    enc = OneHotEncoder(sparse_output=False, categories=[np.arange(n_classes)])
    return enc.fit_transform(y.reshape(-1, 1))


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def save_state(cache, state_dict):
    ensure_dir(cache)
    with open(cache / "state.json", "w") as f:
        json.dump(state_dict, f, indent=2)


def load_state(cache):
    state_path = cache / "state.json"
    if not state_path.exists():
        return None
    with open(state_path, "r") as f:
        return json.load(f)


# ============================================================
# Data loading
# ============================================================
def load_mnist():
    print("Fetching MNIST from OpenML (cached automatically)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)

    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int64)

    return train_test_split(
        X, y, test_size=0.2, random_state=123, stratify=y
    )


# ============================================================
# Model building
# ============================================================
def build_model(seed=None):
    model = FFNN(
        dimensions=(784, 256, 128, 10),
        hidden_func=RELU,
        output_func=softmax,
        cost_func=CostCrossEntropy,
        seed=seed,
    )
    return model


def accuracy_argmax(model, X, y_true):
    probs = model.predict_proba(X)
    y_pred = np.argmax(probs, axis=1)
    return float(np.mean(y_pred == y_true)), y_pred


# ============================================================
# Plotting section (scientific visuals)
# ============================================================
def plot_learning_curves(history, outdir: Path):

    plt.figure(figsize=(10, 6))
    plt.plot(history["train_errors"], label="Train loss", lw=2)
    plt.plot(history["val_errors"], label="Val loss", lw=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Cross Entropy)")
    plt.title("Learning Curve — Loss")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(outdir / "learning_curves.png", dpi=220)
    plt.close()


def plot_confusion(y_true, y_pred, outdir: Path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 7))
    plt.imshow(cm, cmap="viridis")
    plt.title("Confusion Matrix — MNIST")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.colorbar()
    plt.savefig(outdir / "confusion_matrix.png", dpi=220)
    plt.close()

def plot_misclassified(X, y_true, y_pred, outdir: Path, max_show=25):
    import math
    wrong_idx = np.where(y_pred != y_true)[0]
    if wrong_idx.size == 0:
        print("No picture was misclassified!")
        return

    # select maximum max_show examples
    n = int(min(max_show, wrong_idx.size))
    idx = wrong_idx[:n]

    # grid
    cols = 5
    rows = math.ceil(n / cols)

    # figsize proportional to give space to titles
    fig_w = cols * 2.2
    fig_h = rows * 2.8
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), constrained_layout=False)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = axes.ravel()

    for i in range(rows * cols):
        ax = axes[i]
        if i < n:
            img = X[idx[i]].reshape(28, 28)
            ax.imshow(img, cmap="gray", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            # Title with padding + white margin to avoid overlapping with subplot under
            ax.set_title(
                f"T:{y_true[idx[i]]}  P:{y_pred[idx[i]]}",
                fontsize=9,
                pad=8,  # extra space over the picture
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=2),
            )
        else:
            ax.axis("off")

    fig.suptitle("Esempi Misclassificati — FFNN MNIST", fontsize=12, y=0.995)
    # Add vertical space between subplots to avoid titles overlap
    fig.subplots_adjust(top=0.92, hspace=0.6, wspace=0.15)

    outpath = outdir / "misclassified_examples.png"
    fig.savefig(outpath, dpi=220)
    plt.close(fig)



# ============================================================
# Training routine (with resume)
# ============================================================
def train_or_resume(
    cache_dir=".mnist_cache",
    epochs=50,
    batches=128,
    eta=1e-3,
):

    cache = Path(cache_dir)
    weights_file = cache / "weights.npz"
    ensure_dir(cache)

    X_train, X_test, y_train_i, y_test_i = load_mnist()
    t_train = one_hot(y_train_i, 10)
    t_test = one_hot(y_test_i, 10)

    model = build_model(seed=123)

    state = load_state(cache)
    start_epoch = state["epoch_completed"] if state else 0

    # Resume if cache exists
    if weights_file.exists():
        print("Resuming from checkpoint…")
        model.load_weights(str(weights_file))

    scheduler = Adam(eta=eta, rho=0.9, rho2=0.999)

    history = {"train_errors": [], "val_errors": []}

    try:
        for epoch in range(start_epoch, epochs):
            print(f"\n Epoch {epoch+1}/{epochs}")

            scores = model.fit(
                X_train, t_train,
                scheduler=scheduler,
                batches=batches, epochs=1,
                X_val=X_test, t_val=t_test,
                save_on_interrupt=str(weights_file),
            )

            # update history
            history["train_errors"].append(float(scores["train_errors"][-1]))
            history["val_errors"].append(float(scores["val_errors"][-1]))

            # save checkpoint
            model.save_weights(str(weights_file))
            save_state(cache, {"epoch_completed": epoch + 1})

    except KeyboardInterrupt:
        print("\n Training interrupted — checkpoint saved.")
        return

    print("\nTraining finished!")

    # ============================================================
    # Final evaluation + plots
    # ============================================================
    print("\nEvaluating model…")
    test_acc, y_pred = accuracy_argmax(model, X_test, y_test_i)
    print(f"Test accuracy (argmax): {test_acc:.4f}")

    results_dir = Path("output/MINST/mnist_ffnn_results")
    ensure_dir(results_dir)

    plot_learning_curves(history, results_dir)
    plot_confusion(y_test_i, y_pred, results_dir)
    plot_misclassified(X_test, y_test_i, y_pred, results_dir)

    print("\nFigures saved in: ./mnist_ffnn_results/")
    print("   - learning_curves.png")
    print("   - confusion_matrix.png")
    print("   - misclassified_examples.png")


# ============================================================
# Script entry point
# ============================================================
if __name__ == "__main__":
    train_or_resume()
