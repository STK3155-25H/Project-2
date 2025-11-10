from __future__ import annotations
import os
from ..config import OUTPUT_DIR, MODELS_DIR
BASE_DIR = MODELS_DIR
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "complexity_analysis")

import json
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostCrossEntropy
from src.activation_functions import LRELU, RELU, softmax
from ..config import MODELS_DIR, MNIST_COMPLEXITY_OUTPUT_DIR
# -------------------- ACT FUNCS MAP --------------------
act_func_map = {
    'LRELU': LRELU,
    'RELU': RELU
}

# -------------------- UTILITIES --------------------
def build_layout_mnist(n_hidden: int, width: int):
    """Input 784, output 10 classi"""
    if n_hidden <= 0:
        return [784, 10]
    return [784] + [width]*n_hidden + [10]

def extract_val_metrics(history: dict, net: FFNN, X_val, y_val, mode="avg_last_n", last_n=10):
    """Return validation loss and accuracy"""
    y_pred = net.predict(X_val)
    val_loss = float(CostCrossEntropy(y_val)(y_pred))
    pred_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y_val, axis=1)
    val_acc = np.mean(pred_labels == true_labels)

    # Fallback su history
    val_losses_hist = history.get("val_loss", history.get("val_errors"))
    if val_losses_hist is not None and len(val_losses_hist) > 0:
        if mode == "min":
            val_loss = float(np.nanmin(val_losses_hist))
        elif mode == "final":
            val_loss = float(val_losses_hist[-1])
        elif mode == "avg_last_n":
            val_loss = float(np.mean(val_losses_hist[-last_n:]))
    return val_loss, val_acc

# -------------------- PATHS --------------------
BASE_DIR = MODELS_DIR
OUTPUT_DIR = MNIST_COMPLEXITY_OUTPUT_DIR
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def start_new_run(base_dir=BASE_DIR, output_dir=OUTPUT_DIR):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"run_{current_time}"
    os.makedirs(os.path.join(base_dir, run_dir), exist_ok=True)
    os.makedirs(os.path.join(output_dir, run_dir), exist_ok=True)
    return run_dir

# -------------------- PARAMETERS --------------------
SEED = 314
np.random.seed(SEED)
epochs = 50
lr = 0.001
lam_l1 = 0.0
lam_l2 = 0.0
rho = 0.9
rho2 = 0.999
batches = 128

activation_funcs = [RELU, LRELU]
n_hidden_list = [1,2,3,4,5]
n_perceptrons_list = [32,64,128,256,512]
VAL_LOSS_MODE = "avg_last_n"
LAST_N = 10

# -------------------- LOAD MNIST --------------------
def load_mnist():
    print("🔄 Fetching MNIST from OpenML (cached automatically)...")
    X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
    X = X.astype(np.float64) / 255.0
    y = y.astype(np.int64)
    y_onehot = np.eye(10)[y]
    return train_test_split(
        X, y_onehot, test_size=0.2, random_state=SEED, stratify=y
    )

X_train, X_val, y_train, y_val = load_mnist()

# -------------------- START RUN --------------------
run_dir = start_new_run(BASE_DIR, OUTPUT_DIR)

# Salva config
config = {
    'SEED': SEED,
    'epochs': epochs,
    'lr': lr,
    'lam_l1': lam_l1,
    'lam_l2': lam_l2,
    'rho': rho,
    'rho2': rho2,
    'batches': batches,
    'activation_funcs': [f.__name__ for f in activation_funcs],
    'n_hidden_list': n_hidden_list,
    'n_perceptrons_list': n_perceptrons_list,
    'VAL_LOSS_MODE': VAL_LOSS_MODE,
    'LAST_N': LAST_N
}
with open(os.path.join(BASE_DIR, run_dir, "config.json"), 'w') as f:
    json.dump(config, f, indent=4)

# -------------------- LOOP ACTIVATIONS --------------------
for act in activation_funcs:
    heat_loss = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)
    heat_acc = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan)
    for i_h, n_hidden in enumerate(n_hidden_list):
        for j_w, width in enumerate(n_perceptrons_list):
            layout = build_layout_mnist(n_hidden, width)
            model_filename = f"model_hidden_{n_hidden}_width_{width}_act_{act.__name__}.npz"
            model_path = os.path.join(BASE_DIR, run_dir, model_filename)
            done_marker = model_path + ".done"

            if os.path.exists(done_marker):
                print(f"{model_filename} already done. Skipping.")
                continue

            net = FFNN(
                dimensions=layout,
                hidden_func=act,
                output_func=softmax,
                cost_func=CostCrossEntropy,
                seed=SEED
            )
            scheduler = Adam(lr, rho, rho2)
            print(f"Training {model_filename} ...")

            history = net.fit(
                X=X_train, t=y_train,
                scheduler=scheduler,
                batches=batches,
                epochs=epochs,
                lam_l1=lam_l1,
                lam_l2=lam_l2,
                X_val=X_val, t_val=y_val
            )

            net.save_weights(model_path)
            with open(done_marker, "w") as f:
                f.write("ok")

            val_loss, val_acc = extract_val_metrics(history, net, X_val, y_val,
                                                    mode=VAL_LOSS_MODE, last_n=LAST_N)
            heat_loss[i_h,j_w] = val_loss
            heat_acc[i_h,j_w] = val_acc
    # Salvataggio CSV
    df_loss = pd.DataFrame(heat_loss, index=n_hidden_list, columns=n_perceptrons_list)
    df_loss.index.name = 'hidden_layers'
    df_loss.columns.name = 'neurons_per_layer'
    df_loss.to_csv(os.path.join(OUTPUT_DIR, run_dir, f"val_loss_{act.__name__}.csv"))

    df_acc = pd.DataFrame(heat_acc, index=n_hidden_list, columns=n_perceptrons_list)
    df_acc.index.name = 'hidden_layers'
    df_acc.columns.name = 'neurons_per_layer'
    df_acc.to_csv(os.path.join(OUTPUT_DIR, run_dir, f"val_acc_{act.__name__}.csv"))

    # Plot loss heatmap
    plt.figure(figsize=(8,5))
    im = plt.imshow(heat_loss, aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, label="Validation Loss")
    plt.title(f"Validation Loss — activation: {act.__name__}")
    plt.xlabel("Neurons per hidden layer")
    plt.ylabel("Number of hidden layers")
    plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list)
    plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, run_dir, f"heatmap_loss_{act.__name__}.png"))
    plt.close()

    # Plot accuracy heatmap
    plt.figure(figsize=(8,5))
    im = plt.imshow(heat_acc, aspect='auto', origin='upper', interpolation='nearest')
    plt.colorbar(im, label="Validation Accuracy")
    plt.title(f"Validation Accuracy — activation: {act.__name__}")
    plt.xlabel("Neurons per hidden layer")
    plt.ylabel("Number of hidden layers")
    plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list)
    plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, run_dir, f"heatmap_acc_{act.__name__}.png"))
    plt.close()

print("MNIST complexity analysis complete.")
