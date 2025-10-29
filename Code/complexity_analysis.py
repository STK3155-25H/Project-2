import numpy as np
import os
import pandas as pd
SEED = os.environ.get("SEED")

if SEED is not None:
    SEED = int(SEED)
    print("SEED from env:", SEED)
else:
    SEED = 314
    print("SEED from hard-coded value in file ml_core.py :", SEED)
    print("If you want a specific SEED set the SEED environment variable")
np.random.seed(SEED)

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.FFNN import FFNN
from src.scheduler import Adam
from src.cost_functions import CostOLS
from src.activation_functions import sigmoid, identity, LRELU, RELU, tanh, softmax

def runge(x, noise_std=0.05):
    noise = np.random.normal(0, noise_std, size=x.shape)
    return 1 / (1 + 25 * x**2) + noise


X = np.linspace(-1, 1, 200).reshape(-1, 1)
y = runge(X).reshape(-1, 1)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, shuffle=True
)

# ---- Training Settings ---- #
epochs = 1500
lr = 0.001
lam1 = 0.0
lam2 = 0.0
rho = 0.9
rho2 = 0.999
batches = 100

# activation_funcs = [sigmoid, LRELU, RELU, tanh, softmax]
activation_funcs = [LRELU, RELU, tanh]

n_hidden_list = list(range(1, 6))
n_perceptrons_list = [2*i for i in range(1, 21)]

VAL_LOSS_MODE = "min"  # "final" for the loss of the last epoch

def build_layout(n_hidden: int, width: int):
    """
    Crea un layout come [1, width, width, ..., 1]
    Se n_hidden = 0 => [1, 1] (modello lineare)
    """
    if n_hidden <= 0:
        return [1, 1]
    return [1] + [width] * n_hidden + [1]

def extract_val_loss(history: dict):
    """
    Estrae la validation loss dalla history restituita da FFNN.fit().
    Prova diverse chiavi; se assente, calcola manualmente la MSE su X_val.
    """
    # convenzioni possibili nelle tue versioni precedenti
    val = history.get("val_loss", history.get("val_errors"))
    if val is not None and len(val) > 0:
        if VAL_LOSS_MODE == "min":
            return float(np.min(val))
        else:
            return float(val[-1])

    try:
        y_pred = net.predict(X_val) 
        return float(CostOLS(y_pred, y_val))
    except Exception:
        return np.nan

# Create folders if they don't exist
os.makedirs("Models", exist_ok=True)
os.makedirs("output", exist_ok=True)

for act in activation_funcs:
    heat = np.full((len(n_hidden_list), len(n_perceptrons_list)), np.nan, dtype=float)

    for i_h, n_hidden in enumerate(n_hidden_list):
        for j_w, width in enumerate(n_perceptrons_list):
            layout = build_layout(n_hidden, width)

            # Inizializza rete
            net = FFNN(
                dimensions=layout,
                hidden_func=act,
                output_func=identity,
                cost_func=CostOLS,
                seed=SEED,
            )
            scheduler = Adam(lr, rho, rho2)

            # Allena (con validation)
            history = net.fit(
                X=X_train, t=y_train,
                scheduler=scheduler,
                batches=batches,
                epochs=epochs,
                lam_l1=lam1,
                lam_l2=lam2,
                X_val=X_val, t_val=y_val,
            )

            val_loss = None
            val_loss = history.get("val_loss", history.get("val_errors"))
            if val_loss is not None and len(val_loss) > 0:
                val_loss = float(np.min(val_loss) if VAL_LOSS_MODE == "min" else val_loss[-1])
            else:
                try:
                    y_pred = net.predict(X_val)
                    val_loss = float(CostOLS(y_pred, y_val))
                except Exception:
                    val_loss = np.nan

            heat[i_h, j_w] = val_loss

            # Save the model with unique filename including n_hidden and width
            model_filename = f"model_hidden_{n_hidden}_width_{width}_act_{act.__name__}.npz"
            net.save_weights(os.path.join("Models", model_filename))

    # Save the heatmap data to CSV
    df = pd.DataFrame(heat, index=n_hidden_list, columns=n_perceptrons_list)
    df.index.name = 'hidden_layers'
    df.columns.name = 'neurons_per_layer'
    csv_filename = f"val_loss_data_{act.__name__}.csv"
    df.to_csv(os.path.join("output", csv_filename))

    plt.figure(figsize=(10, 5))
    im = plt.imshow(
        heat,
        aspect="auto",
        origin="upper",
        interpolation="nearest"
    )
    plt.colorbar(im, label="Validation Loss (OLS)")
    plt.title(f"Validation Loss Heatmap — activation: {act.__name__}")
    plt.xlabel("Neurons per hidden layer")
    plt.ylabel("Number of hidden layers")

    plt.xticks(ticks=np.arange(len(n_perceptrons_list)), labels=n_perceptrons_list, rotation=45)
    plt.yticks(ticks=np.arange(len(n_hidden_list)), labels=n_hidden_list)

    # Mostra i valori (opzionale: commenta se disturba)
    # for (i, j), v in np.ndenumerate(heat):
    #     if np.isfinite(v):
    #         plt.text(j, i, f"{v:.2e}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plot_filename = f"val_loss_heatmap_{act.__name__}.png"
    plt.savefig(os.path.join("output", plot_filename))
    plt.show()