#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_data(csv_path: str):
    """Legge un CSV e restituisce y_labels, x_labels e matrice dei dati ordinata."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Etichette
    y_labels = df.iloc[:, 0].tolist()
    x_raw = list(df.columns[1:])
    try:
        x_labels = [int(float(x)) for x in x_raw]
    except ValueError:
        x_labels = x_raw

    order_idx = np.argsort(x_labels) if all(isinstance(x, int) for x in x_labels) else np.arange(len(x_labels))
    x_labels_sorted = [x_labels[i] for i in order_idx]

    data = df.iloc[:, 1:].to_numpy(dtype=float)
    data = data[:, order_idx]  # ordina colonne

    return np.array(y_labels), np.array(x_labels_sorted), data


def plot_comparison(csv_relu, csv_lrelu, title, out_path=None, cmap="viridis"):
    # ---- Caricamento dati ----
    y_relu, x_relu, data_relu = load_data(csv_relu)
    y_lrelu, x_lrelu, data_lrelu = load_data(csv_lrelu)

    # Verifica compatibilità
    if not np.array_equal(y_relu, y_lrelu) or not np.array_equal(x_relu, x_lrelu):
        raise ValueError("Le dimensioni o le etichette dei due CSV non coincidono!")

    # ---- Limiti di scala comuni ----
    vmin = min(data_relu.min(), data_lrelu.min())
    vmax = max(data_relu.max(), data_lrelu.max())

    # ---- Figura e assi ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=False)

    # Heatmap ReLU
    im1 = axes[0].imshow(data_relu, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    axes[0].set_title("ReLU", fontsize=12)
    axes[0].set_xlabel("Units per hidden layer")
    axes[0].set_ylabel("Number of hidden layers")

    # Heatmap Leaky ReLU
    im2 = axes[1].imshow(data_lrelu, aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    axes[1].set_title("Leaky ReLU", fontsize=12)
    axes[1].set_xlabel("Units per hidden layer")

    # Imposta tick e label
    for ax in axes:
        ax.set_xticks(np.arange(len(x_relu)))
        ax.set_xticklabels(x_relu, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(y_relu)))
        ax.set_yticklabels(y_relu)

    # ---- Colorbar unica a destra ----
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im2, cax=cax)
    cbar.set_label("Value")

    # Titolo globale
    fig.suptitle(title, fontsize=14)

    # Spaziatura tra i due pannelli
    plt.subplots_adjust(wspace=0.3, top=0.85)

    # ---- Salvataggio ----
    if out_path is None:
        base = os.path.splitext(os.path.basename(csv_relu))[0].replace("RELU", "")
        out_path = f"{base}_comparison_horizontal.pdf"

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato in: {out_path}")
    plt.close()


if __name__ == "__main__":
    # Esempio d’uso:
    # ./compare_heatmaps.py relu_loss.csv lrelu_loss.csv "Validation Loss"
    if len(sys.argv) < 4:
        print("Uso: compare_heatmaps.py <csv_relu> <csv_lrelu> <title>")
        sys.exit(1)

    csv_relu = sys.argv[1]
    csv_lrelu = sys.argv[2]
    title = sys.argv[3]

    plot_comparison(csv_relu, csv_lrelu, title)
