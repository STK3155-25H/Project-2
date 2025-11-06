#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str = "data.csv", out_path: str | None = None):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")

    # ---- Caricamento dati ----
    df = pd.read_csv(csv_path)

    # Prima colonna = numero di hidden layers (etichette assi Y)
    y_labels = df.iloc[:, 0].tolist()

    # Header dalla seconda colonna in poi = unità per layer (etichette asse X)
    # Provo a convertirle in numeri per ordinare correttamente
    x_raw = list(df.columns[1:])
    try:
        x_labels = [int(float(x)) for x in x_raw]
    except ValueError:
        # Se non sono numeriche, le lascio come stringhe
        x_labels = x_raw

    # Ordino le colonne numeriche (se possibile) per avere X crescente
    order_idx = np.argsort(x_labels) if all(isinstance(x, int) for x in x_labels) else np.arange(len(x_labels))
    x_labels_sorted = [x_labels[i] for i in order_idx]

    data = df.iloc[:, 1:].to_numpy(dtype=float)
    data = data[:, order_idx]  # riordina le colonne

    # ---- Plot heatmap ----
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", vmin= 0.00000000000000001, vmax=0.005)  # usa la colormap di default

    # Tick & label
    ax.set_xticks(np.arange(len(x_labels_sorted)))
    ax.set_xticklabels(x_labels_sorted, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Unità per hidden layer")
    ax.set_ylabel("Numero di hidden layers")
    ax.set_title("Heatmap (es. validation loss per layout)")

    # Barra colori
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Valore")

    fig.tight_layout()

    # Salvataggio
    if out_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = f"{base}_heatmap.png"

    plt.savefig(out_path, dpi=200)
    print(f"Heatmap salvata in: {out_path}")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "output/run_20251105_110618/val_loss_data_clean_RELU.csv"
    main(csv)
