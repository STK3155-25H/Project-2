#!/usr/bin/env python3
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path: str = "data.csv", out_path: str | None = None):
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"File not found: {csv_path}")

    # ---- Data loading ----
    df = pd.read_csv(csv_path)

    # First column = number of hidden layers (labels Y axes)
    y_labels = df.iloc[:, 0].tolist()

    # Header from second column = unit per layer (labels X axes)
    # Try to convert to numbers to sort correctly
    x_raw = list(df.columns[1:])
    try:
        x_labels = [int(float(x)) for x in x_raw]
    except ValueError:
        # If not numeric, leave strings
        x_labels = x_raw

    # Sort numeric columns (if possible) to have ascending X 
    order_idx = np.argsort(x_labels) if all(isinstance(x, int) for x in x_labels) else np.arange(len(x_labels))
    x_labels_sorted = [x_labels[i] for i in order_idx]

    data = df.iloc[:, 1:].to_numpy(dtype=float)
    data = data[:, order_idx]  # sorts columns

    # ---- Plot heatmap ----
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(data, aspect="auto", vmin= 0.00000000000000001, vmax=0.005)  # uses default colormap 

    # Tick & label
    ax.set_xticks(np.arange(len(x_labels_sorted)))
    ax.set_xticklabels(x_labels_sorted, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    ax.set_xlabel("Units per hidden layer")
    ax.set_ylabel("Number of hidden layers")
    ax.set_title("Heatmap (es. validation loss per layout)")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    fig.tight_layout()

    # Saving
    if out_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_path = f"{base}_heatmap.png"

    plt.savefig(out_path, dpi=200)
    print(f"Heatmap saved in: {out_path}")

if __name__ == "__main__":
    csv = sys.argv[1] if len(sys.argv) > 1 else "output/run_20251105_110618/val_loss_data_clean_RELU.csv"
    main(csv)
