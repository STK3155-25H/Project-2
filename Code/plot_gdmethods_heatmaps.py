#==========================================================
# Single figure with 2 heatmaps, shared colorbar
# Y-axis ticks: scientific notation 2 digits
# Cells NOT square (aspect='auto')
#==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import numpy as np

def plot_complexity_heatmaps_shared_colorbar(csv_files, titles=None, output_dir="plots", base_name="heatmap"):
    if len(csv_files) != 2:
        raise ValueError("csv_files must be a list of 2 CSV files.")
    
    if titles is None:
        titles = [f"Heatmap {i+1}" for i in range(2)]

    os.makedirs(output_dir, exist_ok=True)

    # Load and transpose
    dfs = [pd.read_csv(f, index_col=0).T for f in csv_files]

    fig = plt.figure(figsize=(7.5, 6), constrained_layout=True)
    gs = GridSpec(1, 3, width_ratios=[1, 1, 0.05], figure=fig)

    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    cax = fig.add_subplot(gs[0, 2])

    # Global vmin/vmax for shared color scale
    all_values = np.concatenate([df.values.flatten() for df in dfs])
    vmin, vmax = 0, 1

    # formatter for y-axis labels → scientific notation, 2 significant digits
    def sci_fmt(v):
        try:
            return f"{float(v):.2e}"
        except:
            return str(v)

    for df, ax, title in zip(dfs, axes, titles):
        # ✅ Cells not square now
        im = ax.imshow(df.values, aspect='auto', origin='upper', vmin=vmin, vmax=vmax)

        ax.set_title(title)
        ax.set_ylabel("Learning Rate")

        ax.set_yticks(range(len(df.index)))
        ax.set_yticklabels([sci_fmt(v) for v in df.index])

        # Remove 2nd x axis labels
        if ax is axes[1]:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        ax.set_xlabel("Optimizer")
        # Split only the long label into 2 lines
        xticks = []
        for v in df.columns:
            if v.lower() == "adagradmomentum":
                xticks.append("adagrad\nmomentum")
            else:
                xticks.append(v)
        ax.set_xticks(range(len(df.columns)))
        ax.set_xticklabels(xticks, rotation=45)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("Validation Loss", rotation=90, labelpad=8)

    output_file = os.path.join(output_dir, f"{base_name}_shared_colorbar.png")
    fig.savefig(output_file, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_file}")


# Example usage
if __name__ == "__main__":
    csv_files = [
        "output/OptimizerSweep/20251108_122801/heatmap_pivot_val_full.csv",
        "output/OptimizerSweep/20251108_122801/heatmap_pivot_val_sgd.csv"
    ]

    titles = [
        "Gradient Descent",
        "Stochastic Gradient Descent",
    ]

    plot_complexity_heatmaps_shared_colorbar(
        csv_files,
        titles,
        output_dir="output/plots",
        base_name="GDLR_Heatmap"
    )
