#==========================================================
# File to create heatmaps for the MNIST complexity experiment
# 4 CSVs → 2 figures → each heatmap has its own colorbar
#==========================================================
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec
import numpy as np

#----------------------------------------------------------
# Plotting function
#----------------------------------------------------------
def plot_complexity_heatmaps_4(csv_files, titles=None, output_dir="plots", base_name="heatmap"):
    if len(csv_files) != 4:
        raise ValueError("csv_files must be a list of 4 CSV files.")
    
    if titles is None:
        titles = [f"Heatmap {i+1}" for i in range(4)]

    os.makedirs(output_dir, exist_ok=True)

    # Load and transpose
    dfs = [pd.read_csv(f, index_col=0).T for f in csv_files]

    #------------------------------------------------------
    # Internal helper: each heatmap gets its own colorbar
    #------------------------------------------------------
    def _plot_pair(dfs_pair, titles_pair, output_file):
        fig = plt.figure(figsize=(7.5, 3), constrained_layout=True)

        # 2 heatmaps → 2 colorbars
        gs = GridSpec(1, 4, width_ratios=[1, 0.1, 1, 0.1], figure=fig)

        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
        caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 3])]

        for df, ax, cax, title in zip(dfs_pair, axes, caxes, titles_pair):

            # Unique min/max per heatmap
            values = df.values.flatten()
            vmin, vmax = values.min(), values.max()

            im = ax.imshow(df.values, aspect='equal', origin='upper',
                           vmin=vmin, vmax=vmax)

            ax.set_title(title)
            ax.set_ylabel("Nodes per layer")
            ax.set_yticks(range(len(df.index)))
            ax.set_yticklabels(df.index)

            ax.set_xlabel("Layers")
            ax.set_xticks(range(len(df.columns)))
            ax.set_xticklabels(df.columns, rotation=45)

            # ✅ Individual colorbar
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Metric Value", rotation=90, labelpad=8)

        fig.savefig(output_file, dpi=300)
        plt.close(fig)
        print(f"Saved: {output_file}")

    #------------------------------------------------------
    # Figure 1 & Figure 2
    #------------------------------------------------------
    _plot_pair(dfs[:2], titles[:2], os.path.join(output_dir, f"{base_name}_fig1.png"))
    _plot_pair(dfs[2:], titles[2:], os.path.join(output_dir, f"{base_name}_fig2.png"))


#----------------------------------------------------------
# Example usage
#----------------------------------------------------------
if __name__ == "__main__":
    csv_files = [
        "output/MNIST_ComplexityAnalysis/run_20251107_220350/val_acc_LRELU.csv",
        "output/MNIST_ComplexityAnalysis/run_20251107_220350/val_loss_LRELU.csv",
        "output/MNIST_ComplexityAnalysis/run_20251107_220350/val_acc_RELU.csv",
        "output/MNIST_ComplexityAnalysis/run_20251107_220350/val_loss_RELU.csv"
    ]

    titles = [
        "Accuracy — LReLU",
        "Loss — LReLU",
        "Accuracy — ReLU",
        "Loss — ReLU"
    ]

    plot_complexity_heatmaps_4(
        csv_files,
        titles,
        output_dir="output/plots",
        base_name="mnist_complexity"
    )
