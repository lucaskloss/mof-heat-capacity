"""Draw a simple feedforward network for the report using Matplotlib.

Run with: python docs/report/generate_neural_network_figure.py
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np


def main():
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 5.3))
    layers = [np.column_stack((np.full(n, x), np.linspace((n - 1) / 2, -(n - 1) / 2, n)))
              for x, n in zip((0, 2.6, 5.2, 7.8), (3, 4, 3, 1))]
    colors = ("#dcecf8", "#dceee4", "#dceee4", "#fff0ce")
    outlines = ("#2369a1", "#23855b", "#23855b", "#a87915")
    labels = ("Input layer", "Hidden layer 1", "Hidden layer 2", "Output layer")
    for previous, following in zip(layers[:-1], layers[1:]):
        for start in previous:
            for end in following:
                ax.add_patch(FancyArrowPatch(start, end, arrowstyle="->",
                             mutation_scale=8, shrinkA=18, shrinkB=18,
                             color="#8295a5", alpha=0.5, linewidth=0.8, zorder=1))
    for layer_index, (nodes, fill, edge, title) in enumerate(zip(layers, colors, outlines, labels)):
        ax.text(nodes[0, 0], 2.05, title, ha="center", weight="bold")
        for node_index, (x, y) in enumerate(nodes, start=1):
            ax.add_patch(Circle((x, y), 0.26, facecolor=fill, edgecolor=edge,
                                linewidth=1.5, zorder=3))
            if layer_index == 0:
                label = rf"$x_{node_index}$"
            elif layer_index == 3:
                label = r"$\hat{y}$"
            else:
                label = rf"$h_{node_index}^{{({layer_index})}}$"
            ax.text(x, y, label, ha="center", va="center", fontsize=11, zorder=4)
    ax.annotate("", xy=(7.8, 2.7), xytext=(0, 2.7),
                arrowprops={"arrowstyle": "->", "color": "#2369a1", "lw": 1.7})
    ax.text(3.9, 2.85, "Forward pass: inputs to prediction", ha="center", color="#2369a1")
    ax.annotate("", xy=(0, -2.15), xytext=(7.8, -2.15),
                arrowprops={"arrowstyle": "->", "color": "#bb3434", "lw": 1.5, "ls": "--"})
    ax.text(3.9, -2.5, "Training: loss gradients propagate backward", ha="center", color="#bb3434")
    ax.text(3.9, -3.05,
            r"Hidden neuron: $h_j^{(\ell+1)}=\sigma\!\left(\sum_i W_{ji}^{(\ell)}h_i^{(\ell)}+b_j^{(\ell)}\right)$",
            ha="center", fontsize=12)
    ax.text(3.9, -3.48, "Connections carry learned weights; each hidden neuron adds a bias and applies an activation.",
            ha="center", fontsize=10, color="#455563")
    ax.set(xlim=(-0.9, 8.7), ylim=(-3.75, 3.2), aspect="equal")
    ax.axis("off")
    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = output_dir / f"simple-neural-network.{suffix}"
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
