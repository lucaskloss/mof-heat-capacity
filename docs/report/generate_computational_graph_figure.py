"""Draw forward and reverse AD passes for the report's scalar-neuron example.

Run with: python docs/report/generate_computational_graph_figure.py
"""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


POSITIONS = {
    "x": (0.8, 2.25), "w": (0.8, 0.55),
    "u": (3.2, 2.25), "b": (3.2, 0.55),
    "z": (5.6, 2.25), "prediction": (8.0, 2.25),
    "y": (8.0, 0.55), "r": (10.4, 2.25), "loss": (12.8, 2.25),
}
EDGES = (
    ("x", "u"), ("w", "u"), ("u", "z"), ("b", "z"),
    ("z", "prediction"), ("prediction", "r"), ("y", "r"), ("r", "loss"),
)


def draw_panel(ax, reverse=False):
    color = "#b43d3d" if reverse else "#2369a1"
    labels = {
        "x": (r"$x=2$", r"$\bar{x}=-0.5$"),
        "w": (r"$w=0.5$", r"$\bar{w}=-2$"),
        "u": (r"$u=wx$", r"$\bar{u}=-1$"),
        "b": (r"$b=-1$", r"$\bar{b}=-1$"),
        "z": (r"$z=u+b$", r"$\bar{z}=-1$"),
        "prediction": (r"$\hat{y}=\tanh z$", r"$\overline{\hat{y}}=-1$"),
        "y": (r"$y=1$", r"$\bar{y}=1$"),
        "r": (r"$r=\hat{y}-y$", r"$\bar{r}=-1$"),
        "loss": (r"$l=r^2/2$", r"$\bar{l}=1$"),
    }
    values = {"u": "$u=1$", "z": "$z=0$", "prediction": r"$\hat{y}=0$",
              "r": "$r=-1$", "loss": "$l=0.5$"}
    boxes = {}
    for key, (x, y) in POSITIONS.items():
        parameter = key in {"w", "b"}
        fill = "#e3f1e9" if parameter else ("#fbecec" if reverse else "#eaf2f9")
        edge = "#23855b" if parameter else color
        box = FancyBboxPatch((x - 0.97, y - 0.46), 1.94, 0.92,
                             boxstyle="round,pad=0.035,rounding_size=0.09",
                             facecolor=fill, edgecolor=edge, linewidth=1.3, zorder=3)
        ax.add_patch(box)
        boxes[key] = box
        if reverse:
            ax.text(x, y, labels[key][1], ha="center", va="center", fontsize=14, zorder=4)
        else:
            ax.text(x, y + (0.16 if key in values else 0), labels[key][0],
                    ha="center", va="center", fontsize=13, zorder=4)
            if key in values:
                ax.text(x, y - 0.22, values[key], ha="center", va="center",
                        fontsize=12, color="#455563", zorder=4)
    for source, target in EDGES:
        if reverse:
            source, target = target, source
        ax.add_patch(FancyArrowPatch(POSITIONS[source], POSITIONS[target],
                                     patchA=boxes[source], patchB=boxes[target],
                                     arrowstyle="-|>", mutation_scale=12,
                                     shrinkA=4, shrinkB=4, color=color,
                                     linewidth=1.5, zorder=2))
    if reverse:
        for x, text in [(4.4, "$1$"), (6.8, r"$1-\hat{y}^{\,2}=1$"),
                        (9.2, "$1$"), (11.6, "$r=-1$")]:
            ax.text(x, 3.06, text, ha="center", va="center", fontsize=12, color=color)
        ax.text(1.83, 3.06, "$w=0.5$", ha="center", fontsize=12, color=color)
        ax.text(1.65, 1.50, "$x=2$", ha="center", fontsize=12, color=color)
        ax.text(4.40, 1.15, "$1$", ha="center", fontsize=12, color=color)
        ax.text(9.20, 1.15, "$-1$", ha="center", fontsize=12, color=color)
        title = "b  Reverse pass: propagate adjoints from the loss"
        footer = r"Arrow labels: local derivatives. At each step, multiply by the incoming adjoint."
    else:
        title = "a  Forward pass: evaluate operations and retain intermediate values"
        footer = r"Green boxes: trainable parameters. $x$ is the input and $y$ is the fixed reference target."
    ax.text(-0.3, 3.72, title, fontsize=14, weight="bold", color=color)
    ax.text(6.7, -0.45, footer, ha="center", fontsize=12, color="#455563")
    ax.set(xlim=(-0.4, 14.05), ylim=(-0.65, 4.05))
    ax.axis("off")


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.5))
    draw_panel(axes[0])
    draw_panel(axes[1], reverse=True)
    fig.subplots_adjust(left=0.025, right=0.99, bottom=0.03, top=0.98, hspace=0.15)
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = output_dir / f"computational-graph-backpropagation.{suffix}"
        fig.savefig(output, dpi=220, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
