"""Draw the pattern and conflict graph for the sparse AD appendix example."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle


BLUE = "#2369a1"
ORANGE = "#b45d12"
COLORS = (BLUE, ORANGE, BLUE, BLUE)
PATTERN = ((1, 1, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, (matrix, graph) = plt.subplots(1, 2, figsize=(10.5, 3.7))
    for row, entries in enumerate(PATTERN):
        for col, present in enumerate(entries):
            matrix.add_patch(Rectangle(
                (col, 2 - row), 1, 1,
                facecolor=COLORS[col] if present else "#f5f6f7",
                edgecolor="white", linewidth=3,
            ))
            matrix.text(col + 0.5, 2.5 - row, str(present), ha="center",
                        va="center", fontsize=16,
                        color="white" if present else "#65717a")
    for col in range(4):
        matrix.text(col + 0.5, 3.16, str(col + 1), ha="center", va="bottom",
                    fontsize=12, color=COLORS[col], weight="bold")
    for row in range(3):
        matrix.text(-0.18, 2.5 - row, str(row + 1), ha="right", va="center",
                    fontsize=12)
    matrix.text(2, 3.68, "Input / column", ha="center", fontsize=12)
    matrix.text(-0.70, 1.5, "Output / row", rotation=90, ha="center",
                va="center", fontsize=12)
    matrix.text(2, -0.50, "1 = possible nonzero     0 = structural zero",
                ha="center", fontsize=10, color="#455563")
    matrix.set(xlim=(-1, 4.3), ylim=(-0.85, 4.5), aspect="equal")
    matrix.text(-0.80, 4.18, "a  Sparsity pattern", fontsize=14, weight="bold")
    matrix.axis("off")

    positions = ((0.35, 2.65), (2.05, 2.65), (3.75, 2.65), (2.05, 0.75))
    for left, right, label in ((0, 1, "row 1"), (1, 2, "row 2")):
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        graph.plot((x1, x2), (y1, y2), color="#455563", linewidth=2, zorder=1)
        graph.text((x1 + x2) / 2, y1 + 0.46, label, ha="center", fontsize=11)
    for index, (x, y) in enumerate(positions):
        graph.add_patch(Circle((x, y), 0.34, facecolor=COLORS[index],
                               edgecolor="white", linewidth=1.5, zorder=2))
        graph.text(x, y, str(index + 1), ha="center", va="center",
                   color="white", fontsize=15, weight="bold", zorder=3)
    graph.text(2.05, 1.65, "Same color = one seed group", ha="center",
               fontsize=11, color="#455563")
    graph.text(2.05, 0.12, "Column 4 conflicts with no other column",
               ha="center", fontsize=10, color="#455563")
    graph.text(2.05, -0.50, "Group 1: {1, 3, 4}     Group 2: {2}",
               ha="center", fontsize=11)
    graph.text(-0.25, 4.18, "b  Column conflict graph", fontsize=14, weight="bold")
    graph.set(xlim=(-0.45, 4.55), ylim=(-0.85, 4.5), aspect="equal")
    graph.axis("off")
    fig.subplots_adjust(left=0.015, right=0.99, bottom=0.02, top=0.99, wspace=0.12)
    output = Path(__file__).resolve().parent / "figures" / "sparse-autodiff"
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        path = output / f"intro-coloring.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
