"""Plot gradient descent on a quadratic loss using NumPy and Matplotlib.

Run with: python docs/report/generate_gradient_descent_figure.py
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def loss(theta):
    return theta[..., 0]**2 + 2 * theta[..., 1]**2


def main():
    learning_rate = 0.12
    points = [np.array([2.6, -2.2])]
    for _ in range(20):
        gradient = np.array([2.0, 4.0]) * points[-1]
        points.append(points[-1] - learning_rate * gradient)
    path = np.array(points)
    heights = loss(path)
    if not np.all(np.diff(heights) < 0):
        raise ValueError("The illustrated loss must decrease at every step.")

    coordinates = np.linspace(-3, 3, 100)
    x, y = np.meshgrid(coordinates, coordinates)
    z = x**2 + 2 * y**2
    plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(7.6, 6.6))
    levels = [0.1, 0.3, 0.7, 1.5, 3, 5, 8, 12, 17, 23, 28]
    filled = ax.contourf(x, y, z, levels=[0, *levels], cmap="Blues", alpha=0.7)
    contours = ax.contour(x, y, z, levels=levels, colors="#537b9a",
                          linewidths=0.7, alpha=0.8)
    ax.clabel(contours, levels=[0.7, 3, 8, 17, 23], inline=True, fontsize=8)
    colorbar = fig.colorbar(filled, ax=ax, pad=0.03, shrink=0.85)
    colorbar.set_label(r"Loss $\mathcal{L}$")
    ax.plot(path[:, 0], path[:, 1], color="#bb3434",
            marker="o", markersize=3.6, linewidth=1.6,
            label="Gradient descent iterates", zorder=5)
    # Each arrow is the actual parameter update, minus eta times the gradient.
    for i in range(6):
        ax.annotate("", xy=path[i + 1], xytext=path[i],
                    arrowprops={"arrowstyle": "->", "color": "#bb3434",
                                "lw": 1.8, "mutation_scale": 13}, zorder=6)
    ax.scatter(*path[0], s=65, color="#bb3434", zorder=7)
    ax.annotate("Start", xy=path[0], xytext=(-12, -22),
                textcoords="offset points", ha="right", color="#9a2727")
    ax.scatter(0, 0, marker="*", s=170, color="#e6a323",
               edgecolor="#775400", linewidth=0.7, label="Minimum", zorder=7)
    ax.annotate("Minimum", xy=(0, 0), xytext=(-15, 16),
                textcoords="offset points", ha="right")
    ax.set(xlabel=r"Parameter $\theta_1$", ylabel=r"Parameter $\theta_2$",
           xlim=(-3, 3), ylim=(-3, 3),
           title=r"Gradient descent on $\mathcal{L}=\theta_1^2+2\theta_2^2$")
    ax.set_aspect("equal")
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=9)
    fig.text(0.48, 0.04,
             r"$\boldsymbol{\theta}_{k+1}=\boldsymbol{\theta}_k"
             r"-\eta\,\nabla\mathcal{L}(\boldsymbol{\theta}_k)$"
             "    |    " + r"$\eta=0.12$", ha="center", fontsize=12)
    fig.subplots_adjust(left=0.11, right=0.94, bottom=0.15, top=0.92)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = output_dir / f"gradient-descent-contours.{suffix}"
        fig.savefig(output, dpi=250, bbox_inches="tight")
        print(f"Wrote {output}")
    plt.close(fig)
    print(f"Loss decreases from {heights[0]:.6f} to {heights[-1]:.6f} in 20 steps.")


if __name__ == "__main__":
    main()
