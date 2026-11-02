"""Generate the report's reproducible illustration of regression and overfitting.

Run with: python docs/report/generate_regression_figures.py
Only NumPy and Matplotlib are required. All data are synthetic.
"""

from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mof-matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial


def underlying_function(x):
    return 0.5 + 0.7 * x - 0.5 * x**2 + x**3


def main():
    rng = np.random.default_rng(7)
    x_train = np.linspace(-1.0, 1.0, 12)
    y_train = underlying_function(x_train) + rng.normal(0.0, 0.18, x_train.size)
    x_validation = rng.uniform(-1.0, 1.0, 200)
    y_validation = underlying_function(x_validation) + rng.normal(0.0, 0.18, 200)
    grid = np.linspace(-1.0, 1.0, 800)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.9), sharex=True,
                             sharey=True, layout="constrained")
    titles = ("Linear fit (degree 1)", "Polynomial fit (degree 3)",
              "Polynomial fit (degree 11)")
    for ax, degree, title in zip(axes, (1, 3, 11), titles):
        model = Polynomial.fit(x_train, y_train, degree)
        train_rmse = np.sqrt(np.mean((model(x_train) - y_train)**2))
        validation_rmse = np.sqrt(np.mean((model(x_validation) - y_validation)**2))
        ax.plot(grid, underlying_function(grid), color="#2369a1", ls="--",
                lw=1.8, label="Underlying function")
        ax.plot(grid, model(grid), color="#23855b", lw=2, label="Fitted model")
        ax.scatter(x_train, y_train, color="#c94c4c", s=30, zorder=3,
                   edgecolors="white", linewidths=0.5, label="Training data")
        ax.set(title=title, xlabel="$x$", xlim=(-1.05, 1.05), ylim=(-2.35, 2.2))
        ax.grid(alpha=0.18)
        ax.text(0.04, 0.96,
                f"Training RMSE: {train_rmse:.3f}\nValidation RMSE: {validation_rmse:.3f}",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
        print(f"Degree {degree}: training RMSE={train_rmse:.6f}, "
              f"validation RMSE={validation_rmse:.6f}")
    axes[0].set_ylabel("$y$")
    axes[1].legend(loc="lower right", fontsize=8, frameon=False)

    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        output = output_dir / f"regression-fits.{suffix}"
        fig.savefig(output, dpi=250)
        print(f"Wrote {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
