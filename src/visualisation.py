"""
visualisation.py — All plotting functions for Step 1.
=====================================================
Each function creates a self-contained figure, saves it to
``config.PLOT_DIR``, and closes the figure to free memory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as cfg
from src.clustering import ElbowResult


# ──────────────────────────────────────────────────────────────
# STYLE DEFAULTS
# ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.facecolor": "white",
})


# ──────────────────────────────────────────────────────────────
# ELBOW CURVE
# ──────────────────────────────────────────────────────────────

def plot_elbow_curve(
    elbow: ElbowResult,
    save_dir: str | None = None,
) -> str:
    """
    Plot the elbow curve (inertia vs k) for K-Means.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(elbow.k_values, elbow.inertias, "bo-", linewidth=1.5, markersize=6)
    ax.axvline(x=cfg.OPTIMAL_K, color="red", linestyle="--", alpha=0.7,
               label=f"Selected k = {cfg.OPTIMAL_K}")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (within-cluster sum of squares)")
    ax.set_title("Elbow Method for Optimal k — Market Regime Clustering")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(save_dir, "kmeans_elbow.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# REGIME SCATTER PLOTS
# ──────────────────────────────────────────────────────────────

def plot_regime_scatter(
    train_raw: pd.DataFrame,
    train_scaled: pd.DataFrame,
    save_dir: str | None = None,
) -> str:
    """
    Three-panel scatter plot showing how the K-Means regimes
    separate along different feature axes.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR
    n_regimes = train_scaled["market_regime"].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for regime in range(n_regimes):
        mask = train_scaled["market_regime"] == regime
        lbl = f"Regime {regime}"

        axes[0].scatter(
            train_raw.loc[mask, "wind"], train_raw.loc[mask, "solar"],
            alpha=0.08, s=1, label=lbl,
        )
        axes[1].scatter(
            train_raw.loc[mask, "load"], train_raw.loc[mask, "spread"],
            alpha=0.08, s=1, label=lbl,
        )
        axes[2].scatter(
            train_raw.loc[mask, "wind"], train_raw.loc[mask, "load"],
            alpha=0.08, s=1, label=lbl,
        )

    axes[0].set_xlabel("Wind Forecast (MW)")
    axes[0].set_ylabel("Solar Forecast (MW)")
    axes[0].set_title("Wind vs Solar by Market Regime")
    axes[0].legend(markerscale=10, fontsize=8)

    axes[1].set_xlabel("Load Forecast (MW)")
    axes[1].set_ylabel("Spread (EUR)")
    axes[1].set_title("Load vs Spread by Market Regime")
    axes[1].legend(markerscale=10, fontsize=8)

    axes[2].set_xlabel("Wind Forecast (MW)")
    axes[2].set_ylabel("Load Forecast (MW)")
    axes[2].set_title("Wind vs Load by Market Regime")
    axes[2].legend(markerscale=10, fontsize=8)

    plt.tight_layout()
    path = os.path.join(save_dir, "kmeans_regimes.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# DATA OVERVIEW
# ──────────────────────────────────────────────────────────────

def plot_data_overview(
    train_raw: pd.DataFrame,
    save_dir: str | None = None,
) -> str:
    """
    Four-panel time-series overview of the core features + spread.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)

    # Downsample for cleaner plots (plot every 4th point = hourly)
    step = 4
    dates = train_raw["date"].iloc[::step]

    for ax, col, colour, title in zip(
        axes,
        ["wind", "solar", "load", "spread"],
        ["#2196F3", "#FF9800", "#4CAF50", "#E91E63"],
        [
            "Wind Power Forecast (MW)",
            "Solar Power Forecast (MW)",
            "Electricity Load Forecast (MW)",
            "Market Spread (EUR)",
        ],
    ):
        ax.plot(dates, train_raw[col].iloc[::step],
                linewidth=0.3, color=colour, alpha=0.7)
        ax.set_ylabel(title, fontsize=9)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("ENSIMAG IF 2025 — Training Data Overview", fontsize=13)
    plt.tight_layout()

    path = os.path.join(save_dir, "data_overview.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# SPREAD DISTRIBUTION
# ──────────────────────────────────────────────────────────────

def plot_spread_distribution(
    train_raw: pd.DataFrame,
    save_dir: str | None = None,
) -> str:
    """
    Histogram and box plot of the target variable.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

    spread = train_raw["spread"]

    # Histogram
    ax1.hist(spread, bins=200, color="#9C27B0", alpha=0.7, edgecolor="none")
    ax1.axvline(spread.mean(), color="red", linestyle="--", label=f"Mean = {spread.mean():.1f}")
    ax1.axvline(spread.median(), color="blue", linestyle="--", label=f"Median = {spread.median():.1f}")
    ax1.set_xlabel("Spread (EUR)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Distribution of Market Spread")
    ax1.legend(fontsize=8)

    # Box plot
    ax2.boxplot(spread, vert=True, widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor="#E1BEE7", color="#9C27B0"),
                medianprops=dict(color="red"))
    ax2.set_ylabel("Spread (EUR)")
    ax2.set_title("Box Plot of Market Spread")

    plt.tight_layout()
    path = os.path.join(save_dir, "spread_distribution.png")
    fig.savefig(path)
    plt.close(fig)
    return path
