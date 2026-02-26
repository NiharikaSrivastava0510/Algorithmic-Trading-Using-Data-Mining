"""
visualisation.py — Plotting functions for Step 1 and Step 2.
=============================================================
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


# ══════════════════════════════════════════════════════════════
# STEP 2 — PLOTS
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# AUTOCORRELATION
# ──────────────────────────────────────────────────────────────

def plot_autocorrelation(
    acf: dict[int, float],
    save_dir: str | None = None,
) -> str:
    """
    Bar chart of the spread's autocorrelation at chosen lags.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    lags = sorted(acf.keys())
    values = [acf[l] for l in lags]
    colours = ["#E91E63" if l in cfg.LAG_OFFSETS else "#9E9E9E" for l in lags]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar([str(l) for l in lags], values, color=colours, edgecolor="none")

    # Annotate the selected lag offsets
    for i, l in enumerate(lags):
        if l in cfg.LAG_OFFSETS:
            ax.annotate(
                cfg.LAG_OFFSETS[l],
                xy=(i, values[i]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8, fontweight="bold", color="#E91E63",
            )

    ax.set_xlabel("Lag (number of 15-min intervals)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Spread Autocorrelation — Justification for Lag Feature Selection")
    ax.axhline(0, color="black", linewidth=0.5)
    plt.tight_layout()

    path = os.path.join(save_dir, "spread_autocorrelation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# LAG FEATURE SCATTER
# ──────────────────────────────────────────────────────────────

def plot_lag_scatter(
    train: pd.DataFrame,
    save_dir: str | None = None,
) -> str:
    """
    Scatter plots: spread vs its lagged values at 15-min, 1-h, 24-h.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    lag_cols = [f"spread_lag_{k}" for k in cfg.LAG_OFFSETS]
    present = [c for c in lag_cols if c in train.columns]

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 5))
    if len(present) == 1:
        axes = [axes]

    for ax, col in zip(axes, present):
        k = int(col.split("_lag_")[-1])
        label = cfg.LAG_OFFSETS.get(k, f"{k} intervals")
        ax.scatter(train[col], train["spread"], alpha=0.02, s=1, color="#2196F3")
        ax.set_xlabel(f"Spread (t − {label})", fontsize=9)
        ax.set_ylabel("Spread (t)", fontsize=9)
        ax.set_title(f"Spread vs Lag-{k} ({label})")

        # Correlation annotation
        corr = train[["spread", col]].corr().iloc[0, 1]
        ax.annotate(
            f"r = {corr:.4f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=10, fontweight="bold", color="#E91E63",
            va="top",
        )

    plt.tight_layout()
    path = os.path.join(save_dir, "lag_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# ROLLING STATISTICS TIME SERIES
# ──────────────────────────────────────────────────────────────

def plot_rolling_timeseries(
    train: pd.DataFrame,
    save_dir: str | None = None,
) -> str:
    """
    Two-week close-up of load with its rolling mean and +/- 1 std band.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    # Take a 2-week slice for readability
    n_points = cfg.INTERVALS_PER_DAY * 14  # 14 days
    start = cfg.INTERVALS_PER_DAY * 180    # start ~6 months in
    end = start + n_points
    if end > len(train):
        start, end = 0, min(n_points, len(train))
    slc = train.iloc[start:end]

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    for ax, col, colour, title in zip(
        axes,
        ["load", "wind", "solar"],
        ["#4CAF50", "#2196F3", "#FF9800"],
        ["Load (MW)", "Wind (MW)", "Solar (MW)"],
    ):
        ax.plot(slc["date"], slc[col], linewidth=0.5, color=colour,
                alpha=0.5, label="Raw")

        # 1-hour rolling mean
        rmean_4 = f"{col}_rmean_4"
        if rmean_4 in slc.columns:
            ax.plot(slc["date"], slc[rmean_4], linewidth=1.2,
                    color="red", label="1h rolling mean")

        # 24-hour rolling mean + std band
        rmean_96 = f"{col}_rmean_96"
        rstd_96 = f"{col}_rstd_96"
        if rmean_96 in slc.columns and rstd_96 in slc.columns:
            ax.plot(slc["date"], slc[rmean_96], linewidth=1.2,
                    color="#333333", label="24h rolling mean")
            ax.fill_between(
                slc["date"],
                slc[rmean_96] - slc[rstd_96],
                slc[rmean_96] + slc[rstd_96],
                alpha=0.15, color="#333333", label="24h ±1 std",
            )

        ax.set_ylabel(title, fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Date")
    fig.suptitle("Rolling Statistics — 2-Week Close-Up", fontsize=13)
    plt.tight_layout()

    path = os.path.join(save_dir, "rolling_timeseries.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# FEATURE CORRELATION HEATMAP
# ──────────────────────────────────────────────────────────────

def plot_feature_correlation(
    train: pd.DataFrame,
    feature_cols: list[str] | None = None,
    save_dir: str | None = None,
) -> str:
    """
    Correlation heatmap of the final feature set vs the target.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR
    feature_cols = feature_cols or cfg.FINAL_FEATURES

    cols_present = [c for c in feature_cols if c in train.columns]
    if "spread" in train.columns:
        cols_present = cols_present + ["spread"]

    corr = train[cols_present].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols_present)))
    ax.set_yticks(range(len(cols_present)))
    ax.set_xticklabels(cols_present, rotation=90, fontsize=6)
    ax.set_yticklabels(cols_present, fontsize=6)
    ax.set_title("Feature Correlation Matrix (Step 2 Final Features)", fontsize=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()

    path = os.path.join(save_dir, "feature_correlation.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ══════════════════════════════════════════════════════════════
# STEP 3 — PLOTS
# ══════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────
# TRAINING CURVES
# ──────────────────────────────────────────────────────────────

def plot_training_curves(
    history: list,
    save_dir: str | None = None,
) -> str:
    """
    Plot training and validation loss curves with LR overlay.

    Parameters
    ----------
    history : list[EpochMetrics]
        From ``TrainingResult.history``.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    epochs = [m.epoch for m in history]
    train_loss = [m.train_loss for m in history]
    val_loss = [m.val_loss for m in history]
    lrs = [m.learning_rate for m in history]

    best_epoch = epochs[np.argmin(val_loss)]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Loss curves
    ax1.plot(epochs, train_loss, label="Train Loss", linewidth=1.5, color="#2196F3")
    ax1.plot(epochs, val_loss, label="Val Loss", linewidth=1.5, color="#E91E63")
    ax1.axvline(best_epoch, color="green", linestyle="--", alpha=0.7,
                label=f"Best epoch = {best_epoch}")
    ax1.set_ylabel("MSE Loss")
    ax1.set_title("Training and Validation Loss Curves")
    ax1.legend(fontsize=9)
    ax1.set_yscale("log")

    # Learning rate
    ax2.plot(epochs, lrs, color="#FF9800", linewidth=1.5)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_yscale("log")

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# PREDICTIONS VS ACTUAL
# ──────────────────────────────────────────────────────────────

def plot_predictions_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title_suffix: str = "Validation",
    save_dir: str | None = None,
) -> str:
    """
    Scatter plot and residual histogram for predicted vs actual spread.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: predicted vs actual
    ax1.scatter(y_true, y_pred, alpha=0.05, s=1, color="#2196F3")
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max()),
    ]
    ax1.plot(lims, lims, "r--", linewidth=1, label="Perfect prediction")
    ax1.set_xlabel("Actual Spread (EUR)")
    ax1.set_ylabel("Predicted Spread (EUR)")
    ax1.set_title(f"Predicted vs Actual — {title_suffix}")
    ax1.legend(fontsize=8)

    # Residual histogram
    residuals = y_pred - y_true
    ax2.hist(residuals, bins=200, color="#9C27B0", alpha=0.7, edgecolor="none")
    ax2.axvline(0, color="red", linestyle="--", linewidth=1)
    ax2.set_xlabel("Residual (Predicted − Actual) EUR")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Residual Distribution — {title_suffix}")

    # Stats annotation
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ax2.annotate(
        f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}",
        xy=(0.95, 0.95), xycoords="axes fraction",
        fontsize=10, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.7),
    )

    plt.tight_layout()
    fname = f"predictions_vs_actual_{title_suffix.lower().replace(' ', '_')}.png"
    path = os.path.join(save_dir, fname)
    fig.savefig(path)
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────
# VALIDATION TIME-SERIES COMPARISON
# ──────────────────────────────────────────────────────────────

def plot_val_timeseries(
    dates: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_days: int = 7,
    save_dir: str | None = None,
) -> str:
    """
    Overlay actual vs predicted spread over a 1-week window.

    Returns the path to the saved PNG.
    """
    save_dir = save_dir or cfg.PLOT_DIR

    # Take the first n_days of validation
    n_points = min(cfg.INTERVALS_PER_DAY * n_days, len(y_true))

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(dates[:n_points], y_true[:n_points],
            linewidth=0.8, color="#2196F3", alpha=0.8, label="Actual")
    ax.plot(dates[:n_points], y_pred[:n_points],
            linewidth=0.8, color="#E91E63", alpha=0.8, label="Predicted")
    ax.set_xlabel("Date")
    ax.set_ylabel("Spread (EUR)")
    ax.set_title(f"Actual vs Predicted Spread — First {n_days} Days of Validation")
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(save_dir, "val_timeseries.png")
    fig.savefig(path)
    plt.close(fig)
    return path
