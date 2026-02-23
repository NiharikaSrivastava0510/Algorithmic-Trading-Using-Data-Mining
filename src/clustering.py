"""
clustering.py — K-Means market regime detection.
=================================================
Responsibilities:
    1. Run the elbow method across a range of *k* values to help
       visualise the optimal number of clusters.
    2. Fit a K-Means model on the **scaled** historical forecasts
       (wind, solar, load) from the training set.
    3. Assign regime labels to both training and test sets.
    4. One-hot encode the regime labels so the neural network
       receives binary indicator columns rather than a single
       ordinal integer.

Why cluster?
    Energy markets exhibit distinct *regimes* — for example a
    low-load night with high wind behaves very differently from a
    midday solar peak with low demand.  Feeding the neural network
    an explicit regime indicator helps it learn regime-specific
    spread dynamics instead of averaging across all states.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

import config as cfg

# Suppress benign numerical warnings from sklearn's KMeans++ init
# on large scaled datasets (overflow in matmul during seed selection).
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class ElbowResult:
    """Stores k values and their corresponding inertias."""
    k_values: list[int]
    inertias: list[float]


@dataclass
class ClusterResult:
    """Stores a fitted KMeans model and the regime column names."""
    model: KMeans
    n_clusters: int
    regime_columns: list[str]


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def run_kmeans_elbow(
    train_scaled: pd.DataFrame,
    cluster_features: list[str] | None = None,
    k_range: range | None = None,
) -> ElbowResult:
    """
    Compute inertia for a range of *k* values (elbow method).

    Parameters
    ----------
    train_scaled : pd.DataFrame
        Scaled training data.
    cluster_features : list[str], optional
        Columns to cluster on. Defaults to ``config.CLUSTER_FEATURES``.
    k_range : range, optional
        Range of cluster counts to try. Defaults to ``config.KMEANS_K_RANGE``.

    Returns
    -------
    ElbowResult
        Contains the list of *k* values and their inertias.
    """
    cluster_features = cluster_features or cfg.CLUSTER_FEATURES
    k_range = k_range or cfg.KMEANS_K_RANGE

    data = train_scaled[cluster_features].values
    inertias = []

    for k in k_range:
        km = KMeans(
            n_clusters=k,
            random_state=cfg.KMEANS_RANDOM_STATE,
            n_init=cfg.KMEANS_N_INIT,
            max_iter=cfg.KMEANS_MAX_ITER,
        )
        km.fit(data)
        inertias.append(km.inertia_)

    return ElbowResult(k_values=list(k_range), inertias=inertias)


def fit_kmeans(
    train_scaled: pd.DataFrame,
    cluster_features: list[str] | None = None,
    n_clusters: int | None = None,
) -> ClusterResult:
    """
    Fit a K-Means model on the training data.

    Parameters
    ----------
    train_scaled : pd.DataFrame
        Scaled training data.
    cluster_features : list[str], optional
        Columns to cluster on. Defaults to ``config.CLUSTER_FEATURES``.
    n_clusters : int, optional
        Number of clusters. Defaults to ``config.OPTIMAL_K``.

    Returns
    -------
    ClusterResult
        Fitted model and regime column names.
    """
    cluster_features = cluster_features or cfg.CLUSTER_FEATURES
    n_clusters = n_clusters or cfg.OPTIMAL_K

    data = train_scaled[cluster_features].values

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=cfg.KMEANS_RANDOM_STATE,
        n_init=cfg.KMEANS_N_INIT,
        max_iter=cfg.KMEANS_MAX_ITER,
    )
    kmeans.fit(data)

    regime_cols = [f"regime_{i}" for i in range(n_clusters)]

    return ClusterResult(
        model=kmeans,
        n_clusters=n_clusters,
        regime_columns=regime_cols,
    )


def assign_regimes(
    df: pd.DataFrame,
    cluster_result: ClusterResult,
    cluster_features: list[str] | None = None,
) -> pd.DataFrame:
    """
    Predict regime labels and one-hot encode them.

    Parameters
    ----------
    df : pd.DataFrame
        Scaled DataFrame (train or test).
    cluster_result : ClusterResult
        Output from ``fit_kmeans()``.
    cluster_features : list[str], optional
        Columns to feed into the model. Defaults to ``config.CLUSTER_FEATURES``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with ``market_regime`` (int label) and
        ``regime_0`` … ``regime_{k-1}`` (one-hot) columns appended.
    """
    cluster_features = cluster_features or cfg.CLUSTER_FEATURES
    out = df.copy()

    available = [c for c in cluster_features if c in out.columns]
    data = out[available].values

    labels = cluster_result.model.predict(data)
    out["market_regime"] = labels

    for i in range(cluster_result.n_clusters):
        out[f"regime_{i}"] = (labels == i).astype(int)

    return out


def print_regime_summary(
    train_scaled: pd.DataFrame,
    train_raw: pd.DataFrame,
) -> None:
    """Print per-regime sample counts and spread statistics."""
    counts = train_scaled["market_regime"].value_counts().sort_index()
    n = len(train_scaled)

    for regime, count in counts.items():
        pct = count / n * 100
        mask = train_scaled["market_regime"] == regime
        mean_s = train_raw.loc[mask, cfg.TARGET].mean()
        std_s = train_raw.loc[mask, cfg.TARGET].std()
        print(
            f"    Regime {regime}: {count:>6} samples ({pct:5.1f}%) | "
            f"Spread mean={mean_s:>8.2f}, std={std_s:>8.2f}"
        )
