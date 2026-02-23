"""
src — Step 1 modules for data acquisition and preparation.
"""

from src.data_loader import load_raw_data, merge_imbalances
from src.feature_engineering import extract_temporal_features, extract_domain_features
from src.sequential_validator import validate_intervals, sort_by_date
from src.scaler import fit_scalers, apply_scaling
from src.clustering import run_kmeans_elbow, fit_kmeans, assign_regimes
from src.visualisation import (
    plot_elbow_curve,
    plot_regime_scatter,
    plot_data_overview,
    plot_spread_distribution,
)

__all__ = [
    "load_raw_data",
    "merge_imbalances",
    "extract_temporal_features",
    "extract_domain_features",
    "validate_intervals",
    "sort_by_date",
    "fit_scalers",
    "apply_scaling",
    "run_kmeans_elbow",
    "fit_kmeans",
    "assign_regimes",
    "plot_elbow_curve",
    "plot_regime_scatter",
    "plot_data_overview",
    "plot_spread_distribution",
]
