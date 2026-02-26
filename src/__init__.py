"""
src — Pipeline modules for data acquisition, preparation, feature
engineering, and neural network training.
"""

# Step 1
from src.data_loader import load_raw_data, merge_imbalances
from src.feature_engineering import extract_temporal_features, extract_domain_features
from src.sequential_validator import validate_intervals, sort_by_date
from src.scaler import fit_scalers, apply_scaling
from src.clustering import run_kmeans_elbow, fit_kmeans, assign_regimes

# Step 2
from src.lag_features import create_lag_features, fill_lag_nans
from src.rolling_features import create_rolling_features
from src.target_analysis import compute_target_stats, compute_autocorrelation

# Step 3
from src.dataset import (
    load_prepared_data,
    load_test_data,
    temporal_train_val_split,
    SpreadSequenceDataset,
    build_dataloaders,
)
from src.model import SpreadLSTM, get_device, count_parameters
from src.trainer import Trainer

from src.visualisation import (
    # Step 1
    plot_elbow_curve,
    plot_regime_scatter,
    plot_data_overview,
    plot_spread_distribution,
    # Step 2
    plot_autocorrelation,
    plot_lag_scatter,
    plot_rolling_timeseries,
    plot_feature_correlation,
    # Step 3
    plot_training_curves,
    plot_predictions_vs_actual,
    plot_val_timeseries,
)

__all__ = [
    # Step 1
    "load_raw_data", "merge_imbalances",
    "extract_temporal_features", "extract_domain_features",
    "validate_intervals", "sort_by_date",
    "fit_scalers", "apply_scaling",
    "run_kmeans_elbow", "fit_kmeans", "assign_regimes",
    # Step 2
    "create_lag_features", "fill_lag_nans",
    "create_rolling_features",
    "compute_target_stats", "compute_autocorrelation",
    # Step 3
    "load_prepared_data", "load_test_data",
    "temporal_train_val_split", "SpreadSequenceDataset", "build_dataloaders",
    "SpreadLSTM", "get_device", "count_parameters",
    "Trainer",
    # Visualisations
    "plot_elbow_curve", "plot_regime_scatter",
    "plot_data_overview", "plot_spread_distribution",
    "plot_autocorrelation", "plot_lag_scatter",
    "plot_rolling_timeseries", "plot_feature_correlation",
    "plot_training_curves", "plot_predictions_vs_actual", "plot_val_timeseries",
]
