#!/usr/bin/env python3
"""
run_step2.py — Pipeline orchestrator for Step 1 + Step 2.
==========================================================
Executes the full pipeline through both steps:

  Step 1
    1(a)  Load and merge the ENSIMAG IF 2025 CSV files.
    1(b)  Extract temporal and domain features.
    1(c)  Sort data and validate 15-minute interval integrity.

  Step 2
    2(a)  Target variable definition and statistical analysis.
    2(b)  Create lagged variables (15-min, 1-h, 24-h momentum).
    2(c)  Generate rolling averages and standard deviations.

  Combined
    Scale all features (Step 1 + Step 2) together.
    Apply K-Means clustering for market regime detection.
    Generate all visualisations and save artefacts.

Pipeline ordering rationale:
    Lag and rolling features are computed on **raw (unscaled)** data
    *after* temporal/domain features exist but *before* the scaler is
    fitted.  This ensures every numerical feature — including the new
    lags and rolling stats — is normalised consistently through a
    single StandardScaler fitted only on training data.

Usage
-----
    python run_step2.py                       # default paths
    python run_step2.py --data-dir ./my_data  # custom data location
"""

import argparse
import os
import sys
import time

import joblib
import pandas as pd

# Ensure the project root is on the Python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from src.data_loader import load_raw_data, merge_imbalances, print_data_summary
from src.feature_engineering import extract_temporal_features, extract_domain_features
from src.sequential_validator import (
    sort_by_date,
    validate_intervals,
    print_interval_report,
)
from src.lag_features import create_lag_features, fill_lag_nans, print_lag_summary
from src.rolling_features import create_rolling_features, print_rolling_summary
from src.target_analysis import (
    compute_target_stats,
    compute_autocorrelation,
    print_target_summary,
)
from src.scaler import fit_scalers, apply_scaling, print_scaling_summary
from src.clustering import (
    run_kmeans_elbow,
    fit_kmeans,
    assign_regimes,
    print_regime_summary,
)
from src.visualisation import (
    plot_elbow_curve,
    plot_regime_scatter,
    plot_data_overview,
    plot_spread_distribution,
    plot_autocorrelation,
    plot_lag_scatter,
    plot_rolling_timeseries,
    plot_feature_correlation,
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str | None = None) -> None:
    """Execute the full Step 1 + Step 2 pipeline."""
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════
    # STEP 1(a)  LOAD & MERGE
    # ═══════════════════════════════════════════════════════════
    _header("STEP 1(a): Loading and Merging Datasets")
    datasets = load_raw_data(data_dir)
    datasets["train"] = merge_imbalances(
        datasets["train"], datasets["imbalances"]
    )
    print_data_summary(datasets)

    train_raw = datasets["train"]
    test_raw = datasets["test"]
    sample = datasets["sample"]

    # ═══════════════════════════════════════════════════════════
    # STEP 1(b)  TEMPORAL & DOMAIN FEATURES
    # ═══════════════════════════════════════════════════════════
    _header("STEP 1(b): Extracting Temporal and Domain Features")

    train = extract_temporal_features(train_raw)
    train = extract_domain_features(train)

    test = extract_temporal_features(test_raw)
    test = extract_domain_features(test)

    print(f"  Core features:     {cfg.CORE_FEATURES}")
    print(f"  Temporal features: {cfg.TEMPORAL_FEATURES}")
    print(f"  Domain features:   {cfg.DOMAIN_FEATURES}")

    # ═══════════════════════════════════════════════════════════
    # STEP 1(c)  SEQUENTIAL STRUCTURING
    # ═══════════════════════════════════════════════════════════
    _header("STEP 1(c): Sequential Structuring (15-min intervals)")

    train = sort_by_date(train)
    test = sort_by_date(test)

    train_report = validate_intervals(train, label="Train")
    test_report = validate_intervals(test, label="Test")
    print_interval_report(train_report, label="Train")
    print_interval_report(test_report, label="Test")

    # ═══════════════════════════════════════════════════════════
    # STEP 2(a)  TARGET VARIABLE DEFINITION & ANALYSIS
    # ═══════════════════════════════════════════════════════════
    _header("STEP 2(a): Target Variable — Spread Definition & Analysis")

    target_stats = compute_target_stats(train)
    acf = compute_autocorrelation(train)
    print_target_summary(target_stats, acf)

    # ═══════════════════════════════════════════════════════════
    # STEP 2(b)  LAGGED VARIABLES
    # ═══════════════════════════════════════════════════════════
    _header("STEP 2(b): Creating Lagged Variables (15-min, 1-h, 24-h)")

    train = create_lag_features(train)
    test = create_lag_features(test)

    # Identify which lag columns exist in each set for NaN filling
    all_lag_cols = cfg.LAG_FEATURES + cfg.LAG_FEATURES_TRAIN_ONLY
    train = fill_lag_nans(train, all_lag_cols)
    test = fill_lag_nans(test, cfg.LAG_FEATURES)

    print("  Train:")
    print_lag_summary(train, is_train=True)
    print("\n  Test:")
    print_lag_summary(test, is_train=False)

    # ═══════════════════════════════════════════════════════════
    # STEP 2(c)  ROLLING AVERAGES & STANDARD DEVIATIONS
    # ═══════════════════════════════════════════════════════════
    _header("STEP 2(c): Generating Rolling Statistics (1-h, 24-h windows)")

    train = create_rolling_features(train)
    test = create_rolling_features(test)

    print("  Train:")
    print_rolling_summary(train, is_train=True)
    print("\n  Test:")
    print_rolling_summary(test, is_train=False)

    # Keep a copy of the raw (pre-scaled) train for plotting
    train_featured = train.copy()

    # ═══════════════════════════════════════════════════════════
    # SCALING  (Step 1(d) — now covers Step 2 features too)
    # ═══════════════════════════════════════════════════════════
    _header("Normalising and Scaling ALL Features (Step 1 + Step 2)")

    scaler_bundle = fit_scalers(
        train,
        features_to_scale=cfg.STEP2_FEATURES_TO_SCALE_TRAIN,
    )

    train_scaled = apply_scaling(train, scaler_bundle, is_train=True)
    test_scaled = apply_scaling(test, scaler_bundle, is_train=False)

    print_scaling_summary(train_scaled, train, scaler_bundle)

    # ═══════════════════════════════════════════════════════════
    # CLUSTERING  (Step 1(e) — unchanged)
    # ═══════════════════════════════════════════════════════════
    _header("K-Means Clustering for Market Regimes")

    elbow = run_kmeans_elbow(train_scaled)
    cluster_result = fit_kmeans(train_scaled)

    train_scaled = assign_regimes(train_scaled, cluster_result)
    test_scaled = assign_regimes(test_scaled, cluster_result)

    print(
        f"  K-Means with k={cluster_result.n_clusters} fitted "
        f"on {cfg.CLUSTER_FEATURES}"
    )
    print(f"\n  Cluster distribution (train):")
    print_regime_summary(train_scaled, train_featured)

    # ═══════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═══════════════════════════════════════════════════════════
    _header("Generating Visualisations")

    # Step 1 plots
    p1 = plot_elbow_curve(elbow)
    print(f"  Saved: {p1}")
    p2 = plot_regime_scatter(train_featured, train_scaled)
    print(f"  Saved: {p2}")
    p3 = plot_data_overview(train_featured)
    print(f"  Saved: {p3}")
    p4 = plot_spread_distribution(train_featured)
    print(f"  Saved: {p4}")

    # Step 2 plots
    p5 = plot_autocorrelation(acf)
    print(f"  Saved: {p5}")
    p6 = plot_lag_scatter(train_featured)
    print(f"  Saved: {p6}")
    p7 = plot_rolling_timeseries(train_featured)
    print(f"  Saved: {p7}")
    p8 = plot_feature_correlation(train_featured, cfg.FINAL_FEATURES)
    print(f"  Saved: {p8}")

    # ═══════════════════════════════════════════════════════════
    # SAVE ARTEFACTS
    # ═══════════════════════════════════════════════════════════
    _header("Saving Processed Data and Artefacts")

    train_scaled.to_csv(
        os.path.join(cfg.ARTEFACT_DIR, "train_prepared.csv"), index=False
    )
    test_scaled.to_csv(
        os.path.join(cfg.ARTEFACT_DIR, "test_prepared.csv"), index=False
    )
    sample.to_csv(
        os.path.join(cfg.ARTEFACT_DIR, "sample.csv"), index=False
    )

    joblib.dump(
        scaler_bundle.feature_scaler,
        os.path.join(cfg.ARTEFACT_DIR, "feature_scaler.pkl"),
    )
    joblib.dump(
        scaler_bundle.target_scaler,
        os.path.join(cfg.ARTEFACT_DIR, "target_scaler.pkl"),
    )
    joblib.dump(
        cluster_result.model,
        os.path.join(cfg.ARTEFACT_DIR, "kmeans_model.pkl"),
    )

    feature_config = {
        "core_features": cfg.CORE_FEATURES,
        "temporal_features": cfg.TEMPORAL_FEATURES,
        "domain_features": cfg.DOMAIN_FEATURES,
        "regime_features": cfg.REGIME_FEATURES,
        "lag_features": cfg.LAG_FEATURES,
        "lag_features_train_only": cfg.LAG_FEATURES_TRAIN_ONLY,
        "rolling_features": cfg.ROLLING_FEATURES,
        "rolling_features_train_only": cfg.ROLLING_FEATURES_TRAIN_ONLY,
        "train_only_features": cfg.TRAIN_ONLY_FEATURES,
        "final_features": cfg.FINAL_FEATURES,
        "features_to_scale": cfg.STEP2_FEATURES_TO_SCALE_TRAIN,
        "target": cfg.TARGET,
        "optimal_k": cfg.OPTIMAL_K,
    }
    joblib.dump(
        feature_config,
        os.path.join(cfg.ARTEFACT_DIR, "feature_config.pkl"),
    )

    print(f"  train_prepared.csv  — {train_scaled.shape}")
    print(f"  test_prepared.csv   — {test_scaled.shape}")
    print(f"  feature_scaler.pkl  — StandardScaler "
          f"({len(cfg.STEP2_FEATURES_TO_SCALE_TRAIN)} features)")
    print(f"  target_scaler.pkl   — StandardScaler (spread)")
    print(f"  kmeans_model.pkl    — KMeans (k={cfg.OPTIMAL_K})")
    print(f"  feature_config.pkl  — feature lists & config")

    print(f"\n  Final feature count for neural network: {len(cfg.FINAL_FEATURES)}")
    print(f"  Features ({len(cfg.FINAL_FEATURES)}):")
    # Print in grouped blocks for readability
    groups = [
        ("Core", cfg.CORE_FEATURES),
        ("Temporal", cfg.TEMPORAL_FEATURES),
        ("Domain", cfg.DOMAIN_FEATURES),
        ("Regime", cfg.REGIME_FEATURES),
        ("Lag", cfg.LAG_FEATURES),
        ("Rolling", cfg.ROLLING_FEATURES),
    ]
    for name, feats in groups:
        print(f"    {name:>10} ({len(feats):>2}): {feats}")

    elapsed = time.time() - t0
    _header(f"STEP 1 + STEP 2 COMPLETE  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1 + 2: Data Acquisition, Preparation, "
                    "Feature Engineering & Target Definition.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=(
            "Path to the folder containing train.csv, imbalances.csv, "
            "test.csv, and sample.csv.  Defaults to config.DATA_DIR."
        ),
    )
    args = parser.parse_args()
    run_pipeline(data_dir=args.data_dir)


if __name__ == "__main__":
    main()
