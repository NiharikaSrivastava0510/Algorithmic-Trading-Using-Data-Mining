#!/usr/bin/env python3
"""
run_step1.py — Main pipeline orchestrator for Step 1.
=====================================================
Executes the full Data Acquisition & Preparation pipeline:

    1(a)  Load and merge the ENSIMAG IF 2025 CSV files.
    1(b)  Extract temporal and domain features.
    1(c)  Sort data and validate 15-minute interval integrity.
    1(d)  Normalise and scale all numerical features.
    1(e)  Apply K-Means clustering for market regime detection.

All outputs (prepared CSVs, fitted scalers, KMeans model, plots)
are saved to ``outputs/``.

Usage
-----
    python run_step1.py                       # default paths from config
    python run_step1.py --data-dir ./my_data  # custom data location
"""

import argparse
import os
import sys
import time

import joblib
import pandas as pd

# Ensure the project root is on the Python path so ``config`` and
# ``src`` are importable regardless of the working directory.
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
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    """Print a bold section header."""
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline(data_dir: str | None = None) -> None:
    """Execute the full Step 1 pipeline."""
    t0 = time.time()

    # ── 1(a)  LOAD & MERGE ──────────────────────────────────
    _header("STEP 1(a): Loading and Merging Datasets")
    datasets = load_raw_data(data_dir)
    datasets["train"] = merge_imbalances(
        datasets["train"], datasets["imbalances"]
    )
    print_data_summary(datasets)

    train_raw = datasets["train"]
    test_raw = datasets["test"]
    sample = datasets["sample"]

    # ── 1(b)  FEATURE ENGINEERING ───────────────────────────
    _header("STEP 1(b): Extracting Core Features")

    train_feat = extract_temporal_features(train_raw)
    train_feat = extract_domain_features(train_feat)

    test_feat = extract_temporal_features(test_raw)
    test_feat = extract_domain_features(test_feat)

    print(f"  Core features:      {cfg.CORE_FEATURES}")
    print(f"  Temporal features:  {cfg.TEMPORAL_FEATURES}")
    print(f"  Domain features:    {cfg.DOMAIN_FEATURES}")
    print(f"  Train-only extras:  {cfg.TRAIN_ONLY_FEATURES}")
    print(
        f"  Total features for model: "
        f"{len(cfg.CORE_FEATURES) + len(cfg.TEMPORAL_FEATURES) + len(cfg.DOMAIN_FEATURES)} "
        f"(+{len(cfg.TRAIN_ONLY_FEATURES)} train-only)"
    )

    # ── 1(c)  SEQUENTIAL STRUCTURING ────────────────────────
    _header("STEP 1(c): Sequential Structuring (15-min intervals)")

    train_feat = sort_by_date(train_feat)
    test_feat = sort_by_date(test_feat)

    train_report = validate_intervals(train_feat, label="Train")
    test_report = validate_intervals(test_feat, label="Test")

    print_interval_report(train_report, label="Train")
    print_interval_report(test_report, label="Test")

    # ── 1(d)  NORMALISATION & SCALING ───────────────────────
    _header("STEP 1(d): Normalising and Scaling Features")

    scaler_bundle = fit_scalers(train_feat)

    train_scaled = apply_scaling(train_feat, scaler_bundle, is_train=True)
    test_scaled = apply_scaling(test_feat, scaler_bundle, is_train=False)

    print_scaling_summary(train_scaled, train_feat, scaler_bundle)

    # ── 1(e)  K-MEANS CLUSTERING ────────────────────────────
    _header("STEP 1(e): K-Means Clustering for Market Regimes")

    elbow = run_kmeans_elbow(train_scaled)
    cluster_result = fit_kmeans(train_scaled)

    train_scaled = assign_regimes(train_scaled, cluster_result)
    test_scaled = assign_regimes(test_scaled, cluster_result)

    print(
        f"  K-Means with k={cluster_result.n_clusters} fitted "
        f"on {cfg.CLUSTER_FEATURES}"
    )
    print(f"\n  Cluster distribution (train):")
    print_regime_summary(train_scaled, train_feat)

    # ── PLOTS ───────────────────────────────────────────────
    _header("Generating Visualisations")

    p1 = plot_elbow_curve(elbow)
    print(f"  Saved: {p1}")

    p2 = plot_regime_scatter(train_feat, train_scaled)
    print(f"  Saved: {p2}")

    p3 = plot_data_overview(train_feat)
    print(f"  Saved: {p3}")

    p4 = plot_spread_distribution(train_feat)
    print(f"  Saved: {p4}")

    # ── SAVE ARTEFACTS ──────────────────────────────────────
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

    # Feature configuration dictionary for downstream steps
    feature_config = {
        "core_features": cfg.CORE_FEATURES,
        "temporal_features": cfg.TEMPORAL_FEATURES,
        "domain_features": cfg.DOMAIN_FEATURES,
        "regime_features": cfg.REGIME_FEATURES,
        "train_only_features": cfg.TRAIN_ONLY_FEATURES,
        "final_features": cfg.FINAL_FEATURES,
        "features_to_scale": cfg.FEATURES_TO_SCALE,
        "target": cfg.TARGET,
        "optimal_k": cfg.OPTIMAL_K,
    }
    joblib.dump(
        feature_config,
        os.path.join(cfg.ARTEFACT_DIR, "feature_config.pkl"),
    )

    print(f"  train_prepared.csv  — {train_scaled.shape}")
    print(f"  test_prepared.csv   — {test_scaled.shape}")
    print(f"  feature_scaler.pkl  — StandardScaler ({len(cfg.FEATURES_TO_SCALE)} features)")
    print(f"  target_scaler.pkl   — StandardScaler (spread)")
    print(f"  kmeans_model.pkl    — KMeans (k={cfg.OPTIMAL_K})")
    print(f"  feature_config.pkl  — feature lists & config")

    print(f"\n  Final feature count for neural network: {len(cfg.FINAL_FEATURES)}")
    print(f"  Features: {cfg.FINAL_FEATURES}")

    elapsed = time.time() - t0
    _header(f"STEP 1 COMPLETE  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Data Acquisition & Preparation for "
                    "the ENSIMAG IF 2025 spread prediction project.",
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
