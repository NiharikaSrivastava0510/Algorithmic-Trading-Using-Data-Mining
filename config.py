"""
Configuration — Central settings for the entire pipeline.
==========================================================
All paths, feature lists, hyperparameters, and constants live here so
that every module draws from a single source of truth.
"""

import os

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw data — point this to wherever the ENSIMAG dataset is stored.
# Users who clone the repo should place the CSV files in  data/raw/
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

# Pipeline outputs
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ARTEFACT_DIR = os.path.join(OUTPUT_DIR, "artefacts")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

# Ensure output directories exist
for _d in [DATA_DIR, ARTEFACT_DIR, PLOT_DIR]:
    os.makedirs(_d, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# RAW FILE NAMES
# ──────────────────────────────────────────────────────────────
TRAIN_FILE = "train.csv"
IMBALANCES_FILE = "imbalances.csv"
TEST_FILE = "test.csv"
SAMPLE_FILE = "sample.csv"

# ──────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS
# ──────────────────────────────────────────────────────────────

# Core columns straight from the CSV
CORE_FEATURES = ["wind", "solar", "load"]

# Available only in the training set (not in test.csv)
TRAIN_ONLY_FEATURES = ["imbalances"]

# Target variable
TARGET = "spread"

# Temporal features engineered from the date column
TEMPORAL_FEATURES = [
    "hour_sin", "hour_cos",
    "month_sin", "month_cos",
    "dow_sin", "dow_cos",
    "interval_sin", "interval_cos",
    "is_weekend",
]

# Domain-specific features derived from core columns
DOMAIN_FEATURES = [
    "net_load",
    "renewable_ratio",
    "wind_solar_ratio",
    "total_renewable",
]

# Columns that require StandardScaler normalisation.
# Cyclical sin/cos are already in [-1, 1]; is_weekend is binary;
# regime one-hot columns are binary — none of those need scaling.
FEATURES_TO_SCALE = (
    CORE_FEATURES
    + TRAIN_ONLY_FEATURES
    + DOMAIN_FEATURES
)

# ──────────────────────────────────────────────────────────────
# K-MEANS CLUSTERING
# ──────────────────────────────────────────────────────────────
CLUSTER_FEATURES = ["wind", "solar", "load"]   # columns used for clustering
OPTIMAL_K = 5                                   # number of market regimes
KMEANS_RANDOM_STATE = 42
KMEANS_N_INIT = 10
KMEANS_MAX_ITER = 300
KMEANS_K_RANGE = range(2, 11)                   # range tested in elbow plot

# Regime one-hot column names (auto-generated from OPTIMAL_K)
REGIME_FEATURES = [f"regime_{i}" for i in range(OPTIMAL_K)]

# ──────────────────────────────────────────────────────────────
# FINAL FEATURE VECTOR (input to the neural network)
# ──────────────────────────────────────────────────────────────
FINAL_FEATURES = (
    CORE_FEATURES
    + TEMPORAL_FEATURES
    + DOMAIN_FEATURES
    + REGIME_FEATURES
)

# ──────────────────────────────────────────────────────────────
# DATA INTEGRITY
# ──────────────────────────────────────────────────────────────
EXPECTED_INTERVAL_MINUTES = 15   # the dataset is sampled at 15-min intervals
INTERVALS_PER_DAY = 96           # 24 * 60 / 15

# ──────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
