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

# Step 1 final features (before Step 2 additions)
STEP1_FINAL_FEATURES = (
    CORE_FEATURES
    + TEMPORAL_FEATURES
    + DOMAIN_FEATURES
    + REGIME_FEATURES
)

# ──────────────────────────────────────────────────────────────
# STEP 2 — LAGGED VARIABLES
# ──────────────────────────────────────────────────────────────

# Lag offsets in number of 15-minute intervals
LAG_OFFSETS = {
    1: "15min",      # previous interval
    4: "1h",         # 1 hour ago
    96: "24h",       # 24 hours ago
}

# Columns on which to create lags (available in both train AND test)
LAG_COLUMNS = ["wind", "solar", "load", "net_load"]

# Spread lags — only available in training data (test has no spread)
LAG_COLUMNS_TRAIN_ONLY = ["spread"]

# Auto-generated lag feature names
LAG_FEATURES = [
    f"{col}_lag_{k}" for col in LAG_COLUMNS for k in LAG_OFFSETS
]
LAG_FEATURES_TRAIN_ONLY = [
    f"{col}_lag_{k}" for col in LAG_COLUMNS_TRAIN_ONLY for k in LAG_OFFSETS
]

# ──────────────────────────────────────────────────────────────
# STEP 2 — ROLLING STATISTICS
# ──────────────────────────────────────────────────────────────

# Rolling window sizes in number of 15-minute intervals
ROLLING_WINDOWS = {
    4: "1h",         # 1-hour rolling window
    96: "24h",       # 24-hour rolling window
}

# Columns on which to compute rolling stats (train + test)
ROLLING_COLUMNS = ["wind", "solar", "load"]

# Spread rolling stats — training only
ROLLING_COLUMNS_TRAIN_ONLY = ["spread"]

# Auto-generated rolling feature names
ROLLING_FEATURES = [
    f"{col}_rmean_{w}" for col in ROLLING_COLUMNS for w in ROLLING_WINDOWS
] + [
    f"{col}_rstd_{w}" for col in ROLLING_COLUMNS for w in ROLLING_WINDOWS
]
ROLLING_FEATURES_TRAIN_ONLY = [
    f"{col}_rmean_{w}" for col in ROLLING_COLUMNS_TRAIN_ONLY for w in ROLLING_WINDOWS
] + [
    f"{col}_rstd_{w}" for col in ROLLING_COLUMNS_TRAIN_ONLY for w in ROLLING_WINDOWS
]

# ──────────────────────────────────────────────────────────────
# STEP 2 — COMBINED SCALING LIST
# ──────────────────────────────────────────────────────────────

# All features that need StandardScaler (Step 1 + Step 2).
# Cyclical sin/cos, is_weekend, and regime one-hots remain unscaled.
STEP2_FEATURES_TO_SCALE = (
    FEATURES_TO_SCALE        # Step 1: core + imbalances + domain
    + LAG_FEATURES           # Step 2: lags (test-available)
    + ROLLING_FEATURES       # Step 2: rolling stats (test-available)
)

# The *training* scaler also includes train-only columns so that
# they are normalised consistently during training.
STEP2_FEATURES_TO_SCALE_TRAIN = (
    STEP2_FEATURES_TO_SCALE
    + LAG_FEATURES_TRAIN_ONLY
    + ROLLING_FEATURES_TRAIN_ONLY
)

# ──────────────────────────────────────────────────────────────
# FINAL FEATURE VECTOR (input to the neural network)
# ──────────────────────────────────────────────────────────────

# After Step 2, the full feature vector used for modelling:
FINAL_FEATURES = (
    CORE_FEATURES
    + TEMPORAL_FEATURES
    + DOMAIN_FEATURES
    + REGIME_FEATURES
    + LAG_FEATURES
    + ROLLING_FEATURES
)

# ──────────────────────────────────────────────────────────────
# STEP 3 — LSTM ARCHITECTURE
# ──────────────────────────────────────────────────────────────

# Sequence windowing
SEQUENCE_LENGTH = 96              # 1 full day of 15-min intervals

# LSTM layers
LSTM_INPUT_SIZE = len(FINAL_FEATURES)   # 45 features
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.2                # dropout between stacked LSTM layers

# Dense head (after LSTM)
DENSE_HIDDEN_SIZE = 64
DENSE_DROPOUT = 0.3

# ──────────────────────────────────────────────────────────────
# STEP 3 — TRAINING
# ──────────────────────────────────────────────────────────────

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5               # L2 regularisation (AdamW)
MAX_EPOCHS = 100
GRADIENT_CLIP_NORM = 1.0          # prevent exploding gradients in LSTM

# Early stopping
EARLY_STOPPING_PATIENCE = 10     # epochs without val improvement

# Learning-rate scheduler (ReduceLROnPlateau)
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 5

# Train/validation split (chronological — last portion held out)
VALIDATION_FRACTION = 0.2         # last 20 % of training data

# ──────────────────────────────────────────────────────────────
# DATA INTEGRITY
# ──────────────────────────────────────────────────────────────
EXPECTED_INTERVAL_MINUTES = 15   # the dataset is sampled at 15-min intervals
INTERVALS_PER_DAY = 96           # 24 * 60 / 15

# ──────────────────────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
