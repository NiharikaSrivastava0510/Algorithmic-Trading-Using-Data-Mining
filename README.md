# Algorithmic Trading Using Data Mining

**Electricity Market Spread Prediction** via Neural Networks, ARIMA, and LLMs.

A data-mining project that predicts the German electricity market spread (imbalance price minus day-ahead price) for every 15-minute interval, using the [ENSIMAG IF 2025 Algorithmic Trading](https://www.kaggle.com/competitions/ensimag-if-2025/data) competition dataset.

> **University of Southampton** -- COMP6248 Data Mining Coursework
>
> Vaishnavi Kanduri, Niharika Srivastava, Hitesh Pawar, Sreekar Mannem, Satyam Shaw, Gokul Palaniandi

---

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Step 1 -- Data Acquisition & Preparation](#step-1----data-acquisition--preparation)
5. [Step 2 -- Feature Engineering & Target Definition](#step-2----feature-engineering--target-definition)
6. [Getting Started](#getting-started)
7. [Running the Pipeline](#running-the-pipeline)
8. [Running Tests](#running-tests)
9. [Configuration](#configuration)
10. [Outputs](#outputs)
11. [References](#references)

---

## Overview

Algorithmic trading models often perform well on historical data but struggle with overfitting when exposed to unseen real-world data. This project explores that challenge by comparing three modelling paradigms on an energy-market time-series:

| Approach | Purpose |
|---|---|
| **Neural Network** | Primary deep-learning model for spread prediction |
| **ARIMA** | Classical time-series baseline for comparison |
| **LLM** | Advanced paradigm to explore pattern recognition |

The pipeline is built in sequential steps:

| Step | Name | Features Added | Script |
|---|---|---|---|
| **1** | Data Acquisition & Preparation | 21 features (core + temporal + domain + regime) | `run_step1.py` |
| **2** | Feature Engineering & Target Definition | +24 features (lags + rolling stats) = **45 total** | `run_step2.py` |

---

## Project Structure

```
Algorithmic Trading/
|
|-- config.py                        # Central configuration (paths, features, hyperparameters)
|-- run_step1.py                     # Step 1 only pipeline
|-- run_step2.py                     # Step 1 + Step 2 combined pipeline
|-- requirements.txt                 # Python dependencies
|-- .gitignore                       # Git ignore rules
|
|-- src/                             # Source modules
|   |-- __init__.py                  # Package exports
|   |-- data_loader.py               # 1(a) Load and merge raw CSVs
|   |-- feature_engineering.py       # 1(b) Temporal + domain feature extraction
|   |-- sequential_validator.py      # 1(c) 15-minute interval integrity checks
|   |-- scaler.py                    # 1(d) StandardScaler normalisation
|   |-- clustering.py                # 1(e) K-Means market regime detection
|   |-- lag_features.py              # 2(b) Lagged variable generation
|   |-- rolling_features.py          # 2(c) Rolling averages and standard deviations
|   |-- target_analysis.py           # 2(a) Target variable analysis
|   +-- visualisation.py             # Plotting functions for all steps
|
|-- tests/                           # Unit tests
|   |-- __init__.py
|   |-- test_step1.py                # 24 tests for Step 1 modules
|   +-- test_step2.py                # 24 tests for Step 2 modules
|
|-- data/
|   +-- raw/                         # Raw CSVs (gitignored; download from Kaggle)
|       |-- train.csv
|       |-- imbalances.csv
|       |-- test.csv
|       +-- sample.csv
|
+-- outputs/                         # Pipeline outputs (gitignored; fully regeneratable)
    |-- artefacts/                    # Prepared data and fitted models
    |   |-- train_prepared.csv       # Scaled training set with all 45 features
    |   |-- test_prepared.csv        # Scaled test set with all 45 features
    |   |-- feature_scaler.pkl       # Fitted StandardScaler (39 features)
    |   |-- target_scaler.pkl        # Fitted StandardScaler (spread)
    |   |-- kmeans_model.pkl         # Fitted KMeans model (5 regimes)
    |   +-- feature_config.pkl       # Feature name lists for downstream steps
    |
    +-- plots/                       # Generated visualisations
        |-- data_overview.png        # Time-series of wind, solar, load, spread
        |-- spread_distribution.png  # Histogram and box plot of the target
        |-- kmeans_elbow.png         # Elbow curve for optimal k selection
        |-- kmeans_regimes.png       # Scatter plots coloured by market regime
        |-- spread_autocorrelation.png  # Autocorrelation bar chart
        |-- lag_scatter.png          # Spread vs lagged spread (15-min, 1-h, 24-h)
        |-- rolling_timeseries.png   # 2-week close-up with rolling mean/std bands
        +-- feature_correlation.png  # Full 45-feature correlation heatmap
```

---

## Dataset

The data comes from the **ENSIMAG IF 2025** Kaggle competition and contains German energy market observations at **15-minute resolution**:

| File | Rows | Columns | Period |
|---|---|---|---|
| `train.csv` | 140,157 | `date`, `wind`, `solar`, `load`, `spread` | 2020-01-01 to 2023-12-31 |
| `imbalances.csv` | 140,157 | `date`, `imbalances` | 2020-01-01 to 2023-12-31 |
| `test.csv` | 24,138 | `ID`, `date`, `wind`, `solar`, `load` | 2024-01-01 to 2024-09-08 |
| `sample.csv` | 24,138 | `ID`, `forecast` | Submission template |

**Target variable:** `spread` -- the difference between the imbalance price and the day-ahead price (EUR).

---

## Step 1 -- Data Acquisition & Preparation

### 1(a) Load & Merge

- Reads all four CSVs with datetime parsing.
- Left-joins `imbalances` into the training set on the `date` column.
- Validates zero missing values.

### 1(b) Feature Engineering

**21 features** are extracted in four groups:

| Group | Count | Features |
|---|---|---|
| **Core** | 3 | `wind`, `solar`, `load` |
| **Temporal** | 9 | Cyclical sin/cos encodings for hour, month, day-of-week, and 15-min interval; binary `is_weekend` flag |
| **Domain** | 4 | `net_load` (load - renewables), `renewable_ratio`, `wind_solar_ratio`, `total_renewable` |
| **Regime** | 5 | One-hot encoded K-Means cluster labels (`regime_0` ... `regime_4`) |

Cyclical encoding maps periodic variables onto the unit circle so the neural network sees smooth transitions (e.g. hour 23 to 0 is continuous, not a jump).

### 1(c) Sequential Structuring

- Data sorted strictly by date to maintain temporal ordering.
- Every consecutive timestamp pair is validated to be exactly 15 minutes apart.
- 8 small gaps detected in training (DST transitions + year boundaries) -- flagged but not fabricated.

### 1(d) Normalisation & Scaling

- A `StandardScaler` is fitted **on the training data only** to prevent data leakage.
- Both train and test are transformed to zero-mean, unit-variance.
- The target (`spread`) is scaled separately so predictions can be inverse-transformed back to EUR.
- Test set gracefully handles the missing `imbalances` column.

### 1(e) K-Means Market Regime Clustering

- K-Means (k=5) clusters the scaled `[wind, solar, load]` features to identify hidden market regimes.
- The elbow method is used to evaluate k=2..10; k=5 selected to capture five distinct states (e.g. low-load night, solar peak, high-wind periods).
- Regime labels are one-hot encoded, giving the neural network an explicit categorical signal about the current market state.

| Regime | Samples | Spread Mean (EUR) | Spread Std (EUR) |
|---|---|---|---|
| 0 | 21,958 (15.7%) | 2.74 | 169.76 |
| 1 | 25,028 (17.9%) | -22.75 | 252.54 |
| 2 | 42,126 (30.1%) | 4.68 | 184.03 |
| 3 | 36,105 (25.8%) | 7.40 | 270.72 |
| 4 | 14,940 (10.7%) | 11.34 | 186.27 |

---

## Step 2 -- Feature Engineering & Target Definition

### 2(a) Target Variable Definition & Analysis

The target variable `spread` (imbalance price minus day-ahead price) is formally defined and analysed:

| Statistic | Value |
|---|---|
| Mean | 0.89 EUR |
| Std | 220.88 EUR |
| Min / Max | -9,271.59 / 15,781.48 EUR |
| Median | 3.23 EUR |
| IQR | 124.60 EUR |
| Skewness | 9.88 (heavy right tail) |
| Kurtosis | 788.31 (extreme outliers) |

The spread is approximately balanced (51% positive, 49% negative) but has extreme tails, confirming the need for robust scaling and careful loss function selection.

**Autocorrelation analysis** quantifies the temporal dependencies that justify our lag feature selection:

| Lag | Interval | Autocorrelation |
|---|---|---|
| 1 | 15 min | 0.4449 |
| 4 | 1 hour | 0.2128 |
| 96 | 24 hours | 0.0716 |

The strongest autocorrelation at lag-1 confirms short-term momentum; the 24-hour lag captures the recurring daily pattern.

### 2(b) Lagged Variables

Lagged features capture **short-term momentum** by providing the neural network with historical values at three time horizons:

| Offset | Interval | Rationale |
|---|---|---|
| lag_1 | 15 minutes | Immediate momentum from the previous interval |
| lag_4 | 1 hour | Short-term trend over the past hour |
| lag_96 | 24 hours | Same time yesterday -- captures daily seasonality |

Lags are created for `wind`, `solar`, `load`, and `net_load` (12 features available at both train and test time). Additional `spread` lags (3 features) are created for training analysis but excluded from the model's feature vector since spread is not available in the test set.

NaN values at the start of the series (where lag history is unavailable) are handled via forward-fill then back-fill.

### 2(c) Rolling Averages & Standard Deviations

Rolling statistics help the model understand **volatility trends**:

| Window | Interval | Features per column |
|---|---|---|
| 4 | 1 hour | rolling mean + rolling std |
| 96 | 24 hours | rolling mean + rolling std |

Computed for `wind`, `solar`, and `load` (12 features available at test time). Additional `spread` rolling stats (4 features) are created for training only.

- **Rolling means** smooth out noise and reveal the underlying trend.
- **Rolling standard deviations** quantify recent volatility -- a spike in wind variability signals an uncertain market where the spread may swing.

Rolling computations use `min_periods=1` so even the first rows produce valid values.

### Step 2 Feature Summary

After Step 2, the neural network receives **45 features**:

| Group | Count | Description |
|---|---|---|
| Core | 3 | `wind`, `solar`, `load` |
| Temporal | 9 | Cyclical time encodings + weekend flag |
| Domain | 4 | `net_load`, `renewable_ratio`, `wind_solar_ratio`, `total_renewable` |
| Regime | 5 | One-hot K-Means cluster labels |
| **Lag** | **12** | **3 offsets x 4 columns (wind, solar, load, net_load)** |
| **Rolling** | **12** | **2 windows x 2 stats x 3 columns (wind, solar, load)** |
| **Total** | **45** | |

---

## Getting Started

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/algorithmic-trading-data-mining.git
cd algorithmic-trading-data-mining

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download the Dataset

1. Go to [ENSIMAG IF 2025 on Kaggle](https://www.kaggle.com/competitions/ensimag-if-2025/data).
2. Download `train.csv`, `imbalances.csv`, `test.csv`, and `sample.csv`.
3. Place them in `data/raw/`:
   ```
   data/raw/
   |-- train.csv
   |-- imbalances.csv
   |-- test.csv
   +-- sample.csv
   ```

---

## Running the Pipeline

```bash
# Step 1 only (data acquisition + preparation)
python run_step1.py

# Step 1 + Step 2 combined (recommended — includes all features)
python run_step2.py

# Custom data directory
python run_step2.py --data-dir /path/to/your/csvs
```

The full pipeline completes in approximately **12 seconds** and produces all outputs in the `outputs/` directory.

---

## Running Tests

```bash
# Run ALL tests (48 total)
pytest tests/ -v

# Step 1 tests only (24 tests)
pytest tests/test_step1.py -v

# Step 2 tests only (24 tests)
pytest tests/test_step2.py -v
```

The test suite covers:

| Module | Tests | What is verified |
|---|---|---|
| `feature_engineering` | 9 | Column creation, no mutation, cyclical ranges [-1, 1], formulas |
| `sequential_validator` | 4 | Sort correctness, gap detection, perfect-interval assertion |
| `scaler` | 5 | Zero-mean / unit-std, target scaling, missing-column handling |
| `clustering` | 6 | Elbow results, model fitting, regime labels, one-hot sum-to-one |
| `lag_features` | 9 | Lag values at correct offsets, NaN handling, train/test column split |
| `rolling_features` | 9 | Rolling mean/std formulas, smoothing effect, zero-NaN guarantee |
| `target_analysis` | 6 | Statistics correctness, percentile ordering, autocorrelation range |

---

## Configuration

All settings are centralised in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/raw/` | Location of raw CSV files |
| `OUTPUT_DIR` | `outputs/` | Root for all pipeline outputs |
| `OPTIMAL_K` | `5` | Number of K-Means clusters |
| `LAG_OFFSETS` | `{1, 4, 96}` | Lag intervals (15-min, 1-h, 24-h) |
| `ROLLING_WINDOWS` | `{4, 96}` | Rolling window sizes (1-h, 24-h) |
| `EXPECTED_INTERVAL_MINUTES` | `15` | Expected gap between timestamps |
| `RANDOM_STATE` | `42` | Global seed for reproducibility |

Feature lists (`CORE_FEATURES`, `TEMPORAL_FEATURES`, `DOMAIN_FEATURES`, `REGIME_FEATURES`, `LAG_FEATURES`, `ROLLING_FEATURES`, `FINAL_FEATURES`) are also defined here so every module draws from a single source of truth.

---

## Outputs

After running `run_step2.py`, the `outputs/` directory contains:

### Artefacts (`outputs/artefacts/`)

| File | Description |
|---|---|
| `train_prepared.csv` | 140,157 rows x 64 columns -- fully featured, scaled training data |
| `test_prepared.csv` | 24,138 rows x 55 columns -- fully featured, scaled test data |
| `feature_scaler.pkl` | Fitted `StandardScaler` for 39 numerical features |
| `target_scaler.pkl` | Fitted `StandardScaler` for the spread target (inverse-transform predictions back to EUR) |
| `kmeans_model.pkl` | Fitted `KMeans` model (k=5) for regime assignment on new data |
| `feature_config.pkl` | Python dictionary with all feature name lists for downstream steps |

### Plots (`outputs/plots/`)

| File | Step | Description |
|---|---|---|
| `data_overview.png` | 1 | Four-panel time-series of wind, solar, load, and spread |
| `spread_distribution.png` | 1 | Histogram and box plot of the target variable |
| `kmeans_elbow.png` | 1 | Inertia vs k with the selected k=5 marked |
| `kmeans_regimes.png` | 1 | Three-panel scatter plots coloured by market regime |
| `spread_autocorrelation.png` | 2 | Autocorrelation bar chart justifying lag selection |
| `lag_scatter.png` | 2 | Spread vs lagged spread at 15-min, 1-h, 24-h with correlation |
| `rolling_timeseries.png` | 2 | 2-week close-up of load/wind/solar with rolling mean and std bands |
| `feature_correlation.png` | 2 | Full 45-feature correlation heatmap |

---

## References

1. Chopra, D., Singh, A., J, J., & M, V. (2025). Algorithmic Trading and Machine Learning. *IITCEE*, 1--6.
2. Sanapala, S., et al. (2023). Optimising Trading Strategies using Linear Regression on Stock Prices. *RMKMATE*, 1--6.
3. Xue, M., et al. (2024). Research on Wildlife Trade Forecasting Based on Linear Regression and ARIMA Models. *ICDACAI*, 875--881.
4. Abivarshini, R., & France, K. (2025). Stock Market Price Prediction Using Deep Learning. *ICICI*, 243--247.
5. Gurung, N., et al. (2024). Algorithmic Trading Strategies: Leveraging ML Models. *JBMS*, 6(2), 132--143.
6. Xia, P., et al. (2025). Prediction of Financial Time-series Data Using ARIMA and ML. *DEAI*, 1619--1622.
7. Deep, A., et al. (2025). Risk-Adjusted Performance of Random Forest Models in HFT. *JRFM*, 18(3), 142.
8. Huang, Y., & Song, Y. (2023). Recurrent Reinforcement Learning and BiLSTM for Algorithmic Trading. *JIFS*, 45(2), 1939--1951.
9. Yilmaz, M., et al. (2024). Algorithmic Stock Trading Based on Ensemble Deep NNs. *SSRN*.
10. Bikanimine, A., et al. (2026). AI-Powered Systems for Algorithmic Trading. *LNNS*, 1584, 634--638.
11. Serban, F., & Vrinceanu, B.-P. (2025). Algorithmic Trading Bots Using AI. *SIST*, 426, 397--406.
