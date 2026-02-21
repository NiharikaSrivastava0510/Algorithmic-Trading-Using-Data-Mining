# Algorithmic Trading Using Data Mining

**Electricity Market Spread Prediction** via Neural Networks, ARIMA, and LLMs.

A data-mining project that predicts the German electricity market spread (imbalance price minus day-ahead price) for every 15-minute interval, using the [ENSIMAG IF 2025 Algorithmic Trading](https://www.kaggle.com/competitions/ensimag-if-2025/data) competition dataset.

> **University of Southampton** -- COMP6248 Data Mining Coursework

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Step 1 -- Data Acquisition & Preparation](#step-1----data-acquisition--preparation)
5. [Getting Started](#getting-started)
6. [Running the Pipeline](#running-the-pipeline)
7. [Running Tests](#running-tests)
8. [Configuration](#configuration)
9. [Outputs](#outputs)
10. [References](#references)

---

## Overview

Algorithmic trading models often perform well on historical data but struggle with overfitting when exposed to unseen real-world data. This project explores that challenge by comparing three modelling paradigms on an energy-market time-series:

| Approach | Purpose |
|---|---|
| **Neural Network** | Primary deep-learning model for spread prediction |
| **ARIMA** | Classical time-series baseline for comparison |
| **LLM** | Advanced paradigm to explore pattern recognition |

The pipeline is built in sequential steps. This repository currently implements **Step 1: Data Acquisition & Preparation**, which transforms the raw Kaggle CSVs into a clean, feature-rich, normalised dataset ready for model training.

---

## Project Structure

```
Algorithmic Trading/
|
|-- config.py                        # Central configuration (paths, features, hyperparameters)
|-- run_step1.py                     # Main pipeline entry point (CLI)
|-- requirements.txt                 # Python dependencies
|-- .gitignore                       # Git ignore rules
|
|-- src/                             # Source modules (Step 1)
|   |-- __init__.py                  # Package exports
|   |-- data_loader.py               # 1(a) Load and merge raw CSVs
|   |-- feature_engineering.py       # 1(b) Temporal + domain feature extraction
|   |-- sequential_validator.py      # 1(c) 15-minute interval integrity checks
|   |-- scaler.py                    # 1(d) StandardScaler normalisation
|   |-- clustering.py                # 1(e) K-Means market regime detection
|   +-- visualisation.py             # Plotting functions for all sub-steps
|
|-- tests/                           # Unit tests
|   |-- __init__.py
|   +-- test_step1.py                # 24 tests covering every module
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
    |   |-- train_prepared.csv       # Scaled training set with all features
    |   |-- test_prepared.csv        # Scaled test set with all features
    |   |-- feature_scaler.pkl       # Fitted StandardScaler (features)
    |   |-- target_scaler.pkl        # Fitted StandardScaler (spread)
    |   |-- kmeans_model.pkl         # Fitted KMeans model (5 regimes)
    |   +-- feature_config.pkl       # Feature name lists for downstream steps
    |
    +-- plots/                       # Generated visualisations
        |-- data_overview.png        # Time-series of wind, solar, load, spread
        |-- spread_distribution.png  # Histogram and box plot of the target
        |-- kmeans_elbow.png         # Elbow curve for optimal k selection
        +-- kmeans_regimes.png       # Scatter plots coloured by market regime
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

**21 features** are extracted in three groups:

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
# Run with default paths (reads from data/raw/)
python run_step1.py

# Or specify a custom data directory
python run_step1.py --data-dir /path/to/your/csvs
```

The pipeline completes in approximately **8 seconds** and produces all outputs in the `outputs/` directory.

---

## Running Tests

```bash
# Run the full test suite (24 tests)
pytest tests/test_step1.py -v
```

The test suite covers:

| Module | Tests | What is verified |
|---|---|---|
| `feature_engineering` | 9 | Column creation, no mutation, cyclical ranges [-1, 1], formulas |
| `sequential_validator` | 4 | Sort correctness, gap detection, perfect-interval assertion |
| `scaler` | 5 | Zero-mean / unit-std, target scaling, missing-column handling |
| `clustering` | 6 | Elbow results, model fitting, regime labels, one-hot sum-to-one |

---

## Configuration

All settings are centralised in [`config.py`](config.py):

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/raw/` | Location of raw CSV files |
| `OUTPUT_DIR` | `outputs/` | Root for all pipeline outputs |
| `OPTIMAL_K` | `5` | Number of K-Means clusters |
| `EXPECTED_INTERVAL_MINUTES` | `15` | Expected gap between timestamps |
| `RANDOM_STATE` | `42` | Global seed for reproducibility |

Feature lists (`CORE_FEATURES`, `TEMPORAL_FEATURES`, `DOMAIN_FEATURES`, `REGIME_FEATURES`, `FINAL_FEATURES`) are also defined here so every module draws from a single source of truth.

---

## Outputs

After running `run_step1.py`, the `outputs/` directory contains:

### Artefacts (`outputs/artefacts/`)

| File | Description |
|---|---|
| `train_prepared.csv` | 140,157 rows x 33 columns -- scaled, feature-engineered training data |
| `test_prepared.csv` | 24,138 rows x 31 columns -- scaled, feature-engineered test data |
| `feature_scaler.pkl` | Fitted `StandardScaler` for 8 numerical features |
| `target_scaler.pkl` | Fitted `StandardScaler` for the spread target (inverse-transform predictions back to EUR) |
| `kmeans_model.pkl` | Fitted `KMeans` model (k=5) for regime assignment on new data |
| `feature_config.pkl` | Python dictionary with all feature name lists for downstream steps |

### Plots (`outputs/plots/`)

| File | Description |
|---|---|
| `data_overview.png` | Four-panel time-series of wind, solar, load, and spread |
| `spread_distribution.png` | Histogram and box plot of the target variable |
| `kmeans_elbow.png` | Inertia vs k with the selected k=5 marked |
| `kmeans_regimes.png` | Three-panel scatter plots coloured by market regime |

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
