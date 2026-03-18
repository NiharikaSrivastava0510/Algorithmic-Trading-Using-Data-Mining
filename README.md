# Algorithmic Trading Using Data Mining

**Electricity Market Spread Prediction** via Neural Networks, ARIMA, and LLMs.

A data-mining project that predicts the German electricity market spread (imbalance price minus day-ahead price) for every 15-minute interval, using the [ENSIMAG IF 2025 Algorithmic Trading](https://www.kaggle.com/competitions/ensimag-if-2025/data) competition dataset.


## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dataset](#dataset)
4. [Step 1 -- Data Acquisition & Preparation](#data-acquisition--preparation)
5. [Step 2 -- Feature Engineering & Target Definition](#feature-engineering--target-definition)
6. [Step 3 -- ARIMA & SARIMA Baseline Models](#arima--sarima-baseline-models)
7. [Step 4 -- Designing the Neural Network Architecture](#designing-the-neural-network-architecture-LSTM)
8. [Step 5 -- Training & Mitigating Overfitting](#training--mitigating-overfitting)
9. [Probability Calibration](#probability-calibration)
10. [Getting Started](#getting-started)
11. [Running the Pipeline](#running-the-pipeline)
12. [Running Tests](#running-tests)
13. [Configuration](#configuration)
14. [Outputs](#outputs)
15. [References](#references)

---

## Overview

Algorithmic trading models often perform well on historical data but struggle with overfitting when exposed to unseen real-world data. This project explores that challenge by comparing three modelling paradigms on an energy-market time-series:

| Approach | Purpose |
|---|---|
| **ARIMA / SARIMA** | Classical time-series baselines for comparison |
| **Neural Network (LSTM)** | Deep-learning model for spread prediction |
| **LLM** | Advanced paradigm to explore pattern recognition |

The pipeline is built in sequential steps:

| Step | Name | Script | Description |
|---|---|---|---|
| **1** | Data Acquisition & Preparation | `run_data_acquisition.py` | 21 features (core + temporal + domain + regime) |
| **2** | Feature Engineering & Target Definition | `run_feature_engineering.py` | +24 features (lags + rolling stats) = **45 total** |
| **3** | ARIMA Baseline | `run_arima.py` | Pure univariate ARIMA(2,d,2) |
| **4** | ARIMAX Baseline | `run_arimax.py` | ARIMAX(2,d,2) with exogenous features |
| **5** | SARIMA Model | `run_sarima.py` | SARIMAX(1,d,1)x(1,1,1,96) with seasonality |
| **6** | LSTM Neural Network | `run_lstm.py` | LSTM model with 230K parameters |
| **7** | Training & Mitigating Overfitting | `run_training_overfitting.py` | Walk-forward CV, enhanced regularisation |


---

## Project Structure

```
Algorithmic Trading/
|
|-- config.py                        # Central configuration (paths, features, hyperparameters)
|-- run_data_acquisition.py          # Data Acquisition & Preparation
|-- run_feature_engineering.py       # Feature Engineering & Target Definition
|-- run_lstm.py                      # LSTM neural network training
|-- run_training_overfitting.py      # Training & Mitigating Overfitting
|-- run_arima.py                     # ARIMA(2,d,2) univariate baseline
|-- run_arimax.py                    # ARIMAX(2,d,2) with exogenous features
|-- run_sarima.py                    # SARIMAX(1,d,1)x(1,1,1,96) seasonal model
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
|   |-- dataset.py                   # 3(a) Sequence windowing and data loading
|   |-- model.py                     # 3(b) SpreadLSTM architecture definition
|   |-- trainer.py                   # 3(c) Training loop with early stopping
|   |-- walk_forward_cv.py           # 4(c) Expanding-window time-series CV
|   |-- probability.py               # Logistic calibration: spread -> P(spread > 0)
|   +-- visualisation.py             # Plotting functions for all steps
|
|-- tests/                           # Unit tests (112 total)
|   |-- __init__.py
|   |-- test_step1.py                # 24 tests for Step 1 modules
|   |-- test_step2.py                # 24 tests for Step 2 modules
|   |-- test_step3.py                # 22 tests for Step 3 modules
|   |-- test_step4.py                # 22 tests for Step 4 modules
|   +-- test_probability.py          # 20 tests for probability calibration
|
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
    |   |-- feature_config.pkl       # Feature name lists for downstream steps
    |   |-- spread_lstm.pt           # Trained LSTM weights + model config
    |   |-- spread_lstm_step4.pt     # Step 4 LSTM with enhanced regularisation
    |   |-- spread_calibrator.pkl    # Logistic calibration model (spread -> probability)
    |   |-- training_history.csv     # Per-epoch training and validation metrics
    |   |-- cv_results.csv           # Walk-forward CV per-fold metrics
    |   |-- submission.csv           # LSTM P(spread > 0) predictions (Kaggle)
    |   |-- submission_arima.csv     # ARIMA P(spread > 0) predictions
    |   |-- submission_arimax.csv    # ARIMAX P(spread > 0) predictions
    |   +-- submission_sarima.csv    # SARIMA P(spread > 0) predictions
    |
    +-- plots/                       # Generated visualisations
        |-- data_overview.png        # Time-series of wind, solar, load, spread
        |-- spread_distribution.png  # Histogram and box plot of the target
        |-- kmeans_elbow.png         # Elbow curve for optimal k selection
        |-- kmeans_regimes.png       # Scatter plots coloured by market regime
        |-- spread_autocorrelation.png  # Autocorrelation bar chart
        |-- lag_scatter.png          # Spread vs lagged spread (15-min, 1-h, 24-h)
        |-- rolling_timeseries.png   # 2-week close-up with rolling mean/std bands
        |-- feature_correlation.png  # Full 45-feature correlation heatmap
        |-- training_curves.png      # Train/val loss and LR over epochs
        |-- predictions_vs_actual_*.png  # Scatter: predicted vs actual spread
        |-- val_timeseries.png       # 7-day validation time-series overlay
        |-- cv_fold_metrics.png      # Step 4: MAE/RMSE/R² across CV folds
        |-- overfitting_analysis.png # Step 4: train vs val loss with gap
        |-- arima_diagnostics.png    # ARIMA: ACF, PACF, residuals, fit, validation
        |-- arimax_diagnostics.png   # ARIMAX: diagnostics with exogenous features
        |-- sarima_diagnostics.png   # SARIMA: diagnostics with seasonal component
        |-- *_forecast_week1.png     # First-week test forecast probability plots
```

---

## Dataset

The data comes from the **ENSIMAG IF 2025** Kaggle competition and contains German energy market observations at **15-minute resolution**:

| File | Rows | Columns | Period |
|---|---|---|---|
| `train.csv` | 140,157 | `date`, `wind`, `solar`, `load`, `spread` | 2020-01-01 to 2023-12-31 |
| `imbalances.csv` | 140,157 | `date`, `imbalances` | 2020-01-01 to 2023-12-31 |
| `test.csv` | 24,138 | `ID`, `date`, `wind`, `solar`, `load` | 2024-01-01 to 2024-09-08 |
| `sample.csv` | 24,138 | `ID`, `forecast` | Submission template (probabilities) |

**Target variable:** `spread` -- the difference between the imbalance price and the day-ahead price (EUR).

**Submission format:** `P(spread > 0)` -- the probability that the spread will be positive for each 15-minute interval.

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

## ARIMA & SARIMA Baseline Models

Three classical time-series models serve as baselines against which the LSTM is compared. They form a progression from simple to complex:

### ARIMA -- Pure Univariate Baseline

| Setting | Value |
|---|---|
| Model | ARIMA(2,d,2) |
| Exogenous features | None |
| Seasonal component | None |
| Training window | ~1 year (35,040 rows) |
| Script | `run_arima.py` |

The simplest possible baseline: forecasts the spread using only its own past values (2 AR terms) and past forecast errors (2 MA terms). The differencing order `d` is determined automatically via the ADF stationarity test.

### ARIMAX -- With Exogenous Regressors

| Setting | Value |
|---|---|
| Model | ARIMAX(2,d,2) |
| Exogenous features | wind, solar, load + Fourier (daily/weekly) + time features |
| Seasonal component | None (seasonality captured by Fourier terms) |
| Training window | ~1 year (35,040 rows) |
| Script | `run_arimax.py` |

Extends ARIMA by incorporating external market signals. Fourier terms (3 daily + 2 weekly harmonics) capture periodic patterns without the computational cost of seasonal differencing.

### SARIMA -- Seasonal Model

| Setting | Value |
|---|---|
| Model | SARIMAX(1,d,1)x(1,1,1,96) |
| Exogenous features | wind, solar, load + Fourier + time + domain features (20 total) |
| Seasonal component | m=96 (one full day of 15-min intervals) |
| Training window | ~1 year (35,040 rows) |
| Script | `run_sarima.py` |

The most expressive ARIMA-family model. Adds seasonal differencing with period m=96 to explicitly capture daily periodicity, plus domain features (`net_load`, `renewable_ratio`, `total_renewable`) matching the LSTM pipeline.


##  Designing the Neural Network Architecture

###  LSTM Network for Sequential Time-Series

An **LSTM (Long Short-Term Memory)** network is used to capture the temporal dependencies in electricity market data. LSTMs are specifically designed for sequential data where long-range patterns matter -- their gating mechanism selectively remembers or forgets information across the 96-step (24-hour) input window.

**Why LSTM?** The electricity spread exhibits autocorrelation at multiple time horizons (15-min, 1-hour, 24-hour). Standard feedforward networks would treat each timestep independently, losing this temporal structure. The LSTM's recurrent architecture naturally processes the ordered sequence of 15-minute intervals.

**Sequence windowing:** The prepared data is segmented into sliding windows of 96 timesteps (one full day). For each window, the model predicts the spread at the final timestep. This gives the network a full 24-hour context to learn intra-day patterns (morning ramps, solar peaks, evening demand).

###  Input Layer Mapping

The input layer maps the 45-feature vector from Step 2 into the LSTM encoder:

```
Input shape:  (batch, 96, 45)
              |       |    |
              |       |    +-- 45 features per interval
              |       +------- 96 intervals (24 hours)
              +--------------- batch of samples
```

A 2-layer stacked LSTM processes the sequence, and only the final hidden state (capturing the full 24-hour context) is passed to the dense head.

###  Dense Output Layer

The output is a **single node with linear activation**, appropriate for predicting a continuous value (the spread in EUR). No sigmoid or ReLU is applied to the final layer, allowing the model to predict both positive and negative spread values. Raw EUR predictions are then converted to P(spread > 0) via [logistic calibration](#probability-calibration) for Kaggle submission.

### Architecture Summary

```
SpreadLSTM(
  Input Dropout:   0.1 (Step 4: randomly zeroes features)

  LSTM Encoder:
    Input:       45 features
    Hidden:      128 units x 2 layers
    Dropout:     0.2 (between LSTM layers)

  Dense Head:
    Linear:      128 -> 64
    BatchNorm1d: 64 (Step 4: stabilises activations)
    ReLU
    Dropout:     0.3
    Linear:      64  -> 1   (linear activation)

  Total parameters: 230,017
)
```

### Anti-Overfitting Measures

Eight complementary regularisation strategies prevent the model from memorising training noise:

| Technique | Setting | Purpose |
|---|---|---|
| **Input dropout** | 0.1 | Randomly zeroes features before the LSTM (Step 4) |
| **LSTM inter-layer dropout** | 0.2 | Regularises recurrent representations between stacked layers |
| **Batch normalisation** | BatchNorm1d(64) | Stabilises activations in the dense head (Step 4) |
| **Dense head dropout** | 0.3 | Prevents co-adaptation in the prediction head |
| **L2 regularisation** | weight_decay = 1e-5 | Penalises large weights via AdamW optimiser |
| **Early stopping** | patience = 10, min_delta = 1e-4 | Halts training when validation loss stops improving |
| **Gradient clipping** | max_norm = 1.0 | Guards against exploding gradients in LSTM backprop |
| **LR scheduling** | ReduceLROnPlateau (factor=0.5) | Halves learning rate when validation loss plateaus |

### Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | AdamW |
| Learning rate | 1e-3 |
| Batch size | 256 |
| Max epochs | 100 |
| Loss function | MSE (Mean Squared Error) |
| Train/Val split | 80% / 20% (chronological) |
| Sequence length | 96 (24 hours) |
| Device | CUDA > MPS > CPU (auto-detected) |

### Training Results

The model trains with early stopping and restores the best checkpoint:

| Metric | Value |
|---|---|
| Best epoch | 2 / 12 |
| Early stopped | Yes (patience=10) |
| Validation MAE | 87.76 EUR |
| Validation RMSE | 259.23 EUR |
| Validation R² | 0.0085 |
| Training time | ~218 seconds (on Apple MPS) |

The low R² score is expected for this initial architecture given the extreme characteristics of the spread distribution (kurtosis = 788, heavy tails with outliers up to +/-15,000 EUR). The model successfully captures the central tendency but struggles with extreme events. Further improvements (attention mechanisms, ensemble methods, feature selection) would be explored in subsequent iterations.

---

##  Training & Mitigating Overfitting

###  Strict Regularisation

This Step  enhances the base LSTM architecture with two additional regularisation techniques:

- **Input dropout (0.1)** -- randomly zeroes input features before the LSTM, forcing the network to learn robust representations that don't rely on any single feature.
- **Batch normalisation** -- `BatchNorm1d(64)` in the dense head stabilises activations and accelerates convergence.

###  Enhanced Early Stopping

The early stopping criterion is refined with a **minimum delta threshold** (`min_delta = 1e-4`). Validation loss must improve by at least this amount to reset the patience counter, preventing noise-driven resets that would allow the model to continue training past the true optimum.

Each epoch also tracks the **overfitting gap** (`val_loss - train_loss`), providing a quantitative measure of generalisation health that is visualised in the diagnostic plots.

###  Walk-Forward Cross-Validation

A strict **expanding-window time-series CV** ensures that future data never leaks into training:

- **4 folds**, each with a 6-month validation window
- The training set expands forward in time; validation is always the next chronological block
- Fold boundaries are computed sequentially (cursor-based) to guarantee non-overlapping validation periods
- Each fold trains a fresh model with early stopping and reports MAE, RMSE, R², and overfitting gap

This directly tests whether the model generalises across different market regimes and time periods, rather than a single arbitrary train/val split.

---


### Common Design Across All Three

- **ADF stationarity test** determines differencing order `d`
- **Walk-forward validation** on the last 7 days (672 timesteps)
- **Logistic calibration** converts raw spread forecasts to P(spread > 0)
- **Train-only normalisation** prevents data leakage into test
- **Diagnostic plots**: ACF, PACF, residuals, in-sample fit, validation probabilities

---

## Probability Calibration

The Kaggle competition expects **P(spread > 0)** -- probability forecasts between 0 and 1 -- not raw EUR spread values. All models (LSTM, ARIMA, ARIMAX, SARIMA) use the same calibration approach:

1. **Train the model** to predict raw spread values (regression)
2. **Fit a `LogisticRegression`** on validation data: `predicted_spread_eur` -> `actual_sign`
3. **Apply the calibrator** to test predictions to produce `P(spread > 0)`

This is implemented in `src/probability.py` via the `SpreadCalibrator` class, which also reports directional accuracy and stores calibration coefficients. A model-free `spread_to_proba_simple()` sigmoid fallback is also available.

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
# Step 1: Data Acquisition & Preparation
python run_data_acquisition.py

# Step 2: Feature Engineering & Target Definition
python run_feature_engineering.py

# Step 3: LSTM Training (requires Step 2 outputs)
python run_lstm.py

# Step 4: Training & Mitigating Overfitting (requires Step 2 outputs)
python run_training_overfitting.py

# ARIMA/SARIMA Baselines (require raw CSVs only)
python run_arima.py        # ~1 min
python run_arimax.py       # ~2 min
python run_sarima.py       # ~5-10 min (m=96 seasonal is slow)

# Custom data directory
python run_feature_engineering.py --data-dir /path/to/your/csvs
```

Steps 1-2 complete in approximately **12 seconds**. LSTM training takes **~3.5 minutes** on Apple MPS (longer on CPU-only machines). All outputs are written to the `outputs/` directory.

---

## Running Tests

```bash
# Run ALL tests (112 total)
pytest tests/ -v

# Step 1 tests only (24 tests)
pytest tests/test_step1.py -v

# Step 2 tests only (24 tests)
pytest tests/test_step2.py -v

# Step 3 tests only (22 tests)
pytest tests/test_step3.py -v

# Step 4 tests only (22 tests)
pytest tests/test_step4.py -v

# Probability calibration tests (20 tests)
pytest tests/test_probability.py -v
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
| `dataset` | 6 | Sequence shapes, target alignment, window slicing, edge cases |
| `split` | 4 | Chronological ordering, no overlap, fraction correctness |
| `model` | 7 | Output shape, float dtype, linear activation, parameter count, device |
| `trainer` | 5 | Training loop, loss tracking, prediction shapes, early stopping |
| `enhanced_model` | 8 | Input dropout, batch norm, backward compatibility |
| `enhanced_trainer` | 6 | Min delta, overfit gap tracking, quiet mode |
| `walk_forward_cv` | 6 | Fold generation, expanding window, no leakage, non-overlapping |
| `probability` | 20 | Calibrator lifecycle, output range, monotonicity, sigmoid fallback |

---

## Configuration

All settings are centralised in [`config.py`](config.py):

### LSTM Settings

| Setting | Default | Description |
|---|---|---|
| `DATA_DIR` | `data/raw/` | Location of raw CSV files |
| `OUTPUT_DIR` | `outputs/` | Root for all pipeline outputs |
| `OPTIMAL_K` | `5` | Number of K-Means clusters |
| `LAG_OFFSETS` | `{1, 4, 96}` | Lag intervals (15-min, 1-h, 24-h) |
| `ROLLING_WINDOWS` | `{4, 96}` | Rolling window sizes (1-h, 24-h) |
| `SEQUENCE_LENGTH` | `96` | LSTM input window (24 hours) |
| `LSTM_HIDDEN_SIZE` | `128` | LSTM hidden state dimension |
| `LSTM_NUM_LAYERS` | `2` | Number of stacked LSTM layers |
| `INPUT_DROPOUT` | `0.1` | Feature dropout before LSTM |
| `BATCH_SIZE` | `256` | Training batch size |
| `LEARNING_RATE` | `1e-3` | Initial AdamW learning rate |
| `MAX_EPOCHS` | `100` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `10` | Epochs without improvement before stopping |
| `EARLY_STOPPING_MIN_DELTA` | `1e-4` | Minimum improvement to reset patience |
| `VALIDATION_FRACTION` | `0.2` | Chronological train/val split ratio |
| `CV_N_FOLDS` | `4` | Walk-forward cross-validation folds |
| `CV_VAL_MONTHS` | `6` | Months per CV validation window |
| `RANDOM_STATE` | `42` | Global seed for reproducibility |

### ARIMA / SARIMA Settings

| Setting | Default | Description |
|---|---|---|
| `ARIMA_TRAIN_ROWS` | `35040` | Training window (~1 year of 15-min data) |
| `ARIMA_VAL_DAYS` | `7` | Walk-forward validation hold-out days |
| `ARIMA_ORDER` | `(2, 0, 2)` | ARIMA(p,d,q) order |
| `ARIMAX_ORDER` | `(2, 0, 2)` | ARIMAX(p,d,q) order |
| `SARIMA_ORDER` | `(1, 0, 1)` | SARIMA non-seasonal order |
| `SARIMA_SEASONAL_ORDER` | `(1, 1, 1, 96)` | SARIMA seasonal (P,D,Q,m) |
| `ARIMA_DAILY_HARMONICS` | `3` | Fourier sin/cos pairs for 24-h cycle |
| `ARIMA_WEEKLY_HARMONICS` | `2` | Fourier sin/cos pairs for 7-day cycle |

Feature lists (`CORE_FEATURES`, `TEMPORAL_FEATURES`, `DOMAIN_FEATURES`, `REGIME_FEATURES`, `LAG_FEATURES`, `ROLLING_FEATURES`, `FINAL_FEATURES`) are also defined here so every module draws from a single source of truth.

---

## Outputs

After running the full pipeline, the `outputs/` directory contains:

### Artefacts (`outputs/artefacts/`)

| File | Source | Description |
|---|---|---|
| `train_prepared.csv` | Step 2 | 140,157 rows x 64 columns -- fully featured, scaled training data |
| `test_prepared.csv` | Step 2 | 24,138 rows x 55 columns -- fully featured, scaled test data |
| `feature_scaler.pkl` | Step 2 | Fitted `StandardScaler` for 39 numerical features |
| `target_scaler.pkl` | Step 2 | Fitted `StandardScaler` for the spread target |
| `kmeans_model.pkl` | Step 1 | Fitted `KMeans` model (k=5) for regime assignment |
| `feature_config.pkl` | Step 2 | Feature name lists for downstream steps |
| `spread_lstm.pt` | Step 3 | Trained LSTM model weights, config, and validation metrics |
| `spread_lstm_step4.pt` | Step 4 | LSTM with enhanced regularisation + CV metrics |
| `spread_calibrator.pkl` | Step 3 | Logistic calibration model (spread -> probability) |
| `training_history.csv` | Step 3/4 | Per-epoch train loss, val loss, and learning rate |
| `cv_results.csv` | Step 4 | Walk-forward CV per-fold metrics |
| `submission.csv` | LSTM | P(spread > 0) predictions for Kaggle submission |
| `submission_arima.csv` | ARIMA | ARIMA baseline probability predictions |
| `submission_arimax.csv` | ARIMAX | ARIMAX baseline probability predictions |
| `submission_sarima.csv` | SARIMA | SARIMA baseline probability predictions |

### Plots (`outputs/plots/`)

| File | Source | Description |
|---|---|---|
| `data_overview.png` | Step 1 | Four-panel time-series of wind, solar, load, and spread |
| `spread_distribution.png` | Step 1 | Histogram and box plot of the target variable |
| `kmeans_elbow.png` | Step 1 | Inertia vs k with the selected k=5 marked |
| `kmeans_regimes.png` | Step 1 | Three-panel scatter plots coloured by market regime |
| `spread_autocorrelation.png` | Step 2 | Autocorrelation bar chart justifying lag selection |
| `lag_scatter.png` | Step 2 | Spread vs lagged spread at 15-min, 1-h, 24-h with correlation |
| `rolling_timeseries.png` | Step 2 | 2-week close-up of load/wind/solar with rolling mean and std bands |
| `feature_correlation.png` | Step 2 | Full 45-feature correlation heatmap |
| `training_curves.png` | Step 3/4 | Train/val loss and learning rate schedule over epochs |
| `predictions_vs_actual_*.png` | Step 3/4 | Scatter plot of predicted vs actual spread (EUR) |
| `val_timeseries.png` | Step 3/4 | 7-day overlay of actual and predicted spread on validation set |
| `cv_fold_metrics.png` | Step 4 | Bar chart of MAE/RMSE/R² across CV folds |
| `overfitting_analysis.png` | Step 4 | Train vs val loss with overfitting gap shaded |
| `arima_diagnostics.png` | ARIMA | ACF, PACF, residuals, in-sample fit, validation probs |
| `arimax_diagnostics.png` | ARIMAX | Diagnostics with exogenous features |
| `sarima_diagnostics.png` | SARIMA | Diagnostics with seasonal component |
| `*_forecast_week1.png` | ARIMA/ARIMAX/SARIMA | First-week test forecast probability plots |

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
