#!/usr/bin/env python3
"""
run_arimax.py — ARIMAX baseline model with exogenous regressors.
=================================================================
ARIMAX(2,d,2) with exogenous features: wind, solar, load, Fourier
seasonality terms, and time features.  Demonstrates the value of
external regressors over the pure univariate ARIMA baseline.

Predictions are calibrated to P(spread > 0) via logistic regression,
matching the Kaggle submission format.

Prerequisites:
    Raw CSV files must exist in ``data/raw/``.

Usage
-----
    python run_arimax.py
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time + Fourier features to a DataFrame with DatetimeIndex."""
    out = df.copy()

    # Time features
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["month"] = df.index.month

    # Fourier terms — daily seasonality (period = 96 intervals = 24 h)
    for k in range(1, cfg.ARIMA_DAILY_HARMONICS + 1):
        out[f"sin_daily_{k}"] = np.sin(2 * np.pi * k * df.index.hour / 24)
        out[f"cos_daily_{k}"] = np.cos(2 * np.pi * k * df.index.hour / 24)

    # Fourier terms — weekly seasonality (period = 672 intervals = 7 days)
    t = (df.index.dayofweek * 96
         + df.index.hour * 4
         + df.index.minute // 15)
    for k in range(1, cfg.ARIMA_WEEKLY_HARMONICS + 1):
        out[f"sin_weekly_{k}"] = np.sin(2 * np.pi * k * t / 672)
        out[f"cos_weekly_{k}"] = np.cos(2 * np.pi * k * t / 672)

    return out


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    t0 = time.time()

    # ═══════════════════════════════════════════════════════════
    # LOAD DATA
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Loading Data")

    train = pd.read_csv(
        os.path.join(cfg.DATA_DIR, cfg.TRAIN_FILE),
        parse_dates=["date"], index_col="date",
    ).asfreq("15min")

    test = pd.read_csv(
        os.path.join(cfg.DATA_DIR, cfg.TEST_FILE),
        parse_dates=["date"], index_col="date",
    ).asfreq("15min")

    print(f"  Train: {train.index[0]} → {train.index[-1]}  ({len(train):,} rows)")
    print(f"  Test:  {test.index[0]} → {test.index[-1]}  ({len(test):,} rows)")

    # ═══════════════════════════════════════════════════════════
    # STATIONARITY CHECK
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Stationarity Check (ADF Test)")

    y = train["spread"].fillna(train["spread"].median())
    adf_stat, p_val, *_ = adfuller(y.iloc[:10000].dropna())
    d = 0 if p_val < 0.05 else 1
    order = (cfg.ARIMAX_ORDER[0], d, cfg.ARIMAX_ORDER[2])

    print(f"  ADF statistic: {adf_stat:.4f}")
    print(f"  p-value:       {p_val:.4f}")
    print(f"  {'Stationary — using d=0' if d == 0 else 'Non-stationary — using d=1'}")
    print(f"  Final ARIMAX order: {order}")

    # ═══════════════════════════════════════════════════════════
    # FEATURE ENGINEERING
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Feature Engineering")

    train = _add_features(train)
    test = _add_features(test)

    base_cols = ["wind", "solar", "load"]
    fourier_cols = [c for c in train.columns if c.startswith(("sin_", "cos_"))]
    time_cols = ["hour", "dow", "month"]
    exog_cols = base_cols + fourier_cols + time_cols

    print(f"  Exogenous features ({len(exog_cols)}): {exog_cols}")

    # Normalise using train statistics only — no leakage
    exog_mean = train[exog_cols].mean()
    exog_std = train[exog_cols].std().replace(0, 1)

    train_exog = ((train[exog_cols] - exog_mean) / exog_std
                  ).replace([np.inf, -np.inf], np.nan).fillna(0)
    test_exog = ((test[exog_cols] - exog_mean) / exog_std
                 ).replace([np.inf, -np.inf], np.nan).fillna(0)

    # ═══════════════════════════════════════════════════════════
    # WALK-FORWARD VALIDATION
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Walk-Forward Validation")

    val_steps = cfg.ARIMA_VAL_STEPS
    n_train = cfg.ARIMA_TRAIN_ROWS

    fit_end = len(y) - val_steps
    fit_start = max(0, fit_end - n_train)

    val_y = y.iloc[-val_steps:]
    val_exog = train_exog.iloc[-val_steps:]
    fit_y = y.iloc[fit_start:fit_end]
    fit_exog = train_exog.iloc[fit_start:fit_end]

    print(f"  Fitting on {len(fit_y):,} rows, validating on {len(val_y):,} rows...")

    val_model = SARIMAX(
        fit_y, exog=fit_exog, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    val_result = val_model.fit(disp=False, maxiter=200)
    val_forecast = val_result.forecast(steps=val_steps, exog=val_exog)

    actual = val_y.values
    pred = val_forecast.values
    mae = np.mean(np.abs(actual - pred))
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    dir_acc = np.mean((actual > 0) == (pred > 0))

    print(f"  Validation MAE:          {mae:.4f}")
    print(f"  Validation RMSE:         {rmse:.4f}")
    print(f"  Directional accuracy:    {dir_acc:.4f}")

    # ═══════════════════════════════════════════════════════════
    # LOGISTIC CALIBRATION
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Logistic Calibration → P(spread > 0)")

    calib = LogisticRegression()
    calib.fit(pred.reshape(-1, 1), (actual > 0).astype(int))
    val_probs = calib.predict_proba(pred.reshape(-1, 1))[:, 1]

    print(f"  Calibration coef:      {calib.coef_[0][0]:.4f}")
    print(f"  Calibration intercept: {calib.intercept_[0]:.4f}")

    # ═══════════════════════════════════════════════════════════
    # FINAL MODEL
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Fitting Final Model")

    y_fit = y.iloc[-n_train:]
    exog_fit = train_exog.iloc[-n_train:]

    print(f"  Fitting ARIMAX{order} on {len(y_fit):,} rows...")
    model = SARIMAX(
        y_fit, exog=exog_fit, order=order,
        enforce_stationarity=False, enforce_invertibility=False,
    )
    result = model.fit(disp=False, maxiter=200)
    print(result.summary())

    # ═══════════════════════════════════════════════════════════
    # TEST FORECAST
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Generating Test Forecasts")

    forecast = result.forecast(steps=len(test), exog=test_exog)
    forecast.index = test.index

    test_probs = calib.predict_proba(forecast.values.reshape(-1, 1))[:, 1]

    print(f"  Raw forecast range: [{forecast.min():.2f}, {forecast.max():.2f}]")
    print(f"  P(spread>0) range:  [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    print(f"  P(spread>0) mean:   {test_probs.mean():.4f}")

    # ═══════════════════════════════════════════════════════════
    # SAVE SUBMISSION
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Saving Artefacts")

    sub = pd.DataFrame({
        "ID": test["ID"].values if "ID" in test.columns else np.arange(len(test)),
        "forecast": test_probs,
    })
    sub_path = os.path.join(cfg.ARTEFACT_DIR, "submission_arimax.csv")
    sub.to_csv(sub_path, index=False)
    print(f"  submission_arimax.csv — {len(sub)} probability forecasts")

    # ═══════════════════════════════════════════════════════════
    # DIAGNOSTIC PLOTS
    # ═══════════════════════════════════════════════════════════
    _header("ARIMAX: Generating Diagnostic Plots")

    fig, axes = plt.subplots(5, 1, figsize=(14, 20))
    fig.suptitle("ARIMAX Diagnostics", fontsize=13)

    plot_acf(y_fit.values, lags=100, ax=axes[0], title="ACF of Spread")
    plot_pacf(y_fit.values, lags=40, ax=axes[1], title="PACF of Spread")

    axes[2].plot(result.resid[-500:], lw=0.5, color="steelblue")
    axes[2].axhline(0, color="r", lw=0.8)
    axes[2].set_title("Residuals — last 500 obs")

    axes[3].plot(y_fit.iloc[-200:].values, label="Actual", lw=1)
    axes[3].plot(result.fittedvalues[-200:].values, label="Fitted",
                 lw=1, alpha=0.7, color="darkorange")
    axes[3].set_title("In-sample fit — last 200 obs")
    axes[3].legend()

    axes[4].plot((actual > 0).astype(int), label="Actual sign", lw=0.6, alpha=0.5)
    axes[4].plot(val_probs, label="P(spread>0)", lw=0.8, color="darkorange")
    axes[4].axhline(0.5, color="r", lw=0.8, ls="--")
    axes[4].set_ylim(-0.1, 1.1)
    axes[4].set_title(f"Validation  MAE={mae:.2f}  RMSE={rmse:.2f}  dir_acc={dir_acc:.3f}")
    axes[4].legend()

    plt.tight_layout()
    p1 = os.path.join(cfg.PLOT_DIR, "arimax_diagnostics.png")
    plt.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"  Saved: {p1}")

    fig2, ax = plt.subplots(figsize=(14, 4))
    ax.plot(test_probs[:672], lw=0.8, color="darkorange")
    ax.axhline(0.5, color="r", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_title("ARIMAX Forecast P(spread > 0) — first week of test")
    plt.tight_layout()
    p2 = os.path.join(cfg.PLOT_DIR, "arimax_forecast_week1.png")
    plt.savefig(p2, dpi=120)
    plt.close(fig2)
    print(f"  Saved: {p2}")

    elapsed = time.time() - t0
    _header(f"ARIMAX COMPLETE  ({elapsed:.1f}s)")


if __name__ == "__main__":
    run_pipeline()
