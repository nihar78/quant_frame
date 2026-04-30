#!/usr/bin/env python3
"""End-to-end quant_frame pipeline demonstration.

This script orchestrates the full quant workflow:

1. Data ingestion via YahooFinanceProvider
2. Time-series alignment (resample + forward-fill)
3. Feature engineering (daily returns, binary target, lags, moving averages)
4. Walk-forward evaluation with XGBoost
5. Vectorized back-test simulation
6. Financial metrics computation
7. Tearsheet plotting
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments

import pandas as pd

from quant_frame import (
    YahooFinanceProvider,
    TimeSeriesAligner,
    TimeSeriesTransformer,
    ZScoreScaler,
    XGBoostStrategy,
    WalkForwardSplitter,
    WalkForwardEvaluator,
    VectorizedSimulator,
    FinancialMetrics,
    plot_financial_tearsheet,
)


def observations_to_dataframe(observations: list) -> pd.DataFrame:
    """Convert a list of TimeSeriesObservation objects to a DataFrame."""
    rows = []
    index = []
    for obs in observations:
        rows.append(obs.features)
        index.append(obs.timestamp)
    df = pd.DataFrame(rows, index=index)
    df.index.name = "Date"
    return df


def main() -> None:
    print("=" * 60)
    print(" quant_frame End-to-End Pipeline Demo ")
    print("=" * 60)

    # 1. INGESTION
    print("\n[1/7] Data Ingestion")
    provider = YahooFinanceProvider(ticker="AAPL", period="5y")
    observations = provider.extract()
    print(f"    Fetched {len(observations):,} observations for {provider.ticker}.")
    df = observations_to_dataframe(observations)
    print(f"    Columns: {list(df.columns)}")

    # 2. ALIGNMENT
    print("\n[2/7] Time-Series Alignment")
    aligner = TimeSeriesAligner()
    df = aligner.resample_frequency(df, freq="D")
    df = aligner.forward_fill(df)
    print(f"    After alignment: {len(df):,} rows.")

    # 3. FEATURE ENGINEERING
    print("\n[3/7] Feature Engineering")
    df["daily_return"] = df["Close"].pct_change()
    df["target"] = (df["daily_return"].shift(-1) > 0).astype(int)
    df = df.dropna()
    print(f"    After base features + dropna: {len(df):,} rows.")

    transformer = TimeSeriesTransformer()
    df = transformer.add_lag(df, column="Close", lag=1)
    df = transformer.add_lag(df, column="Close", lag=3)
    df = transformer.add_lag(df, column="Close", lag=5)
    df = transformer.add_moving_average(df, column="Close", window=5)
    df = transformer.add_moving_average(df, column="Close", window=20)
    df = df.dropna()
    print(f"    After lags/MAs + dropna: {len(df):,} rows.")

    feature_cols = [
        "Close_lag_1",
        "Close_lag_3",
        "Close_lag_5",
        "Close_ma_5",
        "Close_ma_20",
    ]

    # 4. EVALUATION
    print("\n[4/7] Walk-Forward Evaluation")
    strategy = XGBoostStrategy(
        hyperparams={
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "verbosity": 0,
        }
    )
    splitter = WalkForwardSplitter(
        train_size=500,
        test_size=20,
        window_type="rolling",
    )
    scaler = ZScoreScaler()
    evaluator = WalkForwardEvaluator(
        strategy=strategy,
        splitter=splitter,
        transformer=TimeSeriesTransformer(),
        scaler=scaler,
    )
    results_df = evaluator.evaluate(df, target_col="target", feature_cols=feature_cols)
    print(f"    Out-of-sample predictions: {len(results_df):,} rows.")

    # 5. SIMULATION
    print("\n[5/7] Vectorized Back-test Simulation")
    results_df = results_df.join(df[["daily_return"]], how="left")
    simulator = VectorizedSimulator()
    strategy_returns = simulator.simulate(
        results_df,
        signal_col="predicted",
        return_col="daily_return",
    )
    print(f"    Simulated {len(strategy_returns):,} daily strategy returns.")

    # 6. METRICS
    print("\n[6/7] Financial Metrics")
    metrics_df = pd.DataFrame({"strategy_returns": strategy_returns})
    metrics = FinancialMetrics().calculate(
        metrics_df,
        actual_col="strategy_returns",
        pred_col="strategy_returns",
    )
    for key, value in metrics.items():
        print(f"    {key:20s}: {value:+.4f}")

    # 7. PLOTTING
    print("\n[7/7] Tearsheet Plotting")
    fig = plot_financial_tearsheet(strategy_returns)
    output_path = os.path.join(_SCRIPT_DIR, "tearsheet.png")
    fig.savefig(output_path, dpi=150)
    print(f"    Saved tearsheet to: {output_path}")

    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
