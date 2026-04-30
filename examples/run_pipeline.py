#!/usr/bin/env python3
"""End-to-end quant_frame pipeline demonstration.

This script orchestrates the full quant workflow:

1. Data ingestion via YahooFinanceProvider
2. Time-series alignment (resample + forward-fill)
3. Feature engineering (daily returns, binary target, continuous target, lags, moving averages)
4. Walk-forward evaluation with XGBoost
5. Walk-forward evaluation with PPO Agent
6. Vectorized back-test simulation for both
7. Financial metrics computation for both
8. Tearsheet plotting for both
"""

from __future__ import annotations

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.join(_SCRIPT_DIR, "..", "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # non-interactive backend for headless environments  # noqa: E402

import pandas as pd  # noqa: E402

from quant_frame import (  # noqa: E402
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
from quant_frame.strategies import PPOStrategy  # noqa: E402


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
    print("\n[1/8] Data Ingestion")
    provider = YahooFinanceProvider(ticker="AAPL", period="5y")
    observations = provider.extract()
    print(f"    Fetched {len(observations):,} observations for {provider.ticker}.")
    df = observations_to_dataframe(observations)
    print(f"    Columns: {list(df.columns)}")

    # 2. ALIGNMENT
    print("\n[2/8] Time-Series Alignment")
    aligner = TimeSeriesAligner()
    df = aligner.resample_frequency(df, freq="D")
    df = aligner.forward_fill(df)
    print(f"    After alignment: {len(df):,} rows.")

    # 3. FEATURE ENGINEERING
    print("\n[3/8] Feature Engineering")
    df["daily_return"] = df["Close"].pct_change()
    df["target_binary"] = np.where(df["daily_return"].shift(-1) > 0, 1, -1)
    df["target_return"] = df["daily_return"].shift(-1)
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

    # Shared infrastructure
    splitter = WalkForwardSplitter(
        train_size=500,
        test_size=100,
        window_type="rolling",
    )
    scaler = ZScoreScaler()
    shared_transformer = TimeSeriesTransformer()

    # 4. XGBOOST EVALUATION
    print("\n[4/8] Walk-Forward Evaluation – XGBoost")
    xgb_strategy = XGBoostStrategy(
        hyperparams={
            "n_estimators": 100,
            "max_depth": 3,
            "learning_rate": 0.1,
            "verbosity": 0,
        }
    )
    xgb_evaluator = WalkForwardEvaluator(
        strategy=xgb_strategy,
        splitter=splitter,
        transformer=shared_transformer,
        scaler=scaler,
    )
    xgb_results = xgb_evaluator.evaluate(
        df, target_col="target_binary", feature_cols=feature_cols
    )
    print(f"    XGBoost out-of-sample predictions: {len(xgb_results):,} rows.")

    # 5. XGBOOST SIMULATION & METRICS
    print("\n[5/8] Vectorized Back-test Simulation – XGBoost")
    xgb_results = xgb_results.join(df[["daily_return"]], how="left")
    simulator = VectorizedSimulator()
    xgb_strategy_returns = simulator.simulate(
        xgb_results,
        signal_col="predicted",
        return_col="daily_return",
    )
    print(f"    Simulated {len(xgb_strategy_returns):,} daily XGBoost strategy returns.")

    print("\n    XGBoost Financial Metrics:")
    xgb_metrics_df = pd.DataFrame({"strategy_returns": xgb_strategy_returns})
    xgb_metrics = FinancialMetrics().calculate(
        xgb_metrics_df,
        actual_col="strategy_returns",
        pred_col="strategy_returns",
    )
    for key, value in xgb_metrics.items():
        print(f"        {key:20s}: {value:+.4f}")

    fig_xgb = plot_financial_tearsheet(xgb_strategy_returns)
    xgb_plot_path = os.path.join(_SCRIPT_DIR, "tearsheet_xgb.png")
    fig_xgb.savefig(xgb_plot_path, dpi=150)
    print(f"    Saved XGBoost tearsheet to: {xgb_plot_path}")

    # 6. PPO EVALUATION
    print("\n[6/8] Walk-Forward Evaluation – PPO Agent")
    ppo_strategy = PPOStrategy(total_timesteps=2000)
    ppo_evaluator = WalkForwardEvaluator(
        strategy=ppo_strategy,
        splitter=splitter,
        transformer=shared_transformer,
        scaler=scaler,
    )
    ppo_results = ppo_evaluator.evaluate(
        df, target_col="target_return", feature_cols=feature_cols
    )
    print(f"    PPO out-of-sample predictions: {len(ppo_results):,} rows.")

    # 7. PPO SIMULATION & METRICS
    print("\n[7/8] Vectorized Back-test Simulation – PPO Agent")
    ppo_results = ppo_results.join(df[["daily_return"]], how="left")
    ppo_strategy_returns = simulator.simulate(
        ppo_results,
        signal_col="predicted",
        return_col="daily_return",
    )
    print(f"    Simulated {len(ppo_strategy_returns):,} daily PPO strategy returns.")

    print("\n    PPO Financial Metrics:")
    ppo_metrics_df = pd.DataFrame({"strategy_returns": ppo_strategy_returns})
    ppo_metrics = FinancialMetrics().calculate(
        ppo_metrics_df,
        actual_col="strategy_returns",
        pred_col="strategy_returns",
    )
    for key, value in ppo_metrics.items():
        print(f"        {key:20s}: {value:+.4f}")

    fig_ppo = plot_financial_tearsheet(ppo_strategy_returns)
    ppo_plot_path = os.path.join(_SCRIPT_DIR, "tearsheet_ppo.png")
    fig_ppo.savefig(ppo_plot_path, dpi=150)
    print(f"    Saved PPO tearsheet to: {ppo_plot_path}")

    print("\n" + "=" * 60)
    print(" Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
