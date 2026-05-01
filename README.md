# quant-frame

> A domain-agnostic quantitative machine learning framework

![CI/CD](https://img.shields.io/github/actions/workflow/status/YOUR_ORG/quant-frame/ci.yml?label=CI%2FCD&style=flat-square)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)

---

## Overview

`quant-frame` is an end-to-end quantitative machine learning pipeline purpose-built for time-series domains. It orchestrates the full research workflow—from raw data ingestion and strict temporal alignment to feature engineering, model training, walk-forward evaluation, and production-grade tearsheet generation.

A defining characteristic of the framework is its **zero-tolerance policy for data leakage**. Every transformation, scaling operation, and feature computation is executed inside a **walk-forward window**, guaranteeing that no future information (look-ahead bias) contaminates the training set. This makes `quant-frame` particularly suited for financial forecasting, algorithmic trading, and any domain where temporal causality is non-negotiable.

## Why We Built It (Architecture)

Most open-source ML toolkits treat time-series as an afterthought. Standard `fit`/`predict` patterns encourage leakage by allowing global statistics to bleed into historical folds. `quant-frame` inverts this paradigm through a **contract-first architecture** centered on the `BaseModelStrategy` interface.

The `BaseModelStrategy` contract defines a uniform API across five paradigms:

| Paradigm | Implementation | Strategy |
|---|---|---|
| **Supervised Learning** | Gradient-boosted trees via **XGBoost** | `XGBoostStrategy` |
| **Unsupervised Learning** | Gaussian Hidden Markov Models via **hmmlearn** | `GaussianHMMStrategy` |
| **Deep Reinforcement Learning** | Proximal Policy Optimization via **Stable-Baselines3** | `PPOStrategy` |
| **Meta-Learning** | Model averaging & stacking via **EnsembleStrategy** | `EnsembleStrategy` |
| **Agentic Debates** | LLM persona-driven allocation debates via **LLMStrategy** | `LLMStrategy` |

Because every strategy adheres to the same contract, the `WalkForwardEvaluator` is completely agnostic to the underlying algorithm. Swapping from a supervised classifier to an RL agent, an ensemble meta-learner, or an LLM debate panel is a one-line change—no pipeline refactoring required. The framework handles stateful scaler fitting, temporal alignment, gymnasium environment construction, ensemble weight optimization, and LLM persona orchestration transparently.

## Features

- **Strict Time-Series Alignment** – Resample, forward-fill, and gap-detect across heterogeneous frequencies without introducing future data.
- **Stateful Scalers** – `ZScoreScaler` and future transforms are fit exclusively on the in-sample portion of each walk-forward window and applied out-of-sample.
- **Leak-Proof Walk-Forward Evaluator** – Rolling or expanding window splits coupled with per-fold feature engineering eliminate look-ahead bias at every stage.
- **Domain-Agnostic RL Gym** – A custom `gymnasium` environment wraps any time-series DataFrame into a stateful MDP, enabling off-the-shelf DRL via Stable-Baselines3.
- **Mathematical Meta-Learners** – Combine heterogeneous base strategies via `EnsembleStrategy`. Supports arithmetic averaging, geometric mean, and custom stacking weights computed strictly inside each walk-forward window.
- **LLM Multi-Agent Debates** – Orchestrate allocation debates across user-defined text personas with `LLMStrategy`. Strictly domain-agnostic: the framework imposes no financial ontology; all reasoning stems from the provided persona prompts and raw feature vectors.
- **Vectorized Tearsheet Simulation** – Back-test signals against realized returns with a fully vectorized engine, then generate publication-ready tear sheets with Sharpe, Sortino, max drawdown, and cumulative P&L curves.

## Installation

### Local Installation

```bash
git clone https://github.com/YOUR_ORG/quant-frame.git
cd quant-frame
pip install -e .
```

### Developer Installation

Includes test runners, type checkers, and documentation generators:

```bash
pip install -e ".[dev]"
```

### LLM-Enabled Installation

If you plan to use the OpenAI-backed `LLMStrategy` for multi-agent debates, install with the optional `llm` extra (includes `openai` and related dependencies):

```bash
pip install -e ".[dev,llm]"
```

## Quickstart

The snippet below demonstrates the canonical workflow: ingest price data, engineer lagged features, run a leak-proof walk-forward evaluation with XGBoost, simulate returns, and plot a tearsheet.

```python
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

# 1. Ingest & align
provider = YahooFinanceProvider(ticker="AAPL", period="5y")
df = pd.DataFrame([obs.features for obs in provider.extract()])
df.index = pd.to_datetime([obs.timestamp for obs in provider.extract()])

aligner = TimeSeriesAligner()
df = aligner.resample_frequency(df, freq="D")
df = aligner.forward_fill(df)

# 2. Engineer features (all in-sample during evaluation)
df["daily_return"] = df["Close"].pct_change()
df["target_binary"] = (df["daily_return"].shift(-1) > 0).astype(int)

transformer = TimeSeriesTransformer()
df = transformer.add_lag(df, "Close", lag=1)
df = transformer.add_lag(df, "Close", lag=5)
df = transformer.add_moving_average(df, "Close", window=20)
df = df.dropna()

feature_cols = ["Close_lag_1", "Close_lag_5", "Close_ma_20"]

# 3. Walk-forward evaluation (zero leakage)
splitter = WalkForwardSplitter(train_size=500, test_size=100, window_type="rolling")
evaluator = WalkForwardEvaluator(
    strategy=XGBoostStrategy(hyperparams={"n_estimators": 100, "max_depth": 3}),
    splitter=splitter,
    transformer=transformer,
    scaler=ZScoreScaler(),
)
results = evaluator.evaluate(df, target_col="target_binary", feature_cols=feature_cols)

# 4. Simulate & analyze
results = results.join(df[["daily_return"]], how="left")
returns = VectorizedSimulator().simulate(results, signal_col="predicted", return_col="daily_return")
metrics = FinancialMetrics().calculate(pd.DataFrame({"strategy_returns": returns}),
                                        actual_col="strategy_returns", pred_col="strategy_returns")
print(metrics)

# 5. Plot tearsheet
fig = plot_financial_tearsheet(returns)
fig.savefig("tearsheet.png", dpi=150)
```

## Documentation

Full API reference and architectural deep-dives are available via MkDocs:

```bash
mkdocs serve
```

Navigate to `http://127.0.0.1:8000` to browse the documentation site.

---

**License:** MIT · **Contributions:** Pull requests and issue reports are welcome.
