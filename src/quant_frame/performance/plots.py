"""Financial tearsheet plotting utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_financial_tearsheet(returns: pd.Series) -> plt.Figure:
    """Plot a financial tearsheet with cumulative returns and drawdown curves.

    The tearsheet consists of two vertically stacked subplots:

    1. **Wealth index** – the cumulative compounded growth of ``1`` unit of
       capital over the sampled period.
    2. **Drawdown curve** – the peak-to-trough decline expressed as a negative
       percentage, filled in red.

    Args:
        returns: A :class:`pandas.Series` of daily (or periodic) strategy
            returns.  The index is typically a :class:`pandas.DatetimeIndex`
            but any ordered index is acceptable.

    Returns:
        A :class:`matplotlib.figure.Figure` containing the two subplots.
        Callers may call ``fig.savefig(...)`` or ``plt.show()`` as needed.
    """
    wealth_index: pd.Series = (1.0 + returns).cumprod()
    running_max: pd.Series = wealth_index.cummax()
    drawdown: pd.Series = (wealth_index - running_max) / running_max

    fig, (ax_cum, ax_dd) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(10, 8),
        sharex=True,
    )

    # Top subplot – cumulative returns (wealth index)
    ax_cum.plot(wealth_index.index, wealth_index, linewidth=1.5)
    ax_cum.set_title("Cumulative Returns (Wealth Index)")
    ax_cum.set_ylabel("Growth of $1")
    ax_cum.grid(True, linestyle="--", alpha=0.7)

    # Bottom subplot – drawdown filled area
    ax_dd.fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.5)
    ax_dd.set_title("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.set_ylabel("Drawdown")
    ax_dd.grid(True, linestyle="--", alpha=0.7)

    fig.tight_layout()
    return fig
