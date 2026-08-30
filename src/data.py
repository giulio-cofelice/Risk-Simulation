"""
Data ingestion for the risk-model evaluation project.

This is the only module that touches disk or the network. Every downstream
module (var_models, expected_shortfall, backtests, plots) works on an
already-clean DataFrame produced here -- so this file is the single place to
audit for lookahead bias or silent data corruption.
"""

import numpy as np
import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "JPM", "XOM", "JNJ", "PG"]
DEFAULT_WEIGHTS = [0.2, 0.2, 0.2, 0.2, 0.2]


def fetch_prices(tickers=DEFAULT_TICKERS, start="2011-01-01", end="2026-07-13"):
    """
    Download daily split/dividend-adjusted close prices and align them onto a
    single trading calendar.

    Two explicit messiness decisions:
    - auto_adjust=True asks yfinance for adjusted close, so stock splits and
      dividend payouts don't show up as fake one-day price jumps/crashes that
      would corrupt the return series and inflate volatility estimates.
    - Any date on which ANY ticker is missing a price is dropped entirely,
      rather than forward-filled. Forward-filling would insert synthetic
      zero-return days, which artificially deflates the volatility estimates
      VaR depends on -- a worse failure mode for a risk model than losing a
      handful of rows.
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]

    if isinstance(prices, pd.Series):  # yfinance returns a Series for a single ticker
        prices = prices.to_frame(tickers[0])

    prices = prices.dropna(how="any").sort_index()
    return prices


def compute_returns(prices):
    """Daily simple (arithmetic) returns per asset."""
    return prices.pct_change().dropna(how="any")


def compute_portfolio_returns(asset_returns, weights=DEFAULT_WEIGHTS):
    """
    Collapse per-asset returns into a single fixed-weight portfolio return
    series: r_p,t = sum_i w_i * r_i,t.

    Simple returns are used deliberately, not log returns. Portfolio return is
    only exactly equal to the weighted sum of constituent returns under simple
    returns -- log returns are additive over TIME for one asset but not
    additive CROSS-SECTIONALLY across assets, so using them here would make
    the portfolio series an approximation rather than an identity.

    This also assumes the portfolio is rebalanced back to fixed weights every
    day (w constant, not drifting with price moves) -- the standard
    simplifying assumption in the VaR backtesting literature, and one worth
    being able to state explicitly if asked.
    """
    weights = np.asarray(weights, dtype=float)
    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")
    if len(weights) != asset_returns.shape[1]:
        raise ValueError("Number of weights must match number of assets.")

    portfolio_returns = asset_returns @ weights
    portfolio_returns.name = "portfolio"
    return portfolio_returns


def train_test_split_returns(returns, train_fraction=0.7):
    """
    Split a return series in TIME order into a training window and a
    hold-out test window -- no shuffling, ever. This is the boundary that
    keeps the whole project honest: VaR/ES get estimated only on the train
    slice, and M4's backtests check those frozen estimates against test-slice
    days the model never saw. Shuffling here would leak future information
    into estimation (lookahead bias) and silently invalidate every backtest
    built on top of it.

    Deliberately a single static split, not a rolling/walk-forward one: the
    backtest is asking "if this model were frozen at one point in time, how
    long does it stay valid," which is also the honest boundary given this
    project doesn't use time-varying volatility methods (EWMA/GARCH) to
    justify re-estimating partway through.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")

    split_idx = int(len(returns) * train_fraction)
    train_returns = returns.iloc[:split_idx]
    test_returns = returns.iloc[split_idx:]
    return train_returns, test_returns
