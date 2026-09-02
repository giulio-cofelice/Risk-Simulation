"""Visualization, deliberately isolated from analysis logic in the other modules."""

import matplotlib.pyplot as plt


def plot_cumulative_growth(asset_returns, portfolio_returns, save_path=None):
    """
    Plot cumulative growth of $1 invested in each asset and in the
    fixed-weight portfolio.

    This is a data sanity check as much as a result: no gaps, no impossible
    outliers from unadjusted splits, and the portfolio line should sit
    sensibly among its constituents rather than off on its own.
    """
    cumulative = (1 + asset_returns).cumprod()
    cumulative["portfolio"] = (1 + portfolio_returns).cumprod()

    fig, ax = plt.subplots(figsize=(10, 6))
    cumulative.plot(ax=ax, linewidth=1.2)
    ax.set_title("Cumulative Growth of $1")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_var_backtest(test_returns, var_threshold, violations, method_name, confidence_label, save_path=None):
    """
    The visual form of M4's backtest: actual portfolio returns over the
    TEST window, with the (fixed, train-estimated) VaR threshold drawn as a
    horizontal line, and every violation -- a day the real loss broke that
    promise -- marked as a red dot.

    Takes test_returns, var_threshold, and violations all already computed
    (by data.train_test_split_returns, a fitted VaR model, and
    backtests.count_violations respectively) -- this function only draws
    them, per plots.py's standing rule.

    What to look for: whether the red dots are scattered randomly across
    the window (independence, what Christoffersen's test checks) or bunched
    into one stretch (a model that fails exactly when it matters), and
    whether there are roughly the right NUMBER of them for the promised
    confidence level (what Kupiec's test checks) -- both visible in one
    picture instead of two separate numbers.
    """
    violations = violations.astype(bool)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_returns.index, test_returns.values, linewidth=0.7, color="#4C72B0", label="Daily portfolio return")
    ax.axhline(
        -var_threshold,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"{method_name} VaR ({confidence_label}) = {var_threshold:.2%}",
    )
    ax.scatter(
        test_returns.index[violations],
        test_returns.values[violations],
        color="crimson",
        zorder=5,
        s=30,
        label=f"Violations ({int(violations.sum())})",
    )
    ax.set_title(f"{method_name} VaR Backtest ({confidence_label}) -- Test Period")
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily portfolio return")
    ax.legend(loc="lower left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_monte_carlo_outcomes(simulated_returns, var_99, save_path=None):
    """
    Histogram of MonteCarloVaR's simulated draws -- literally the "possible
    outcomes" the simulation generates for tomorrow's portfolio return --
    with its 99% VaR cutoff marked as a vertical line.

    The bell shape isn't a finding, it's a reminder: these draws come from
    a Normal(mu, sigma) by construction, so the histogram is exactly as
    thin-tailed as the Gaussian assumption itself. Comparing this to the
    real (fatter-tailed) return distribution is what the M2/M3 sanity
    checks already did numerically -- this plot shows what "simulating from
    a Normal" actually looks like, not a comparison in itself.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(simulated_returns, bins=100, color="#4C72B0", edgecolor="none")
    ax.axvline(
        -var_99,
        color="crimson",
        linestyle="--",
        linewidth=1.5,
        label=f"Monte Carlo VaR (99%) = {var_99:.2%}",
    )
    ax.set_title("Monte Carlo Simulated Outcomes")
    ax.set_xlabel("Simulated daily portfolio return")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
