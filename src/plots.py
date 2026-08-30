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
