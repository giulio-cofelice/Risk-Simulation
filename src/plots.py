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
