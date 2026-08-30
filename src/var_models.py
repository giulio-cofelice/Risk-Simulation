"""
Value-at-Risk models: three estimation methods behind one shared interface.

Every subclass follows fit(returns) -> calculate(alpha):
  - fit() is called ONCE on TRAINING returns only. It estimates and stores
    whatever the method needs (a sample, or a mean/sigma, or simulated
    draws). It knows nothing about a train/test split -- that split happens
    outside this file (in data.py / the notebook), so the lookahead-bias
    boundary stays in one place.
  - calculate(alpha) can be called any number of times afterwards, at any
    confidence level, without re-estimating. alpha is the left-tail
    probability: alpha=0.01 -> 99% VaR, alpha=0.05 -> 95% VaR.

SIGN CONVENTION: calculate() always returns VaR as a POSITIVE loss
magnitude. A daily return distribution's alpha-quantile is itself a
negative number (a loss); every method below flips its sign before
returning, so "VaR_99% = 0.023" reads as "we expect to lose more than 2.3%
of portfolio value on 1% of trading days" -- never a raw negative return.
This convention is what makes the three methods directly comparable and
interchangeable behind this interface.
"""

import numpy as np
from scipy.stats import norm


class VaRModel:
    """Common interface every VaR method implements. See module docstring."""

    def fit(self, returns):
        raise NotImplementedError

    def calculate(self, alpha):
        raise NotImplementedError


class HistoricalVaR(VaRModel):
    """
    Empirical (non-parametric) VaR: makes no assumption about the shape of
    the return distribution, just reads the alpha-quantile straight off the
    observed training sample. This is what lets it catch fat tails that a
    Gaussian model would miss -- at the cost of only being as good as the
    historical window actually observed.
    """

    def __init__(self):
        self._sorted_returns = None

    def fit(self, returns):
        self._sorted_returns = np.sort(np.asarray(returns))
        return self

    def calculate(self, alpha):
        if self._sorted_returns is None:
            raise RuntimeError("Call fit() before calculate().")
        quantile = np.quantile(self._sorted_returns, alpha)
        return -quantile


class ParametricVaR(VaRModel):
    """
    Gaussian (variance-covariance) VaR: assumes returns are Normal(mu,
    sigma) and reports VaR in closed form as -(mu + z_alpha * sigma), where
    z_alpha is the standard normal's alpha-quantile. Fast and smooth, but
    only as good as the Normal assumption -- real daily returns have fatter
    tails than a Normal, which is exactly the failure mode M4's backtest is
    designed to catch.
    """

    def __init__(self):
        self._mu = None
        self._sigma = None

    def fit(self, returns):
        returns = np.asarray(returns)
        self._mu = returns.mean()
        self._sigma = returns.std(ddof=1)  # ddof=1: mu is itself estimated from this sample
        return self

    def calculate(self, alpha):
        if self._mu is None:
            raise RuntimeError("Call fit() before calculate().")
        z = norm.ppf(alpha)
        return -(self._mu + z * self._sigma)


class MonteCarloVaR(VaRModel):
    """
    Simulation-based VaR: fits the same Normal(mu, sigma) as ParametricVaR,
    then draws n_simulations synthetic returns from it and reads the
    alpha-quantile off the simulated sample the same way HistoricalVaR reads
    it off real data.

    Because it simulates from the SAME Gaussian assumption ParametricVaR
    integrates in closed form, MonteCarloVaR converges to ParametricVaR's
    answer as n_simulations grows -- it is not adding independent
    information here, it is demonstrating the simulation technique on a
    single (already-aggregated) portfolio return series. Monte Carlo only
    becomes necessary rather than illustrative once you simulate correlated
    multi-asset paths through a nonlinear portfolio, which is outside this
    project's scope.
    """

    def __init__(self, n_simulations=100_000, seed=42):
        self.n_simulations = n_simulations
        self.seed = seed
        self._simulated_returns = None

    def fit(self, returns):
        returns = np.asarray(returns)
        mu = returns.mean()
        sigma = returns.std(ddof=1)
        rng = np.random.default_rng(self.seed)  # fixed seed: reproducible VaR, not a new number every run
        self._simulated_returns = rng.normal(mu, sigma, size=self.n_simulations)
        return self

    def calculate(self, alpha):
        if self._simulated_returns is None:
            raise RuntimeError("Call fit() before calculate().")
        quantile = np.quantile(self._simulated_returns, alpha)
        return -quantile

    @property
    def simulated_returns(self):
        """
        Read-only access to the simulated draws from fit(). Exists so
        plots.py can visualize the simulated outcome distribution without
        recomputing it or reaching into a private attribute -- plots.py's
        whole design principle is drawing numbers it's handed, never
        calculating its own.
        """
        if self._simulated_returns is None:
            raise RuntimeError("Call fit() before accessing simulated_returns.")
        return self._simulated_returns
