"""
Expected Shortfall (ES / Conditional VaR): the average loss GIVEN that the
loss is already in the worst alpha% of days, as opposed to VaR, which only
reports the threshold those worst days start at.

Kept in its own file, deliberately not bolted onto the VaRModel hierarchy in
var_models.py. That separation is itself a point worth being able to make:
VaR is not a "coherent" risk measure (it can fail subadditivity -- a
combined portfolio's VaR can exceed the sum of its parts' VaRs, which would
absurdly say diversification increased risk). ES, defined as the tail
average below, IS coherent, which is why bank regulation (Basel's
Fundamental Review of the Trading Book) moved from 99% VaR to 97.5% ES as
the standard market-risk measure. ES isn't "one more number next to VaR" --
it's the theoretically sound version of the same idea.

Plain functions here, not classes: unlike the three VaR methods, there is no
interchangeability contract to enforce (nothing downstream needs to swap ES
implementations transparently), so the inheritance used in var_models.py
would be structure without a purpose. Each function below takes TRAINING
returns and alpha, and returns ES as a POSITIVE loss magnitude -- the same
sign convention as var_models.py, for the same reason: comparability.
"""

import numpy as np
from scipy.stats import norm


def historical_es(returns, alpha):
    """
    Average of the k worst observed returns, k = ceil(alpha * n) -- e.g. at
    alpha=0.01 on 3,900 days, the worst ~39 days. No distributional
    assumption, same spirit as HistoricalVaR: read the tail average straight
    off what actually happened.
    """
    sorted_returns = np.sort(np.asarray(returns))
    n = len(sorted_returns)
    k = max(1, int(np.ceil(alpha * n)))
    tail = sorted_returns[:k]
    return -tail.mean()


def parametric_es(returns, alpha):
    """
    Closed-form ES under the Normal(mu, sigma) assumption. The tail
    expectation of a normal variable below its alpha-quantile is:

        E[X | X <= q_alpha] = mu - sigma * phi(z_alpha) / alpha

    where z_alpha = Phi^-1(alpha) (norm.ppf) and phi is the standard normal
    density (norm.pdf) -- mechanically, the "center of mass" of the sliver
    of the bell curve past the cutoff, normalized by how much probability
    mass is in that sliver. Same mu/sigma estimation (ddof=1) as
    ParametricVaR.
    """
    returns = np.asarray(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    z = norm.ppf(alpha)
    tail_expectation = mu - sigma * norm.pdf(z) / alpha
    return -tail_expectation


def monte_carlo_es(returns, alpha, n_simulations=100_000, seed=42):
    """
    Fits the same Normal(mu, sigma) as MonteCarloVaR, draws n_simulations
    synthetic returns from it, then applies the exact same tail-average
    recipe as historical_es to the simulated sample instead of real data --
    same relationship MonteCarloVaR has to HistoricalVaR. Converges to
    parametric_es's closed-form answer as n_simulations grows, for the same
    reason MonteCarloVaR converges to ParametricVaR.
    """
    returns = np.asarray(returns)
    mu = returns.mean()
    sigma = returns.std(ddof=1)
    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(mu, sigma, size=n_simulations)
    return historical_es(simulated_returns, alpha)
