import numpy as np
from scipy.stats import norm

from src.expected_shortfall import historical_es, parametric_es


def test_parametric_es_matches_analytical_normal_formula():
    """
    For X ~ Normal(mu, sigma), the tail expectation E[X | X <= q_alpha] has
    a known closed form: mu - sigma * phi(z_alpha) / alpha, with
    z_alpha = Phi^-1(alpha) and phi the standard normal density. As in the
    ParametricVaR test, we build a 2-point sample whose sample mean and
    ddof=1 std equal the chosen mu/sigma EXACTLY (mean of [mu-d, mu+d] is
    mu; its ddof=1 std is d*sqrt(2), so d = sigma/sqrt(2) gives std = sigma
    exactly), so this checks the formula, not estimation noise.
    """
    mu, sigma, alpha = 0.001, 0.02, 0.01
    d = sigma / np.sqrt(2)
    returns = [mu - d, mu + d]

    z = norm.ppf(alpha)
    expected = -(mu - sigma * norm.pdf(z) / alpha)

    assert np.isclose(parametric_es(returns, alpha), expected)


def test_historical_es_matches_hand_computed_tail_average():
    """
    10 hand-built returns, alpha=0.2 -> k = ceil(0.2*10) = 2 worst
    observations averaged. Sorted ascending, the 2 worst (smallest) are
    -0.06 and -0.04; their average is -0.05, so ES = 0.05.
    """
    returns = [0.01, -0.04, 0.02, -0.06, 0.00, 0.03, -0.01, 0.015, -0.02, 0.025]
    # sorted ascending: [-0.06, -0.04, -0.02, -0.01, 0.00, 0.01, 0.015, 0.02, 0.025, 0.03]
    #                   worst 2: -0.06, -0.04 -> mean = -0.05

    assert np.isclose(historical_es(returns, 0.2), 0.05)
