import numpy as np

from src.var_models import HistoricalVaR, ParametricVaR


def test_parametric_var_matches_analytical_normal_quantile():
    """
    VaR_99% for a Normal(mu, sigma) has a known closed form:
    -(mu + z_0.01 * sigma), with z_0.01 = -2.326 (1st percentile of the
    standard normal). We build a 2-point sample whose mean and ddof=1 std
    are EXACTLY mu and sigma by construction (mean of [mu-d, mu+d] is mu;
    its sample std is d*sqrt(2), so d = sigma/sqrt(2) gives std = sigma
    exactly) -- so fit() estimates the chosen mu/sigma with no sampling
    noise, and we're testing calculate()'s formula, not estimation error.
    """
    mu, sigma = 0.001, 0.02
    z_99 = -2.326
    d = sigma / np.sqrt(2)
    returns = [mu - d, mu + d]

    model = ParametricVaR().fit(returns)
    expected = -(mu + z_99 * sigma)

    assert np.isclose(model.calculate(0.01), expected, atol=1e-4)


def test_historical_var_matches_hand_computed_empirical_quantile():
    """
    11 hand-built returns, alpha=0.1. numpy's default (linear-interpolation)
    quantile sits at position alpha*(n-1) = 0.1*10 = 1.0 exactly -- an
    integer, so no interpolation is needed and the answer is just the
    2nd-smallest return (index 1) in the sorted sample: -0.03. Chosen
    deliberately so the expected value can be read off by hand.
    """
    returns = [0.02, -0.05, 0.00, 0.04, -0.03, 0.06, 0.01, -0.02, 0.03, 0.05, -0.01]
    # sorted ascending: [-0.05, -0.03, -0.02, -0.01, 0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    #                                index:  0      1

    model = HistoricalVaR().fit(returns)

    assert np.isclose(model.calculate(0.1), 0.03)
