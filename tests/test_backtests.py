import numpy as np

from src.backtests import (
    christoffersen_independence_test,
    count_transitions,
    kupiec_pof_test,
)


def test_kupiec_lr_is_exactly_zero_when_observed_rate_equals_alpha():
    """
    When the observed violation rate x/n EQUALS the promised rate alpha,
    the constrained (null) and unconstrained (alt) likelihoods are the
    SAME model, so the likelihood-ratio statistic must be exactly 0 and
    the p-value exactly 1 -- no estimation needed to see this by hand.
    100 test days, alpha=0.05 -> exactly 5 violations.
    """
    n, alpha = 100, 0.05
    violations = np.zeros(n, dtype=int)
    violations[:5] = 1  # exactly n*alpha violations

    lr_stat, p_value = kupiec_pof_test(violations, alpha)

    assert np.isclose(lr_stat, 0.0, atol=1e-10)
    assert np.isclose(p_value, 1.0)


def test_kupiec_rejects_a_clearly_miscalibrated_model():
    """
    20 violations out of 100 days against a promised 1% rate is wildly
    off (20x too many) -- Kupiec must reject at any conventional
    significance level.
    """
    n, alpha = 100, 0.01
    violations = np.zeros(n, dtype=int)
    violations[:20] = 1

    _, p_value = kupiec_pof_test(violations, alpha)

    assert p_value < 0.01


def test_count_transitions_matches_hand_count_on_alternating_sequence():
    """
    Perfectly alternating 0,1,0,1,... is the clearest possible case to
    hand-count: every transition is either 0->1 or 1->0, never 0->0 or
    1->1. 10 observations -> 9 transitions, split 5/4 by the fact it
    starts on 0: 0->1 five times (indices 0,2,4,6,8), 1->0 four times
    (indices 1,3,5,7).
    """
    violations = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    n00, n01, n10, n11 = count_transitions(violations)

    assert (n00, n01, n10, n11) == (0, 5, 4, 0)


def test_christoffersen_rejects_clustered_violations():
    """
    Perfect alternation is the most extreme form of dependence possible
    (today's state perfectly predicts a flip tomorrow) -- independence
    must be rejected at any conventional significance level.
    """
    violations = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 5  # 50 obs, same pattern

    _, p_value = christoffersen_independence_test(violations)

    assert p_value < 0.01
