"""
Statistical backtests for VaR models: Kupiec's proportion-of-failures (POF)
test and Christoffersen's independence test, run over the temporal hold-out
set produced by data.train_test_split_returns. This is the module the whole
project builds up to -- everything in M1-M3 exists to produce the numbers
these functions check.

Plain functions throughout, deliberately -- a Kupiec test is a function that
takes a violation series and returns a statistic, not an object with state.

Scope boundary: this backtests VaR only. Backtesting Expected Shortfall
properly (e.g. the Acerbi-Szekely test) needs machinery outside this
project's toolkit and roadmap; ES's role here is the M3 coherence argument,
not something statistically backtested.
"""

import numpy as np
from scipy.stats import chi2


def count_violations(test_returns, var_threshold):
    """
    Binary exception series over the test window: 1 on a day the actual
    loss exceeded the (fixed, train-estimated) VaR threshold, 0 otherwise.

    var_threshold is the POSITIVE loss magnitude VaR reports (see
    var_models.py's sign convention), so a violation is
    actual_return < -var_threshold.
    """
    test_returns = np.asarray(test_returns)
    return (test_returns < -var_threshold).astype(int)


def _binomial_log_likelihood(n, x, p):
    """
    log L(p) = (n-x)*ln(1-p) + x*ln(p) for x successes out of n trials at
    rate p. Guards the x=0 and x=n edges explicitly (0*ln(0) := 0, its
    calculus limit) instead of letting log(0) blow up -- both Kupiec and
    Christoffersen hit these edges whenever a violation count is exactly 0
    (a well-behaved model can easily have zero violations in the tail).
    """
    log_l = 0.0
    if x > 0:
        log_l += x * np.log(p)
    if x < n:
        log_l += (n - x) * np.log(1 - p)
    return log_l


def kupiec_pof_test(violations, alpha):
    """
    Likelihood-ratio test of whether the OBSERVED violation rate matches the
    PROMISED rate alpha, ignoring when the violations happened.

    LR_pof = -2 * [ logL(alpha) - logL(x/n) ]  ~ chi2(1) under H0

    logL(alpha) is the likelihood if the model's promised rate were exactly
    right; logL(x/n) is the best-fit (unconstrained) likelihood at the
    observed rate. Returns (LR statistic, p-value).
    """
    n = len(violations)
    x = int(np.sum(violations))
    pi_hat = x / n

    log_l_null = _binomial_log_likelihood(n, x, alpha)
    log_l_alt = _binomial_log_likelihood(n, x, pi_hat)

    lr_stat = -2 * (log_l_null - log_l_alt)
    p_value = chi2.sf(lr_stat, df=1)
    return lr_stat, p_value


def count_transitions(violations):
    """
    Count day-to-day (violation[t] -> violation[t+1]) transitions:
    n00 = no violation -> no violation
    n01 = no violation -> violation
    n10 = violation    -> no violation
    n11 = violation    -> violation

    Split out from christoffersen_independence_test as its own function
    because this counting step -- not the log-likelihood math -- is where an
    off-by-one bug would actually hide, so it's worth being independently
    checkable and hand-verifiable.
    """
    violations = np.asarray(violations)
    prev, curr = violations[:-1], violations[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    return n00, n01, n10, n11


def christoffersen_independence_test(violations):
    """
    Likelihood-ratio test of whether violations are INDEPENDENT over time --
    catches a model whose violation count looks fine but whose violations
    cluster together (e.g. all in one crash week), which Kupiec's test alone
    cannot see.

    Treats the violation series as a 2-state Markov chain and compares the
    unconstrained model (violation probability allowed to depend on
    yesterday: separate pi01, pi11) against the constrained null (a single
    shared pi, i.e. true independence):

    LR_ind = -2 * [ logL(pi; pooled) - logL(pi01, pi11; split) ]  ~ chi2(1)

    Returns (LR statistic, p-value).
    """
    n00, n01, n10, n11 = count_transitions(violations)

    n0 = n00 + n01  # days followed a no-violation day
    n1 = n10 + n11  # days followed a violation day
    pi01 = n01 / n0 if n0 > 0 else 0.0
    pi11 = n11 / n1 if n1 > 0 else 0.0
    pi_pooled = (n01 + n11) / (n0 + n1)

    log_l_null = _binomial_log_likelihood(n0, n01, pi_pooled) + _binomial_log_likelihood(n1, n11, pi_pooled)
    log_l_alt = _binomial_log_likelihood(n0, n01, pi01) + _binomial_log_likelihood(n1, n11, pi11)

    lr_stat = -2 * (log_l_null - log_l_alt)
    p_value = chi2.sf(lr_stat, df=1)
    return lr_stat, p_value


def conditional_coverage_test(violations, alpha):
    """
    Christoffersen's (1998) combined test: correct violation FREQUENCY AND
    INDEPENDENCE, jointly. LR_cc = LR_pof + LR_ind ~ chi2(2). A model can
    fail this even if it individually passes Kupiec and Christoffersen at
    the margin, since it pools both statistics before comparing to a
    stricter joint critical value.
    """
    lr_pof, _ = kupiec_pof_test(violations, alpha)
    lr_ind, _ = christoffersen_independence_test(violations)
    lr_cc = lr_pof + lr_ind
    p_value = chi2.sf(lr_cc, df=2)
    return lr_cc, p_value


def backtest_var(test_returns, var_threshold, alpha, significance=0.05):
    """
    Run the full backtest suite for one VaR method at one confidence level
    and return a single flat dict -- one row of the eventual results table
    comparing all methods side by side.

    A test "passes" when its p-value is >= significance, i.e. we fail to
    reject the null of correct calibration / independence.
    """
    violations = count_violations(test_returns, var_threshold)
    n = len(violations)
    x = int(violations.sum())

    kupiec_stat, kupiec_p = kupiec_pof_test(violations, alpha)
    christoffersen_stat, christoffersen_p = christoffersen_independence_test(violations)
    cc_stat, cc_p = conditional_coverage_test(violations, alpha)

    return {
        "n_obs": n,
        "n_violations": x,
        "expected_violations": alpha * n,
        "violation_rate": x / n,
        "kupiec_stat": kupiec_stat,
        "kupiec_p_value": kupiec_p,
        "kupiec_pass": kupiec_p >= significance,
        "christoffersen_stat": christoffersen_stat,
        "christoffersen_p_value": christoffersen_p,
        "christoffersen_pass": christoffersen_p >= significance,
        "conditional_coverage_stat": cc_stat,
        "conditional_coverage_p_value": cc_p,
        "conditional_coverage_pass": cc_p >= significance,
    }
