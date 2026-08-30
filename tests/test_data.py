import pandas as pd

from src.data import train_test_split_returns


def test_train_test_split_preserves_temporal_order_with_no_overlap():
    """
    The one thing this function must never get wrong: train must be
    entirely BEFORE test, in date order, with no shared or skipped dates
    -- any violation of this would leak future information into training
    (lookahead bias), which is fatal to every backtest built on top of it.
    """
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    returns = pd.Series(range(10), index=dates)

    train, test = train_test_split_returns(returns, train_fraction=0.7)

    assert len(train) == 7
    assert len(test) == 3
    assert train.index.max() < test.index.min()
    assert len(train) + len(test) == len(returns)
