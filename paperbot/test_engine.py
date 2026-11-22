"""Sanity checks for T+1 and costs. Run: python -m paperbot.test_engine"""

from __future__ import annotations

import pandas as pd

from paperbot.costs import trade_costs
from paperbot.engine import Account, execute_target


def test_t1_and_costs() -> None:
    day1 = pd.Timestamp("2024-01-02")
    day2 = pd.Timestamp("2024-01-03")
    prices = {"510300": 4.0}
    acct = Account(cash=10_000.0)
    trades = execute_target(acct, {"510300": 1000}, prices, day1)
    assert len(trades) == 1 and trades[0].side == "buy" and trades[0].shares == 1000
    # Same-day sell must be blocked by T+1.
    blocked = execute_target(acct, {"510300": 0}, prices, day1)
    assert blocked == []
    assert acct.shares_of("510300") == 1000
    # Next day can sell.
    sold = execute_target(acct, {"510300": 0}, prices, day2)
    assert len(sold) == 1 and sold[0].side == "sell" and sold[0].shares == 1000
    assert acct.shares_of("510300") == 0
    # Stamp tax should be charged on the sell.
    assert sold[0].stamp_tax > 0
    # Commission floor.
    tiny = trade_costs(100.0, "buy", day2, "sh")
    assert tiny["commission"] == 5.0
    print("engine checks passed")


if __name__ == "__main__":
    test_t1_and_costs()
