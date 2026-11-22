"""Shared simulation engine for backtests and the paper account.

Rules (same in both modes):
- After close of day T, a signal may be computed from data through T.
- The order is filled at the close of the next trading day, with slippage.
- Shares bought on day D cannot be sold until the next trading day (T+1).
- Trades are rounded to 100-share lots.
- No broker, exchange, or wallet is ever contacted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from paperbot import config
from paperbot.costs import fill_price, trade_costs


@dataclass
class Lot:
    shares: int
    acquire_date: pd.Timestamp


@dataclass
class Trade:
    date: pd.Timestamp
    symbol: str
    side: str
    shares: int
    price: float
    notional: float
    commission: float
    stamp_tax: float
    transfer_fee: float
    cost: float


@dataclass
class Account:
    cash: float
    lots: dict[str, list[Lot]] = field(default_factory=dict)

    def shares_of(self, symbol: str) -> int:
        return int(sum(lot.shares for lot in self.lots.get(symbol, [])))

    def sellable_shares(self, symbol: str, trade_date: pd.Timestamp) -> int:
        day = pd.Timestamp(trade_date).normalize()
        return int(
            sum(lot.shares for lot in self.lots.get(symbol, []) if lot.acquire_date < day)
        )

    def position_value(self, prices: dict[str, float]) -> float:
        total = 0.0
        for symbol, lots in self.lots.items():
            qty = sum(lot.shares for lot in lots)
            if qty:
                total += qty * float(prices[symbol])
        return total

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + self.position_value(prices)


def _market(symbol: str) -> str:
    try:
        return config.etf_info(symbol)["market"]
    except KeyError:
        return "sz" if str(symbol).startswith("15") else "sh"


def _add_lot(account: Account, symbol: str, shares: int, day: pd.Timestamp) -> None:
    account.lots.setdefault(symbol, []).append(Lot(shares=shares, acquire_date=day))


def _consume_lots(account: Account, symbol: str, shares: int, day: pd.Timestamp) -> None:
    remaining = shares
    kept: list[Lot] = []
    for lot in account.lots.get(symbol, []):
        if remaining <= 0 or lot.acquire_date >= day:
            kept.append(lot)
            continue
        take = min(lot.shares, remaining)
        lot.shares -= take
        remaining -= take
        if lot.shares > 0:
            kept.append(lot)
    if remaining > 0:
        raise RuntimeError(f"T+1 violation or short sell attempted: {symbol} short by {remaining}")
    account.lots[symbol] = kept


def execute_target(
    account: Account,
    target_shares: dict[str, int],
    prices: dict[str, float],
    trade_date: pd.Timestamp,
) -> list[Trade]:
    """Move the account toward target_shares with T+1, lot size and costs.

    Sells are processed first so cash can fund buys on the same day.
    Buys that still don't fit in cash are scaled down to a feasible lot.
    """
    trades: list[Trade] = []
    day = pd.Timestamp(trade_date).normalize()
    symbols = sorted(set(target_shares) | set(account.lots) | set(prices))

    desired = {s: int(target_shares.get(s, 0)) for s in symbols}
    for s, qty in desired.items():
        if qty % config.LOT_SIZE != 0:
            desired[s] = (qty // config.LOT_SIZE) * config.LOT_SIZE

    # Sells first.
    for symbol in symbols:
        current = account.shares_of(symbol)
        goal = desired.get(symbol, 0)
        if goal >= current:
            continue
        to_sell = current - goal
        sellable = account.sellable_shares(symbol, day)
        to_sell = min(to_sell, sellable)
        to_sell = (to_sell // config.LOT_SIZE) * config.LOT_SIZE
        if to_sell <= 0:
            continue
        px = fill_price(prices[symbol], "sell")
        notional = px * to_sell
        costs = trade_costs(notional, "sell", day, _market(symbol))
        proceeds = notional - costs["total"]
        _consume_lots(account, symbol, to_sell, day)
        account.cash += proceeds
        trades.append(
            Trade(
                date=day,
                symbol=symbol,
                side="sell",
                shares=to_sell,
                price=px,
                notional=round(notional, 4),
                commission=costs["commission"],
                stamp_tax=costs["stamp_tax"],
                transfer_fee=costs["transfer_fee"],
                cost=costs["total"],
            )
        )

    # Buys next, largest first, scaled to remaining cash.
    buy_list = []
    for symbol in symbols:
        current = account.shares_of(symbol)
        goal = desired.get(symbol, 0)
        if goal > current:
            buy_list.append((symbol, goal - current))
    buy_list.sort(key=lambda x: x[1] * prices[x[0]], reverse=True)

    for symbol, to_buy in buy_list:
        to_buy = (to_buy // config.LOT_SIZE) * config.LOT_SIZE
        if to_buy <= 0:
            continue
        raw = prices[symbol]
        px = fill_price(raw, "buy")
        # Iteratively shrink lots until cash covers notional + costs.
        while to_buy >= config.LOT_SIZE:
            notional = px * to_buy
            costs = trade_costs(notional, "buy", day, _market(symbol))
            need = notional + costs["total"]
            if need <= account.cash + 1e-8:
                account.cash -= need
                _add_lot(account, symbol, to_buy, day)
                trades.append(
                    Trade(
                        date=day,
                        symbol=symbol,
                        side="buy",
                        shares=to_buy,
                        price=px,
                        notional=round(notional, 4),
                        commission=costs["commission"],
                        stamp_tax=costs["stamp_tax"],
                        transfer_fee=costs["transfer_fee"],
                        cost=costs["total"],
                    )
                )
                break
            to_buy -= config.LOT_SIZE
    return trades


def shares_for_weight(
    equity: float,
    weight: float,
    price: float,
) -> int:
    """Lots we can buy for a target weight, ignoring costs (execution will scale)."""
    if weight <= 0 or price <= 0 or equity <= 0:
        return 0
    raw = equity * weight / price
    return int(raw // config.LOT_SIZE) * config.LOT_SIZE


def run_simulation(
    close: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]],
    initial_cash: float = config.INITIAL_CASH,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """Run a daily simulation.

    ``signal_fn(history_through_T, T)`` must return target weights that sum to
    at most 1.0. Weights are converted to lots at the *next* bar's close.
    """
    close = close.sort_index()
    days = close.index
    if start:
        days = days[days >= pd.Timestamp(start)]
    if end:
        days = days[days <= pd.Timestamp(end)]
    days = pd.DatetimeIndex(days)
    if len(days) < 3:
        raise ValueError("not enough bars to simulate")

    # Need one bar before the first fill to form the first signal.
    full_index = close.index
    start_loc = full_index.get_loc(days[0])
    if isinstance(start_loc, slice):
        start_loc = start_loc.start

    account = Account(cash=float(initial_cash))
    pending_weights: dict[str, float] | None = None
    equity_rows = []
    trade_rows: list[Trade] = []

    for i, day in enumerate(days):
        prices = close.loc[day].astype(float).to_dict()

        if pending_weights is not None:
            equity_before = account.equity(prices)
            target_shares = {
                s: shares_for_weight(equity_before, w, prices[s])
                for s, w in pending_weights.items()
            }
            # Explicitly flatten symbols not in the target to 0.
            for s in list(account.lots):
                target_shares.setdefault(s, 0)
            fills = execute_target(account, target_shares, prices, day)
            trade_rows.extend(fills)
            pending_weights = None

        equity = account.equity(prices)
        pos = {s: account.shares_of(s) for s in close.columns if account.shares_of(s)}
        equity_rows.append(
            {
                "date": day,
                "cash": account.cash,
                "equity": equity,
                "position_value": account.position_value(prices),
                "positions": pos,
            }
        )

        hist = close.loc[:day]
        weights = signal_fn(hist, day)
        # None = hold current lots (no order tomorrow).
        # {} = go to cash. Missing symbols in a dict are flattened to 0.
        if weights is None:
            pending_weights = None
        else:
            pending_weights = {s: float(w) for s, w in weights.items() if float(w) != 0.0}

    equity_df = pd.DataFrame(equity_rows).set_index("date")
    trades_df = pd.DataFrame([t.__dict__ for t in trade_rows])
    if not trades_df.empty:
        trades_df["date"] = pd.to_datetime(trades_df["date"])
    return {
        "equity": equity_df,
        "trades": trades_df,
        "account": account,
        "last_prices": close.loc[days[-1]].astype(float).to_dict(),
        "pending_weights": pending_weights or {},
    }


def metrics_from_equity(
    equity: pd.Series,
    trades: pd.DataFrame,
    label: str,
    period_name: str,
) -> dict:
    eq = equity.dropna().astype(float)
    if len(eq) < 2:
        return {
            "strategy": label,
            "period": period_name,
            "start": None,
            "end": None,
            "n_days": int(len(eq)),
            "total_return": np.nan,
            "ann_return": np.nan,
            "max_drawdown": np.nan,
            "sharpe": np.nan,
            "ann_turnover": np.nan,
            "n_trades": 0,
            "end_equity": float(eq.iloc[-1]) if len(eq) else np.nan,
        }
    rets = eq.pct_change().dropna()
    n = int(rets.shape[0])
    total = float(eq.iloc[-1] / eq.iloc[0] - 1.0)
    years = n / config.TRADING_DAYS_PER_YEAR
    ann = float((eq.iloc[-1] / eq.iloc[0]) ** (1.0 / years) - 1.0) if years > 0 else np.nan
    dd = float((eq / eq.cummax() - 1.0).min())
    vol = float(rets.std(ddof=1)) * np.sqrt(config.TRADING_DAYS_PER_YEAR) if n > 1 else np.nan
    mean = float(rets.mean()) * config.TRADING_DAYS_PER_YEAR
    sharpe = float((mean - config.RISK_FREE_RATE) / vol) if vol and vol > 0 else np.nan

    turnover = np.nan
    n_trades = 0
    if trades is not None and not trades.empty:
        t = trades[(trades["date"] >= eq.index[0]) & (trades["date"] <= eq.index[-1])]
        n_trades = int(len(t))
        traded = float(t["notional"].abs().sum()) if n_trades else 0.0
        # One-way turnover, annualised.
        avg_eq = float(eq.mean())
        if avg_eq > 0 and n > 0:
            turnover = (traded / 2.0) / avg_eq * (config.TRADING_DAYS_PER_YEAR / n)

    return {
        "strategy": label,
        "period": period_name,
        "start": str(eq.index[0].date()),
        "end": str(eq.index[-1].date()),
        "n_days": n,
        "total_return": total,
        "ann_return": ann,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "ann_turnover": turnover,
        "n_trades": n_trades,
        "end_equity": float(eq.iloc[-1]),
    }
