"""Simple, explainable baseline strategies. Signals are weights, not orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from paperbot import config


SignalFn = Callable[[pd.DataFrame, pd.Timestamp], dict[str, float]]


@dataclass(frozen=True)
class StrategySpec:
    key: str
    name: str
    description: str
    params: dict
    signal_fn: SignalFn


def _as_of(hist: pd.DataFrame, day: pd.Timestamp) -> pd.DataFrame:
    return hist.loc[: pd.Timestamp(day).normalize()]


def buy_hold_hs300(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
    _ = (_as_of(hist, day), day)
    return {config.BENCHMARK_SYMBOL: 1.0}


def make_dual_ma(fast: int, slow: int, symbol: str = config.BENCHMARK_SYMBOL) -> SignalFn:
    if fast >= slow:
        raise ValueError("fast window must be < slow window")

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
        s = _as_of(hist, day)[symbol].dropna()
        if len(s) < slow:
            return {}
        ma_f = float(s.iloc[-fast:].mean())
        ma_s = float(s.iloc[-slow:].mean())
        return {symbol: 1.0} if ma_f > ma_s else {}

    return signal


def make_momentum_rotation(
    lookback: int,
    rebalance_every: int = 20,
    abs_filter: bool = True,
    symbols: list[str] | None = None,
) -> SignalFn:
    universe = symbols or list(config.ETFS.keys())

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float]:
        panel = _as_of(hist, day)[universe].dropna()
        if len(panel) < lookback + 2:
            return {}
        # Rebalance only every N trading days, measured from the start of the
        # available history so the schedule is deterministic and has no look-ahead.
        # On non-rebalance days the engine still needs a target: we repeat the
        # same rule (which is a function of today's lookback return). A more
        # sticky implementation would need position state; to keep the signal
        # pure, we recompute daily but the target only *changes* on rebalance
        # dates. We encode that by returning a sentinel? No — the backtest
        # treats each day's signal as the next target. For stickiness we use
        # the calendar position in the full panel.
        pos = len(panel) - 1
        # Hold the previous target between rebalance dates by recomputing the
        # lookback using the last rebalance day's close (data through that day).
        last_reb_pos = (pos // rebalance_every) * rebalance_every
        if last_reb_pos < lookback:
            return {}
        window = panel.iloc[: last_reb_pos + 1]
        ret = window.iloc[-1] / window.iloc[-lookback] - 1.0
        leader = str(ret.idxmax())
        best = float(ret.max())
        if abs_filter and best <= 0:
            return {}
        return {leader: 1.0}

    return signal


def make_equal_weight(rebalance_every: int = 20, symbols: list[str] | None = None) -> SignalFn:
    """Target equal weights, but only emit a new target on rebalance days.

    Returning None on other days tells the engine to hold current lots and
    not trade (otherwise daily lot rounding would inflate turnover).
    """
    universe = symbols or list(config.ETFS.keys())
    w = 1.0 / len(universe)

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)[universe].dropna()
        if panel.empty:
            return {}
        # Rebalance on the first bar, and on the first trading day of each month.
        if len(panel) == 1 or panel.index[-1].month != panel.index[-2].month:
            return {s: w for s in universe}
        return None

    return signal


def describe_params(params: dict) -> str:
    if not params:
        return "—"
    return ", ".join(f"{k}={v}" for k, v in params.items())


# ----- Improved / cross-asset families (optional; not the daily default) -----

CORE = list(config.ETFS.keys())
CROSS_ASSET = ["510300", "511010", "518880"]  # equity, treasury, gold
CROSS_ASSET_US = ["510300", "511010", "518880", "513100"]


def _is_month_start(panel: pd.DataFrame) -> bool:
    if panel.empty:
        return False
    if len(panel) == 1:
        return True
    return panel.index[-1].month != panel.index[-2].month


def _realized_vol(series: pd.Series, lookback: int) -> float | None:
    r = series.pct_change().dropna()
    if len(r) < lookback:
        return None
    sig = float(r.iloc[-lookback:].std(ddof=1))
    if sig <= 0:
        return None
    return sig * (config.TRADING_DAYS_PER_YEAR ** 0.5)


def make_vol_target_symbol(
    symbol: str,
    vol_lookback: int,
    target_vol: float,
    max_weight: float = 1.0,
) -> SignalFn:
    """Long only one symbol, scale by trailing volatility, leftover stays cash."""

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if symbol not in panel.columns:
            return {}
        if not _is_month_start(panel):
            return None
        s = panel[symbol].dropna()
        vol = _realized_vol(s, vol_lookback)
        if vol is None:
            return {symbol: 1.0}
        w = min(max_weight, float(target_vol) / vol)
        return {symbol: w} if w >= 0.02 else {}

    return signal


def make_vol_target_equal_weight(
    vol_lookback: int,
    target_vol: float,
    symbols: list[str] | None = None,
    max_scale: float = 1.0,
    leftover: str | None = None,
) -> SignalFn:
    """Equal-weight the universe, then scale the book to a volatility target.

    Residual weight stays in cash, or goes to ``leftover`` (for example a bond ETF).
    """
    universe = symbols or CORE

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        cols = [s for s in universe if s in panel.columns]
        live = panel[cols].dropna()
        if live.empty:
            return {}
        if not _is_month_start(panel):
            return None
        weights = _vol_scale_equal(live, cols, vol_lookback, target_vol, max_scale)
        if leftover:
            invested = sum(weights.values())
            residual = max(0.0, 1.0 - invested)
            if residual >= 0.02:
                weights = dict(weights)
                weights[leftover] = weights.get(leftover, 0.0) + residual
        return weights

    return signal


def _vol_scale_equal(panel, universe, vol_lookback, target_vol, max_scale):
    rets = panel[universe].pct_change().mean(axis=1).dropna()
    if len(rets) < vol_lookback:
        w = 1.0 / len(universe)
        return {s: w for s in universe}
    sig = float(rets.iloc[-vol_lookback:].std(ddof=1)) * (config.TRADING_DAYS_PER_YEAR ** 0.5)
    if sig <= 0:
        w = 1.0 / len(universe)
        return {s: w for s in universe}
    scale = min(max_scale, float(target_vol) / sig)
    if scale < 0.02:
        return {}
    w = scale / len(universe)
    return {s: w for s in universe}


def make_trend_equal_weight(
    ma_window: int,
    symbols: list[str] | None = None,
    defensive: str | None = None,
) -> SignalFn:
    """Equal-weight names above their moving average; otherwise cash or a defensive ETF."""
    universe = symbols or CORE

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if not _is_month_start(panel):
            return None
        live = []
        for s in universe:
            if s not in panel.columns:
                continue
            ser = panel[s].dropna()
            if len(ser) < ma_window:
                continue
            if float(ser.iloc[-1]) > float(ser.iloc[-ma_window:].mean()):
                live.append(s)
        if live:
            w = 1.0 / len(live)
            return {s: w for s in live}
        if defensive and defensive in panel.columns:
            return {defensive: 1.0}
        return {}

    return signal


def make_risk_parity(
    symbols: list[str],
    vol_lookback: int,
    ma_window: int | None = None,
    defensive: str | None = "511010",
) -> SignalFn:
    """Long-only inverse-volatility weights, monthly, no leverage."""

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if not _is_month_start(panel):
            return None
        chosen = []
        vols = {}
        for s in symbols:
            if s not in panel.columns:
                continue
            ser = panel[s].dropna()
            if ma_window:
                if len(ser) < ma_window:
                    continue
                if float(ser.iloc[-1]) <= float(ser.iloc[-ma_window:].mean()):
                    continue
            vol = _realized_vol(ser, vol_lookback)
            if vol is None:
                continue
            chosen.append(s)
            vols[s] = vol
        if not chosen:
            if defensive and defensive in panel.columns:
                return {defensive: 1.0}
            return {}
        inv = {s: 1.0 / vols[s] for s in chosen}
        total = sum(inv.values())
        return {s: inv[s] / total for s in chosen}

    return signal


def make_fixed_mix(weights: dict[str, float]) -> SignalFn:
    """Static mix, rebalanced on the first trading day of each month."""

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if not _is_month_start(panel):
            return None
        return {s: float(w) for s, w in weights.items() if w > 0}

    return signal


def make_dual_ma_defensive(fast: int, slow: int, symbol: str, defensive: str) -> SignalFn:
    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if not _is_month_start(panel):
            return None
        s = panel[symbol].dropna()
        if len(s) < slow:
            return {defensive: 1.0} if defensive in panel.columns else {}
        ma_f = float(s.iloc[-fast:].mean())
        ma_s = float(s.iloc[-slow:].mean())
        if ma_f > ma_s:
            return {symbol: 1.0}
        return {defensive: 1.0} if defensive in panel.columns else {}

    return signal


def make_momentum_topn(
    lookback: int,
    top_n: int = 2,
    hysteresis: float = 0.0,
    symbols: list[str] | None = None,
    defensive: str | None = None,
    abs_filter: bool = True,
) -> SignalFn:
    """Monthly top-N momentum with optional hold buffer and a defensive sleeve."""
    universe = symbols or CORE

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)[universe].dropna()
        if not _is_month_start(panel):
            return None
        if len(panel) < lookback + 2:
            return {}
        ret = panel.iloc[-1] / panel.iloc[-lookback] - 1.0
        ranked = ret.sort_values(ascending=False)
        # Reconstruct last month's members to apply hysteresis without extra state.
        prev = panel.iloc[:-1]
        prev_hold: list[str] = []
        if len(prev) >= lookback + 2:
            # last rebalance = previous month-start
            months = prev.index.to_series().dt.to_period("M")
            if months.nunique() >= 2:
                last_m = months.iloc[-1]
                # last row of the previous calendar month
                prev_month_end = prev.index[months != last_m].max()
                window = prev.loc[:prev_month_end]
                if len(window) >= lookback:
                    pret = window.iloc[-1] / window.iloc[-lookback] - 1.0
                    prev_hold = list(pret.sort_values(ascending=False).head(top_n).index)
                    if abs_filter and float(pret.max()) <= 0:
                        prev_hold = []

        picks = list(ranked.head(top_n).index)
        if abs_filter and float(ranked.iloc[0]) <= 0:
            picks = []
        if hysteresis > 0 and prev_hold:
            kept = []
            for s in prev_hold:
                if s in picks:
                    kept.append(s)
                    continue
                # keep old name if it is within hysteresis of the marginal new pick
                if s in ranked.index and picks:
                    hurdle = float(ranked.iloc[min(top_n, len(ranked)) - 1])
                    if float(ranked[s]) + 1e-12 >= hurdle - hysteresis:
                        kept.append(s)
            # fill remaining slots with new leaders not already kept
            for s in picks:
                if s not in kept and len(kept) < top_n:
                    kept.append(s)
            picks = kept[:top_n]
        if not picks:
            if defensive:
                return {defensive: 1.0}
            return {}
        w = 1.0 / len(picks)
        return {s: w for s in picks}

    return signal


def make_vol_target_mix(base_fn: SignalFn, vol_lookback: int, target_vol: float, symbols: list[str]) -> SignalFn:
    """Scale a monthly base mix by trailing vol of that mix's synthetic returns."""

    def signal(hist: pd.DataFrame, day: pd.Timestamp) -> dict[str, float] | None:
        panel = _as_of(hist, day)
        if not _is_month_start(panel):
            return None
        base = base_fn(hist, day)
        if not base:
            return base
        # Reconstruct synthetic portfolio returns with the *current* target weights
        # only for volatility; this is standard and uses data through T.
        cols = [s for s in base if s in panel.columns]
        if not cols:
            return {}
        rets = panel[cols].pct_change()
        w = pd.Series({s: base[s] for s in cols})
        w = w / w.sum()
        port = (rets * w).sum(axis=1).dropna()
        if len(port) < vol_lookback:
            return base
        sig = float(port.iloc[-vol_lookback:].std(ddof=1)) * (config.TRADING_DAYS_PER_YEAR ** 0.5)
        if sig <= 0:
            return base
        scale = min(1.0, float(target_vol) / sig)
        if scale < 0.02:
            return {}
        return {s: float(base[s] * scale) for s in base}

    return signal
