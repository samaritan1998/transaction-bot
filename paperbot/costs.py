"""A-share trading cost model for the paper account and backtests."""

from __future__ import annotations

import pandas as pd

from paperbot import config


def stamp_tax_rate(trade_date) -> float:
    """Stamp duty applies on the sell side only."""
    day = pd.Timestamp(trade_date).normalize()
    cut = pd.Timestamp(config.STAMP_TAX_CUT_DATE)
    return config.STAMP_TAX_NEW if day >= cut else config.STAMP_TAX_OLD


def transfer_fee_rate(trade_date, market: str) -> float:
    """China Clearing transfer fee. Shenzhen had no transfer fee before 2022-04-29."""
    day = pd.Timestamp(trade_date).normalize()
    start = pd.Timestamp(config.TRANSFER_FEE_START)
    if day >= start:
        return config.TRANSFER_FEE_RATE
    return config.TRANSFER_FEE_SH_OLD if market == "sh" else 0.0


def fill_price(raw_price: float, side: str) -> float:
    """Slippage: buy slightly worse than the close, sell slightly worse than the close."""
    if side == "buy":
        return float(raw_price) * (1.0 + config.SLIPPAGE)
    if side == "sell":
        return float(raw_price) * (1.0 - config.SLIPPAGE)
    raise ValueError(f"unknown side: {side}")


def trade_costs(notional: float, side: str, trade_date, market: str) -> dict:
    """Return commission, stamp tax, transfer fee and total cost in yuan.

    ``notional`` is the absolute fill value (price * shares) after slippage.
    """
    notional = abs(float(notional))
    commission = max(notional * config.COMMISSION_RATE, config.COMMISSION_MIN if notional > 0 else 0.0)
    stamp = notional * stamp_tax_rate(trade_date) if side == "sell" else 0.0
    transfer = notional * transfer_fee_rate(trade_date, market)
    return {
        "commission": round(commission, 4),
        "stamp_tax": round(stamp, 4),
        "transfer_fee": round(transfer, 4),
        "total": round(commission + stamp + transfer, 4),
    }
