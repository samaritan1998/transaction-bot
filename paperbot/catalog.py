"""Resolve a strategy key to a callable. Used by the paper job and research runs."""

from __future__ import annotations

import json

from paperbot import config
from paperbot.strategies import StrategySpec


def default_specs() -> list[StrategySpec]:
    from paperbot.backtest import build_specs

    path = config.BACKTEST_DIR / "chosen.json"
    if path.exists():
        chosen = json.loads(path.read_text())
        ma = chosen.get("dual_ma_params") or {"fast": 20, "slow": 250}
        mom = chosen.get("momentum_params") or {
            "lookback": 120,
            "rebalance_every": 20,
            "abs_filter": True,
        }
    else:
        ma = {"fast": 20, "slow": 250}
        mom = {"lookback": 120, "rebalance_every": 20, "abs_filter": True}
    return build_specs(ma, mom)



def _make_vol_target_mix_dual_ma(fast, slow, vol_lookback, target_vol, symbol):
    from paperbot.strategies import make_dual_ma, make_vol_target_mix

    base = make_dual_ma(fast, slow, symbol=symbol)
    return make_vol_target_mix(base, vol_lookback, target_vol, [symbol])


def factories() -> dict:
    from paperbot import strategies as S

    return {
        "make_vol_target_symbol": S.make_vol_target_symbol,
        "make_vol_target_equal_weight": S.make_vol_target_equal_weight,
        "make_trend_equal_weight": S.make_trend_equal_weight,
        "make_risk_parity": S.make_risk_parity,
        "make_fixed_mix": S.make_fixed_mix,
        "make_dual_ma_defensive": S.make_dual_ma_defensive,
        "make_momentum_topn": S.make_momentum_topn,
        "make_dual_ma": S.make_dual_ma,
        "make_equal_weight": S.make_equal_weight,
        "make_momentum_rotation": S.make_momentum_rotation,
        "buy_hold_hs300": lambda **_k: S.buy_hold_hs300,
        "make_vol_target_mix_dual_ma": _make_vol_target_mix_dual_ma,
    }


def specs_from_records(records: list[dict]) -> list[StrategySpec]:
    facts = factories()
    out = []
    for rec in records:
        factory = rec["factory"]
        if factory not in facts:
            raise KeyError(f"unknown strategy factory {factory}")
        params = dict(rec.get("params") or {})
        signal_fn = facts[factory](**params)
        out.append(
            StrategySpec(
                key=rec["key"],
                name=rec["name"],
                description=rec.get("description") or "",
                params=params,
                signal_fn=signal_fn,
            )
        )
    return out


def optional_specs() -> list[StrategySpec]:
    """Load optional strategies frozen after the improvement study.

    Missing file means no optional strategies have been registered yet.
    """
    path = config.IMPROVE_DIR / "optional_strategies.json"
    if not path.exists():
        return []
    records = json.loads(path.read_text())
    return specs_from_records(records)


def resolve_strategy(key: str | None = None) -> StrategySpec:
    key = key or config.PAPER_STRATEGY or "equal_weight"
    if key in ("auto", "", None):
        # Auto used to mean "best on the original four-strategy test set".
        # It must not start pointing at later research searches.
        path = config.BACKTEST_DIR / "chosen.json"
        if path.exists():
            key = json.loads(path.read_text())["best_on_test"]["key"]
        else:
            key = "equal_weight"
    specs = default_specs() + optional_specs()
    for spec in specs:
        if spec.key == key:
            return spec
    known = ", ".join(s.key for s in specs)
    raise KeyError(f"unknown paper strategy {key}. known: {known}")
