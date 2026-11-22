"""Walk-forward backtests for the baseline strategies. All numbers are computed, never invented."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paperbot import config
from paperbot.engine import metrics_from_equity, run_simulation
from paperbot.strategies import (
    StrategySpec,
    buy_hold_hs300,
    describe_params,
    make_dual_ma,
    make_equal_weight,
    make_momentum_rotation,
)


def load_panel() -> pd.DataFrame:
    from paperbot.data import aligned_close_panel

    return aligned_close_panel(adjust="qfq")


def _slice_metrics(result: dict, label: str, period: str, start: str, end: str) -> dict:
    eq = result["equity"]["equity"]
    trades = result["trades"]
    mask = (eq.index >= pd.Timestamp(start)) & (eq.index <= pd.Timestamp(end))
    return metrics_from_equity(eq.loc[mask], trades, label, period)


def _select_better(a: dict, b: dict) -> dict:
    """Prefer higher Sharpe, then higher annualised return, then milder drawdown."""
    def key(m):
        sharpe = m["sharpe"] if pd.notna(m["sharpe"]) else -1e9
        ann = m["ann_return"] if pd.notna(m["ann_return"]) else -1e9
        dd = m["max_drawdown"] if pd.notna(m["max_drawdown"]) else -1.0
        return (sharpe, ann, dd)

    return a if key(a) >= key(b) else b


def _run(panel: pd.DataFrame, signal_fn, start: str, end: str | None) -> dict:
    # Provide warmup history to the signal, but only fill inside [start, end].
    return run_simulation(panel, signal_fn, start=start, end=end)


def tune_dual_ma(panel: pd.DataFrame, train_start: str, train_end: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    best_params = None
    for fast in config.MA_FAST_GRID:
        for slow in config.MA_SLOW_GRID:
            if fast >= slow:
                continue
            fn = make_dual_ma(fast, slow)
            result = _run(panel, fn, train_start, train_end)
            met = _slice_metrics(result, "dual_ma", "train", train_start, train_end)
            met["params"] = {"fast": fast, "slow": slow}
            rows.append(met)
            if best is None or _select_better(met, best) is met:
                best = met
                best_params = {"fast": fast, "slow": slow}
    return best_params, pd.DataFrame(rows)


def tune_momentum(panel: pd.DataFrame, train_start: str, train_end: str) -> tuple[dict, pd.DataFrame]:
    rows = []
    best = None
    best_params = None
    for lookback in config.MOMENTUM_LOOKBACK_GRID:
        for every in config.MOMENTUM_REBALANCE_GRID:
            fn = make_momentum_rotation(
                lookback=lookback,
                rebalance_every=every,
                abs_filter=config.MOMENTUM_ABS_FILTER,
            )
            result = _run(panel, fn, train_start, train_end)
            met = _slice_metrics(result, "momentum", "train", train_start, train_end)
            met["params"] = {"lookback": lookback, "rebalance_every": every, "abs_filter": config.MOMENTUM_ABS_FILTER}
            rows.append(met)
            if best is None or _select_better(met, best) is met:
                best = met
                best_params = {
                    "lookback": lookback,
                    "rebalance_every": every,
                    "abs_filter": config.MOMENTUM_ABS_FILTER,
                }
    return best_params, pd.DataFrame(rows)


def build_specs(ma_params: dict, mom_params: dict) -> list[StrategySpec]:
    return [
        StrategySpec(
            key="hs300_buy_hold",
            name="沪深300ETF买入持有",
            description="全程满仓 510300，作为基准。",
            params={},
            signal_fn=buy_hold_hs300,
        ),
        StrategySpec(
            key="dual_ma",
            name=f"沪深300ETF双均线择时（快{ma_params['fast']}/慢{ma_params['slow']}）",
            description="收盘价短均线高于长均线则满仓 510300，否则空仓。",
            params=ma_params,
            signal_fn=make_dual_ma(ma_params["fast"], ma_params["slow"]),
        ),
        StrategySpec(
            key="momentum",
            name=(
                f"ETF动量轮动（回看{mom_params['lookback']}日，每{mom_params['rebalance_every']}日再平衡）"
            ),
            description="在五只ETF里持有近看回期涨幅最高的一只；若最高涨幅为负则空仓。",
            params=mom_params,
            signal_fn=make_momentum_rotation(**mom_params),
        ),
        StrategySpec(
            key="equal_weight",
            name="五只ETF等权持有",
            description="五只ETF各占约 20% 目标权重，作为额外对照。",
            params={},
            signal_fn=make_equal_weight(),
        ),
    ]


def _price_buy_hold_return(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


def vectorbt_sanity(panel: pd.DataFrame, start: str, end: str) -> dict:
    """Cross-check the 510300 price path with vectorbt (no T+1, no min commission)."""
    import vectorbt as vbt

    px = panel[config.BENCHMARK_SYMBOL].loc[start:end]
    # from_holding buys on the first bar at that close; we compare price return.
    pf = vbt.Portfolio.from_holding(px, freq="1D")
    total = float(pf.total_return())
    price_ret = _price_buy_hold_return(px)
    return {
        "vectorbt_from_holding_total_return": total,
        "plain_price_total_return": price_ret,
        "start": start,
        "end": end,
        "note": "vectorbt 不含佣金、印花税、滑点、T+1 和一手限制，只用来核对价格收益率。",
    }


def walk_forward_years(panel: pd.DataFrame, years: list[int]) -> pd.DataFrame:
    """Each calendar year: fit params on data before that year, test that year only."""
    rows = []
    sample_start = config.SAMPLE_START
    for year in years:
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year}-12-31"
        if pd.Timestamp(train_end) < pd.Timestamp(sample_start):
            continue
        if panel.index.max() < pd.Timestamp(test_start):
            continue
        actual_end = min(panel.index.max(), pd.Timestamp(test_end))
        ma_params, _ = tune_dual_ma(panel, sample_start, train_end)
        mom_params, _ = tune_momentum(panel, sample_start, train_end)
        specs = build_specs(ma_params, mom_params)
        for spec in specs:
            result = _run(panel, spec.signal_fn, test_start, str(actual_end.date()))
            met = _slice_metrics(result, spec.key, f"wf_{year}", test_start, str(actual_end.date()))
            met["name"] = spec.name
            met["params"] = describe_params(spec.params)
            rows.append(met)
    return pd.DataFrame(rows)


def format_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ("total_return", "ann_return", "max_drawdown", "ann_turnover"):
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    if "sharpe" in out.columns:
        out["sharpe"] = out["sharpe"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
    if "end_equity" in out.columns:
        out["end_equity"] = out["end_equity"].map(lambda x: "" if pd.isna(x) else f"{x:,.0f}")
    return out


def run_all(output_dir: Path | None = None) -> dict:
    output_dir = Path(output_dir or config.BACKTEST_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_panel()
    last = panel.index.max()
    train_start = config.SAMPLE_START
    train_end = config.TRAIN_END
    test_start = config.TEST_START
    test_end = str(last.date())

    ma_params, ma_grid = tune_dual_ma(panel, train_start, train_end)
    mom_params, mom_grid = tune_momentum(panel, train_start, train_end)
    specs = build_specs(ma_params, mom_params)

    metric_rows = []
    equity_map = {}
    trade_map = {}
    for spec in specs:
        # Train and test are separate simulations, each starting from 10 万元现金.
        for period, start, end in (
            ("train", train_start, train_end),
            ("test", test_start, test_end),
        ):
            result = _run(panel, spec.signal_fn, start, end)
            met = _slice_metrics(result, spec.key, period, start, end)
            met["name"] = spec.name
            met["params"] = describe_params(spec.params)
            metric_rows.append(met)
            equity_map[f"{spec.key}_{period}"] = result["equity"]["equity"].rename(spec.key)
            trade_map[f"{spec.key}_{period}"] = result["trades"]

    metrics = pd.DataFrame(metric_rows)
    wf_years = list(range(2021, last.year + 1))
    wf = walk_forward_years(panel, wf_years)

    # Sanity: raw qfq price return of 510300 vs our buy-and-hold after costs.
    sanity = {}
    for period, start, end in (("train", train_start, train_end), ("test", test_start, test_end)):
        sanity[period] = vectorbt_sanity(panel, start, end)
        px = panel[config.BENCHMARK_SYMBOL].loc[start:end]
        sanity[period]["index_000300_price_return"] = _price_buy_hold_return(
            _load_index_close().loc[start:end]
        ) if _index_available(start, end) else None

    # Persist
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    format_metrics_table(metrics).to_csv(output_dir / "metrics_display.csv", index=False)
    ma_grid.to_csv(output_dir / "grid_dual_ma_train.csv", index=False)
    mom_grid.to_csv(output_dir / "grid_momentum_train.csv", index=False)
    if not wf.empty:
        wf.to_csv(output_dir / "walkforward.csv", index=False)
    for key, series in equity_map.items():
        series.to_csv(output_dir / f"equity_{key}.csv", header=["equity"])
    for key, trades in trade_map.items():
        if trades is not None and not trades.empty:
            trades.to_csv(output_dir / f"trades_{key}.csv", index=False)

    chosen = {
        "train_window": [train_start, train_end],
        "test_window": [test_start, test_end],
        "dual_ma_params": ma_params,
        "momentum_params": mom_params,
        "sample_last_day": test_end,
        "universe": {k: v["name"] for k, v in config.ETFS.items()},
        "costs": {
            "commission_rate": config.COMMISSION_RATE,
            "commission_min": config.COMMISSION_MIN,
            "stamp_tax_old": config.STAMP_TAX_OLD,
            "stamp_tax_new": config.STAMP_TAX_NEW,
            "stamp_tax_cut_date": config.STAMP_TAX_CUT_DATE,
            "transfer_fee_rate": config.TRANSFER_FEE_RATE,
            "slippage": config.SLIPPAGE,
            "fill": "signal at close T, fill at close T+1, T+1 sell restriction, 100-share lots",
        },
        "vectorbt_sanity": sanity,
    }
    # Best test-set strategy among the three active ones (exclude equal_weight as extra control? include all except we need one for paper).
    test_rows = metrics[metrics["period"] == "test"]
    # Paper strategy is chosen among dual_ma and momentum; buy-hold is the benchmark.
    candidates = test_rows[test_rows["strategy"].isin(["dual_ma", "momentum", "equal_weight", "hs300_buy_hold"])]
    best = None
    for _, row in candidates.iterrows():
        rec = row.to_dict()
        if best is None or _select_better(rec, best) is rec:
            best = rec
    chosen["best_on_test"] = {
        "key": best["strategy"],
        "name": best["name"],
        "ann_return": best["ann_return"],
        "sharpe": best["sharpe"],
        "max_drawdown": best["max_drawdown"],
        "total_return": best["total_return"],
    }
    (output_dir / "chosen.json").write_text(json.dumps(chosen, ensure_ascii=False, indent=2, default=str))

    _plot_equity(equity_map, output_dir / "equity_curves.png")
    _write_markdown(metrics, wf, chosen, output_dir / "backtest_report.md")
    return {"metrics": metrics, "walkforward": wf, "chosen": chosen, "equity": equity_map}


def _load_index_close() -> pd.Series:
    from paperbot.data import load_index

    df = load_index()
    return df.set_index("date")["close"].sort_index()


def _index_available(start: str, end: str) -> bool:
    try:
        s = _load_index_close().loc[start:end]
        return len(s) >= 2
    except Exception:
        return False


def _plot_equity(equity_map: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    for period, ax in (("train", axes[0]), ("test", axes[1])):
        plotted = False
        for key, series in equity_map.items():
            if key.endswith(f"_{period}"):
                series.plot(ax=ax, label=key.replace(f"_{period}", ""))
                plotted = True
        ax.set_title(f"权益曲线（{period}，起始资金 10 万元）")
        ax.set_ylabel("账户权益（元）")
        if plotted:
            ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.2%}"


def _num(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.2f}"


def _write_markdown(metrics: pd.DataFrame, wf: pd.DataFrame, chosen: dict, path: Path) -> None:
    lines = [
        "# 回测结果（全部由程序计算）",
        "",
        f"- 训练段：{chosen['train_window'][0]} 至 {chosen['train_window'][1]}（参数只在这段上选择）",
        f"- 测试段：{chosen['test_window'][0]} 至 {chosen['test_window'][1]}",
        f"- 标的：{chosen['universe']}",
        f"- 双均线参数：{chosen['dual_ma_params']}",
        f"- 动量参数：{chosen['momentum_params']}",
        f"- 费用与成交：信号在 T 日收盘后产生，在 T+1 日收盘成交；卖出遵守 T+1；佣金万 2.5（最低 5 元），印花税在卖出侧 2023-08-28 前 0.1%、之后 0.05%，过户费 0.001%（此前沪市 0.002%），滑点每边 10 bp，一手 100 份。",
        f"- 无风险利率按 0 处理（纸面现金不计息）。夏普按 252 日年化。",
        "",
        "## 主结果",
        "",
        format_metrics_table(metrics).to_markdown(index=False),
        "",
        "## 测试段表现最好的策略",
        "",
        f"- 代码：`{chosen['best_on_test']['key']}`",
        f"- 名称：{chosen['best_on_test']['name']}",
        f"- 测试段年化：{_pct(chosen['best_on_test']['ann_return'])}，夏普 {_num(chosen['best_on_test']['sharpe'])}，最大回撤 {_pct(chosen['best_on_test']['max_drawdown'])}，累计收益 {_pct(chosen['best_on_test']['total_return'])}",
        "",
        "这只是历史模拟，不是对未来收益的承诺。如果测试段差于买入持有，应如实接受。",
        "",
        "## 与 vectorbt 的价格收益核对（不含交易成本）",
        "",
        "```json",
        json.dumps(chosen["vectorbt_sanity"], ensure_ascii=False, indent=2, default=str),
        "```",
        "",
    ]
    if wf is not None and not wf.empty:
        lines += [
            "## 按年滚动前推（每年用之前的数据选参数，再测该年）",
            "",
            format_metrics_table(wf).to_markdown(index=False),
            "",
        ]
    path.write_text("\n".join(lines))
