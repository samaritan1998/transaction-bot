"""Train-only parameter search for improved / cross-asset strategies.

Writes only under results/improve/. Never overwrites results/backtest/chosen.json.
All reported numbers come from the shared simulation engine (costs, slippage, T+1).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paperbot import config
from paperbot.backtest import _run, _select_better, _slice_metrics, format_metrics_table
from paperbot.catalog import specs_from_records
from paperbot.engine import metrics_from_equity
from paperbot.strategies import (
    CORE,
    CROSS_ASSET,
    CROSS_ASSET_US,
    StrategySpec,
    buy_hold_hs300,
    describe_params,
    make_dual_ma_defensive,
    make_fixed_mix,
    make_momentum_topn,
    make_risk_parity,
    make_trend_equal_weight,
    make_vol_target_equal_weight,
    make_vol_target_mix,
    make_vol_target_symbol,
    make_dual_ma,
    make_equal_weight,
    make_momentum_rotation,
)


# Pre-declared robustness bar. A family winner must meet ALL of these
# on the 2024-01-02..test-end window, after params were chosen on 2018-2023 only.
SHARPE_EDGE = 0.15
DD_TOLERANCE = 0.05  # allowed extra drawdown vs equal-weight, in absolute terms
ANN_RET_TOLERANCE = 0.02  # allowed shortfall vs the weaker of BH and EW
MIN_TRAIN_SHARPE = 0.10


def research_symbols() -> list[str]:
    return list(dict.fromkeys(CORE + CROSS_ASSET_US))


def load_research_panel() -> pd.DataFrame:
    from paperbot.data import aligned_close_panel

    return aligned_close_panel(adjust="qfq", symbols=research_symbols())


def _candidate_records() -> list[dict]:
    """Every configuration evaluated on the training window. 47 new configs."""
    recs: list[dict] = []

    def add(family, key, name, description, factory, params):
        recs.append(
            {
                "family": family,
                "key": key,
                "name": name,
                "description": description,
                "factory": factory,
                "params": params,
            }
        )

    for lb in (20, 60):
        for tv in (0.08, 0.10, 0.12, 0.15):
            add(
                "vol_target_bh",
                f"vol_bh_lb{lb}_tv{int(tv*100)}",
                f"沪深300目标波动率（回看{lb}日，目标{tv:.0%}）",
                "按 510300 过去波动率缩放仓位，剩余现金，每月再平衡。",
                "make_vol_target_symbol",
                {"symbol": "510300", "vol_lookback": lb, "target_vol": tv, "max_weight": 1.0},
            )
            add(
                "vol_target_ew",
                f"vol_ew_lb{lb}_tv{int(tv*100)}",
                f"五只ETF等权目标波动率（回看{lb}日，目标{tv:.0%}）",
                "等权五只核心ETF，再按组合波动率缩仓，剩余现金。",
                "make_vol_target_equal_weight",
                {"vol_lookback": lb, "target_vol": tv, "max_scale": 1.0},
            )

    for tv in (0.08, 0.10, 0.12):
        add(
            "vol_ew_bond",
            f"vol_ew_bond_tv{int(tv*100)}",
            f"五只ETF等权目标波动率，剩余进国债（回看60日，目标{tv:.0%}）",
            "等权五只核心ETF并按组合波动率缩仓，剩余配置 511010。",
            "make_vol_target_equal_weight",
            {"vol_lookback": 60, "target_vol": tv, "max_scale": 1.0, "leftover": "511010"},
        )

    for ma in (60, 120, 250):
        add(
            "trend_ew",
            f"trend_ew_ma{ma}",
            f"五只ETF趋势过滤等权（均线{ma}日，空仓）",
            "只持有收盘价高于自身均线的核心ETF，等权；全不满足则空仓。",
            "make_trend_equal_weight",
            {"ma_window": ma, "symbols": CORE, "defensive": None},
        )
        add(
            "trend_ew_bond",
            f"trend_ew_bond_ma{ma}",
            f"五只ETF趋势过滤等权（均线{ma}日，防守国债）",
            "只持有收盘价高于自身均线的核心ETF；全不满足则持有 511010。",
            "make_trend_equal_weight",
            {"ma_window": ma, "symbols": CORE, "defensive": "511010"},
        )

    for lb in (20, 60):
        add(
            "risk_parity_3",
            f"rp3_lb{lb}",
            f"股债金风险平价（回看{lb}日）",
            "510300 / 511010 / 518880 按逆波动率加权，月度再平衡，不加杠杆。",
            "make_risk_parity",
            {"symbols": CROSS_ASSET, "vol_lookback": lb, "ma_window": None, "defensive": "511010"},
        )
        add(
            "risk_parity_4",
            f"rp4_lb{lb}",
            f"股债金纳风险平价（回看{lb}日）",
            "再加入 513100 纳指ETF，同样逆波动率加权。",
            "make_risk_parity",
            {"symbols": CROSS_ASSET_US, "vol_lookback": lb, "ma_window": None, "defensive": "511010"},
        )

    for ma in (120, 250):
        add(
            "rp_trend_3",
            f"rp3_trend_ma{ma}",
            f"股债金风险平价+趋势（回看60日，均线{ma}）",
            "三资产风险平价，仅纳入价格高于均线的品种，否则偏国债。",
            "make_risk_parity",
            {"symbols": CROSS_ASSET, "vol_lookback": 60, "ma_window": ma, "defensive": "511010"},
        )
        add(
            "rp_trend_4",
            f"rp4_trend_ma{ma}",
            f"股债金纳风险平价+趋势（回看60日，均线{ma}）",
            "四资产风险平价，仅纳入价格高于均线的品种。",
            "make_risk_parity",
            {"symbols": CROSS_ASSET_US, "vol_lookback": 60, "ma_window": ma, "defensive": "511010"},
        )

    add(
        "sixty_forty",
        "sixty_forty",
        "股债 60/40（510300+511010，每月再平衡）",
        "固定 60% 沪深300ETF、40% 国债ETF。",
        "make_fixed_mix",
        {"weights": {"510300": 0.60, "511010": 0.40}},
    )
    add(
        "perm_lite",
        "perm_lite",
        "股债金 40/40/20（每月再平衡）",
        "固定 40% 沪深300ETF、40% 国债ETF、20% 黄金ETF。",
        "make_fixed_mix",
        {"weights": {"510300": 0.40, "511010": 0.40, "518880": 0.20}},
    )
    add(
        "dual_ma_bond",
        "dual_ma_bond_20_250",
        "沪深300双均线 20/250，空仓改持国债",
        "沿用训练段已选定的 20/250；均线空头时持有 511010 而非现金。",
        "make_dual_ma_defensive",
        {"fast": 20, "slow": 250, "symbol": "510300", "defensive": "511010"},
    )

    for lb in (60, 120):
        for hyst in (0.0, 0.03):
            for defensive, tag in ((None, "cash"), ("511010", "bond")):
                add(
                    "mom_top2",
                    f"mom2_lb{lb}_h{int(hyst*100)}_{tag}",
                    f"月度动量前二（回看{lb}，缓冲{hyst:.0%}，防守{tag}）",
                    "在五只核心ETF中持有近看回期涨幅前两名；最高涨幅为负则现金或国债。换手缓冲减少边界抖动。",
                    "make_momentum_topn",
                    {
                        "lookback": lb,
                        "top_n": 2,
                        "hysteresis": hyst,
                        "symbols": CORE,
                        "defensive": defensive,
                        "abs_filter": True,
                    },
                )

    for tv in (0.10, 0.12, 0.15):
        # Wrapped after the fact in build_signal because the factory needs a base_fn.
        add(
            "vol_dual_ma",
            f"vol_ma20_250_tv{int(tv*100)}",
            f"双均线 20/250 再加目标波动率 {tv:.0%}",
            "20/250 多头时持有 510300 并按波动率缩仓，空头空仓。",
            "make_vol_target_mix_dual_ma",
            {"fast": 20, "slow": 250, "vol_lookback": 60, "target_vol": tv, "symbol": "510300"},
        )
    return recs


def _signal_from_record(rec: dict):
    if rec["factory"] == "make_vol_target_mix_dual_ma":
        p = rec["params"]
        base = make_dual_ma(p["fast"], p["slow"], symbol=p["symbol"])
        return make_vol_target_mix(base, p["vol_lookback"], p["target_vol"], [p["symbol"]])
    specs = specs_from_records([rec])
    return specs[0].signal_fn


def _baselines(ma_params, mom_params) -> list[StrategySpec]:
    return [
        StrategySpec("hs300_buy_hold", "沪深300ETF买入持有", "基准。", {}, buy_hold_hs300),
        StrategySpec(
            "equal_weight",
            "五只ETF等权持有",
            "当前默认纸面策略。",
            {},
            make_equal_weight(),
        ),
        StrategySpec(
            "dual_ma",
            f"沪深300ETF双均线择时（快{ma_params['fast']}/慢{ma_params['slow']}）",
            "训练段已选定的双均线。",
            ma_params,
            make_dual_ma(ma_params["fast"], ma_params["slow"]),
        ),
        StrategySpec(
            "momentum",
            f"ETF动量轮动（回看{mom_params['lookback']}日）",
            "训练段已选定的动量。",
            mom_params,
            make_momentum_rotation(**mom_params),
        ),
    ]


def _year_slices(equity: pd.Series, trades: pd.DataFrame, label: str) -> list[dict]:
    rows = []
    years = sorted(set(equity.index.year))
    for year in years:
        mask = equity.index.year == year
        eq = equity.loc[mask]
        if len(eq) < 2:
            continue
        start = str(eq.index[0].date())
        end = str(eq.index[-1].date())
        met = metrics_from_equity(eq, trades, label, f"year_{year}")
        met["start"] = start
        met["end"] = end
        rows.append(met)
    return rows


def _passes_bar(cand: dict, bh_test: dict, ew_test: dict, train: dict, year_rows: list[dict]) -> list[str]:
    """Return a list of failed checks; empty means the bar is met."""
    fails = []
    bench_sharpe = max(float(bh_test["sharpe"]), float(ew_test["sharpe"]))
    if not (float(cand["sharpe"]) >= bench_sharpe + SHARPE_EDGE):
        fails.append(
            f"测试段夏普 {cand['sharpe']:.2f} 未达到对照最高夏普 {bench_sharpe:.2f} + {SHARPE_EDGE:.2f}"
        )
    ew_dd = float(ew_test["max_drawdown"])
    if float(cand["max_drawdown"]) < ew_dd - DD_TOLERANCE:
        fails.append(
            f"测试段最大回撤 {cand['max_drawdown']:.2%} 比等权 {ew_dd:.2%} 差超过 {DD_TOLERANCE:.0%}"
        )
    floor_ann = min(float(bh_test["ann_return"]), float(ew_test["ann_return"])) - ANN_RET_TOLERANCE
    if float(cand["ann_return"]) < floor_ann:
        fails.append(
            f"测试段年化 {cand['ann_return']:.2%} 低于对照较弱一方减容忍后的 {floor_ann:.2%}"
        )
    if float(train["sharpe"]) < MIN_TRAIN_SHARPE:
        fails.append(f"训练段夏普 {train['sharpe']:.2f} 低于 {MIN_TRAIN_SHARPE:.2f}，训练期不稳健")
    # Calendar years inside the test window of this (restarted) test simulation.
    test_years = [r for r in year_rows if int(r["period"].split("_")[1]) >= 2024]
    if test_years:
        # Compare each year vs BH year from the BH test simulation, passed in later if tagged.
        pass
    return fails


def _pct(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.2%}"


def _num(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.2f}"


def run_improve(output_dir: Path | None = None) -> dict:
    output_dir = Path(output_dir or config.IMPROVE_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_research_panel()
    last = panel.index.max()
    train_start = config.SAMPLE_START
    train_end = config.TRAIN_END
    test_start = config.TEST_START
    test_end = str(last.date())

    chosen_path = config.BACKTEST_DIR / "chosen.json"
    if chosen_path.exists():
        prior = json.loads(chosen_path.read_text())
        ma_params = prior.get("dual_ma_params") or {"fast": 20, "slow": 250}
        mom_params = prior.get("momentum_params") or {
            "lookback": 120,
            "rebalance_every": 20,
            "abs_filter": True,
        }
        prior_ma_grid = 9
        prior_mom_grid = 3
    else:
        ma_params = {"fast": 20, "slow": 250}
        mom_params = {"lookback": 120, "rebalance_every": 20, "abs_filter": True}
        prior_ma_grid = 0
        prior_mom_grid = 0

    records = _candidate_records()
    n_new = len(records)

    # --- baselines, same engine, extended panel (unused columns do not change BH/EW) ---
    baseline_rows = []
    baseline_test_equity = {}
    baseline_test_trades = {}
    for spec in _baselines(ma_params, mom_params):
        for period, start, end in (("train", train_start, train_end), ("test", test_start, test_end)):
            result = _run(panel, spec.signal_fn, start, end)
            met = _slice_metrics(result, spec.key, period, start, end)
            met["name"] = spec.name
            met["family"] = "baseline"
            met["params"] = describe_params(spec.params)
            baseline_rows.append(met)
            if period == "test":
                baseline_test_equity[spec.key] = result["equity"]["equity"]
                baseline_test_trades[spec.key] = result["trades"]

    # --- every new config on TRAIN only for selection; also run TEST for the appendix ---
    all_rows = []
    for i, rec in enumerate(records, 1):
        fn = _signal_from_record(rec)
        print(f"[{i}/{n_new}] train+test {rec['key']}", flush=True)
        for period, start, end in (("train", train_start, train_end), ("test", test_start, test_end)):
            result = _run(panel, fn, start, end)
            met = _slice_metrics(result, rec["key"], period, start, end)
            met["name"] = rec["name"]
            met["family"] = rec["family"]
            met["params"] = describe_params(rec["params"])
            met["factory"] = rec["factory"]
            all_rows.append(met)

    all_df = pd.DataFrame(all_rows)
    train_df = all_df[all_df["period"] == "train"].copy()
    test_df = all_df[all_df["period"] == "test"].copy()

    # Select one winner per family on TRAIN Sharpe / ann / DD only.
    family_winners = []
    for family, grp in train_df.groupby("family", sort=False):
        best = None
        best_row = None
        for _, row in grp.iterrows():
            rec = row.to_dict()
            if best is None or _select_better(rec, best) is rec:
                best = rec
                best_row = row
        family_winners.append(best_row)

    winners_train = pd.DataFrame(family_winners)
    winner_keys = list(winners_train["strategy"])
    rec_by_key = {r["key"]: r for r in records}

    # Re-run family winners once more to keep equity/trades for yearly slices and WF.
    winner_test_equity = {}
    winner_test_trades = {}
    winner_rows = []
    year_rows = []
    wf_rows = []
    for key in winner_keys:
        rec = rec_by_key[key]
        fn = _signal_from_record(rec)
        train_res = _run(panel, fn, train_start, train_end)
        test_res = _run(panel, fn, test_start, test_end)
        tr = _slice_metrics(train_res, key, "train", train_start, train_end)
        te = _slice_metrics(test_res, key, "test", test_start, test_end)
        for met, period in ((tr, "train"), (te, "test")):
            met["name"] = rec["name"]
            met["family"] = rec["family"]
            met["params"] = describe_params(rec["params"])
            winner_rows.append(met)
        winner_test_equity[key] = test_res["equity"]["equity"]
        winner_test_trades[key] = test_res["trades"]
        year_rows.extend(_year_slices(test_res["equity"]["equity"], test_res["trades"], key))
        # Frozen-param walk-forward: each calendar year restarts from 10 万元现金.
        for year in range(2021, last.year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            if panel.index.max() < pd.Timestamp(start):
                continue
            actual_end = min(panel.index.max(), pd.Timestamp(end))
            wf_res = _run(panel, fn, start, str(actual_end.date()))
            met = _slice_metrics(wf_res, key, f"wf_{year}", start, str(actual_end.date()))
            met["name"] = rec["name"]
            met["family"] = rec["family"]
            wf_rows.append(met)

    # Yearly slices + WF for the two main benchmarks as well.
    for bkey in ("hs300_buy_hold", "equal_weight"):
        eq = baseline_test_equity[bkey]
        trd = baseline_test_trades[bkey]
        year_rows.extend(_year_slices(eq, trd, bkey))
        fn = buy_hold_hs300 if bkey == "hs300_buy_hold" else make_equal_weight()
        for year in range(2021, last.year + 1):
            start = f"{year}-01-01"
            end = f"{year}-12-31"
            actual_end = min(panel.index.max(), pd.Timestamp(end))
            wf_res = _run(panel, fn, start, str(actual_end.date()))
            met = _slice_metrics(wf_res, bkey, f"wf_{year}", start, str(actual_end.date()))
            met["name"] = "沪深300ETF买入持有" if bkey == "hs300_buy_hold" else "五只ETF等权持有"
            met["family"] = "baseline"
            wf_rows.append(met)

    baseline_df = pd.DataFrame(baseline_rows)
    winners_df = pd.DataFrame(winner_rows)
    years_df = pd.DataFrame(year_rows)
    wf_df = pd.DataFrame(wf_rows)

    bh_test = baseline_df[(baseline_df["strategy"] == "hs300_buy_hold") & (baseline_df["period"] == "test")].iloc[0].to_dict()
    ew_test = baseline_df[(baseline_df["strategy"] == "equal_weight") & (baseline_df["period"] == "test")].iloc[0].to_dict()
    bh_train = baseline_df[(baseline_df["strategy"] == "hs300_buy_hold") & (baseline_df["period"] == "train")].iloc[0].to_dict()
    ew_train = baseline_df[(baseline_df["strategy"] == "equal_weight") & (baseline_df["period"] == "train")].iloc[0].to_dict()

    # Yearly BH sharpe on the test-window slices, for the 2-of-3 year rule.
    bh_year = {
        int(r["period"].split("_")[1]): r
        for r in years_df.to_dict("records")
        if r["strategy"] == "hs300_buy_hold" and str(r["period"]).startswith("year_")
    }

    judgement = []
    promoted = []
    for key in winner_keys:
        rec = rec_by_key[key]
        te = winners_df[(winners_df["strategy"] == key) & (winners_df["period"] == "test")].iloc[0].to_dict()
        tr = winners_df[(winners_df["strategy"] == key) & (winners_df["period"] == "train")].iloc[0].to_dict()
        yrows = [r for r in years_df.to_dict("records") if r["strategy"] == key]
        fails = _passes_bar(te, bh_test, ew_test, tr, yrows)
        test_years = [r for r in yrows if int(r["period"].split("_")[1]) >= 2024]
        n_beat_bh = 0
        year_notes = []
        for r in test_years:
            y = int(r["period"].split("_")[1])
            if y in bh_year and float(r["sharpe"]) >= float(bh_year[y]["sharpe"]):
                n_beat_bh += 1
            if float(r["max_drawdown"]) < -0.40 and float(bh_year.get(y, r)["max_drawdown"]) > -0.35:
                fails.append(f"{y} 年最大回撤 {r['max_drawdown']:.2%}，明显差于同期买入持有")
            year_notes.append(
                f"{y} 夏普 {float(r['sharpe']):.2f} / 买入持有 {float(bh_year[y]['sharpe']):.2f}"
                if y in bh_year
                else f"{y}"
            )
        if len(test_years) >= 2 and n_beat_bh < 2:
            fails.append(f"测试窗内仅 {n_beat_bh}/{len(test_years)} 个日历年夏普不低于买入持有（{', '.join(year_notes)}）")
        item = {
            "key": key,
            "family": rec["family"],
            "name": rec["name"],
            "params": rec["params"],
            "factory": rec["factory"],
            "description": rec["description"],
            "train": {k: tr[k] for k in ("total_return", "ann_return", "max_drawdown", "sharpe", "ann_turnover", "n_trades", "end_equity")},
            "test": {k: te[k] for k in ("total_return", "ann_return", "max_drawdown", "sharpe", "ann_turnover", "n_trades", "end_equity")},
            "fails": fails,
            "passed": len(fails) == 0,
        }
        judgement.append(item)
        if item["passed"]:
            promoted.append(item)

    # Persist tables. Do not touch results/backtest/.
    baseline_df.to_csv(output_dir / "baseline_metrics.csv", index=False)
    all_df.to_csv(output_dir / "all_configs_metrics.csv", index=False)
    train_df.to_csv(output_dir / "train_grid.csv", index=False)
    test_df.to_csv(output_dir / "test_peek.csv", index=False)
    winners_df.to_csv(output_dir / "family_winners.csv", index=False)
    years_df.to_csv(output_dir / "yearly_test_slices.csv", index=False)
    wf_df.to_csv(output_dir / "walkforward_frozen.csv", index=False)
    format_metrics_table(baseline_df).to_csv(output_dir / "baseline_metrics_display.csv", index=False)
    format_metrics_table(winners_df).to_csv(output_dir / "family_winners_display.csv", index=False)

    if promoted:
        optional = []
        for item in promoted:
            rec = rec_by_key[item["key"]]
            if rec["factory"] == "make_vol_target_mix_dual_ma":
                # Reconstructable via a registered factory name the catalog understands.
                # Keep as make_dual_ma_defensive? No — volume scale. Store under a
                # special optional wrapper: we add a thin factory in catalog if needed.
                # For now write the record; catalog will grow a wrapper if promoted.
                optional.append(
                    {
                        "key": rec["key"],
                        "name": rec["name"],
                        "description": rec["description"],
                        "factory": rec["factory"],
                        "params": rec["params"],
                    }
                )
            else:
                optional.append(
                    {
                        "key": rec["key"],
                        "name": rec["name"],
                        "description": rec["description"],
                        "factory": rec["factory"],
                        "params": rec["params"],
                    }
                )
        (output_dir / "optional_strategies.json").write_text(
            json.dumps(optional, ensure_ascii=False, indent=2)
        )
    else:
        # Do not leave a stale promotion file from a previous run.
        path = output_dir / "optional_strategies.json"
        if path.exists():
            path.unlink()

    summary = {
        "train_window": [train_start, train_end],
        "test_window": [test_start, test_end],
        "n_new_configs": n_new,
        "n_prior_baseline_grids": prior_ma_grid + prior_mom_grid,
        "n_families": int(train_df["family"].nunique()),
        "winner_keys": winner_keys,
        "n_promoted": len(promoted),
        "promoted_keys": [p["key"] for p in promoted],
        "bar": {
            "sharpe_edge": SHARPE_EDGE,
            "dd_tolerance": DD_TOLERANCE,
            "ann_ret_tolerance": ANN_RET_TOLERANCE,
            "min_train_sharpe": MIN_TRAIN_SHARPE,
            "year_rule": "at least 2 calendar years in the test window with Sharpe >= BH that year",
        },
        "judgement": judgement,
        "benchmarks": {
            "hs300_buy_hold": {"train": {k: bh_train[k] for k in ("ann_return", "max_drawdown", "sharpe", "ann_turnover", "total_return")},
                               "test": {k: bh_test[k] for k in ("ann_return", "max_drawdown", "sharpe", "ann_turnover", "total_return")}},
            "equal_weight": {"train": {k: ew_train[k] for k in ("ann_return", "max_drawdown", "sharpe", "ann_turnover", "total_return")},
                             "test": {k: ew_test[k] for k in ("ann_return", "max_drawdown", "sharpe", "ann_turnover", "total_return")}},
        },
        "note": "test_peek.csv is diagnostic only and was not used to pick parameters.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str))

    _write_report(
        output_dir=output_dir,
        baseline_df=baseline_df,
        winners_df=winners_df,
        years_df=years_df,
        wf_df=wf_df,
        all_df=all_df,
        summary=summary,
        panel_last=str(last.date()),
    )
    return summary


def _md_table(df: pd.DataFrame, cols: list[str]) -> str:
    show = format_metrics_table(df[cols].copy())
    return show.to_markdown(index=False)


def _write_report(*, output_dir: Path, baseline_df, winners_df, years_df, wf_df, all_df, summary, panel_last: str):
    # Sentiment feasibility is written here as a fixed, already-verified conclusion,
    # plus whatever forward log exists.
    from paperbot.sentiment import status_for_report

    def block(df, keys, period):
        rows = []
        for k in keys:
            sub = df[(df["strategy"] == k) & (df["period"] == period)]
            if sub.empty:
                continue
            rows.append(sub.iloc[0])
        if not rows:
            return "(无)"
        out = pd.DataFrame(rows)
        cols = [
            "strategy",
            "name",
            "ann_return",
            "max_drawdown",
            "sharpe",
            "ann_turnover",
            "total_return",
            "n_trades",
            "end_equity",
        ]
        cols = [c for c in cols if c in out.columns]
        return _md_table(out, cols)

    winner_keys = summary["winner_keys"]
    train_tbl = block(pd.concat([baseline_df, winners_df], ignore_index=True), ["hs300_buy_hold", "equal_weight", "dual_ma", "momentum"] + winner_keys, "train")
    test_tbl = block(pd.concat([baseline_df, winners_df], ignore_index=True), ["hs300_buy_hold", "equal_weight", "dual_ma", "momentum"] + winner_keys, "test")

    # Closest on test among family winners (for honest discussion, not selection).
    test_w = winners_df[winners_df["period"] == "test"].copy()
    test_w = test_w.sort_values(["sharpe", "ann_return"], ascending=False)

    lines = [
        "# 策略改进实验报告",
        "",
        "只做模拟，不接券商、交易所或钱包。费用、滑点、T+1、一手 100 份与原回测相同。",
        "参数只在训练段 2018-01-02 至 2023-12-29 上选择；测试段为 2024-01-02 至 "
        + summary["test_window"][1]
        + "。本报告数字全部由引擎跑出，没有手填。",
        "",
        "## 实验纪律",
        "",
        f"- 本轮新试配置：**{summary['n_new_configs']}** 种，分属 {summary['n_families']} 个策略族。每个族只取训练段（夏普优先，其次年化，再次回撤）最好的一组参数，再看测试段。",
        "- 此前基线网格另有双均线 9 组、动量 3 组（见 `results/backtest/`），不计入本轮 47，但训练段已用过，存在多重检验负担。",
        "- 全部配置的测试段数字写在 `test_peek.csv`，**只作诊断，不参与选参**。",
        "- 预先声明的「明显更好且稳健」门槛（须同时满足）：",
        f"  1. 测试段夏普 ≥ max(沪深300买入持有, 五只等权) + {SHARPE_EDGE:.2f}；",
        f"  2. 测试段最大回撤不比等权差过 {DD_TOLERANCE:.0%}；",
        f"  3. 测试段年化不低于两对照中较弱一方减 {ANN_RET_TOLERANCE:.0%}；",
        f"  4. 训练段夏普 ≥ {MIN_TRAIN_SHARPE:.2f}；",
        "  5. 测试窗内至少两个日历年的夏普不低于当年买入持有，且不出现远差于基准的灾难回撤。",
        "- 即使有策略越过门槛，也只注册为 `run_daily.py` 的可选策略，**不替换默认的五只 ETF 等权**，以免改动每个工作日 16:30 的自动任务。",
        "- 国债 ETF 511010 单价约 141 元，一手约 1.4 万元，10 万元账户上的仓位较粗，这会反映在真实模拟里。",
        "",
        "## 对照与各族训练胜出者：训练段",
        "",
        train_tbl,
        "",
        "## 对照与各族训练胜出者：测试段",
        "",
        test_tbl,
        "",
        "## 推荐结论",
        "",
    ]

    if summary["n_promoted"] == 0:
        lines += [
            "**没有找到在测试段风险调整后、且按预先门槛足够稳健、能明显跑赢沪深300ETF买入持有和五只ETF等权的新策略。**",
            "",
            "下面是各族训练胜出者相对于两对照的差距（测试段），便于看清「接近但不够」还是「明显更差」。",
            "",
        ]
    else:
        names = ", ".join(summary["promoted_keys"])
        lines += [
            f"以下策略越过了预先门槛，已写入 `optional_strategies.json`，可通过把 `paperbot/config.py` 里的 `PAPER_STRATEGY` 设为对应 key 启用。默认仍是 `equal_weight`：{names}",
            "",
        ]

    # Judgement details
    lines.append("| 策略族胜出者 | 训练年化 / 回撤 / 夏普 | 测试年化 / 回撤 / 夏普 | 是否过门槛 | 原因 |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in summary["judgement"]:
        tr, te = item["train"], item["test"]
        reason = "通过" if item["passed"] else "；".join(item["fails"])
        lines.append(
            f"| {item['key']}（{item['name']}） "
            f"| {_pct(tr['ann_return'])} / {_pct(tr['max_drawdown'])} / {_num(tr['sharpe'])} "
            f"| {_pct(te['ann_return'])} / {_pct(te['max_drawdown'])} / {_num(te['sharpe'])} "
            f"| {'是' if item['passed'] else '否'} | {reason} |"
        )

    # Closest discussion
    if not test_w.empty:
        top = test_w.iloc[0]
        lines += [
            "",
            "### 测试段夏普最高的族胜出者（事后观察，不是选参依据）",
            "",
            f"`{top['strategy']}`：年化 {_pct(top['ann_return'])}，最大回撤 {_pct(top['max_drawdown'])}，夏普 {_num(top['sharpe'])}，年化换手 {_pct(top['ann_turnover'])}。",
            "",
        ]

    # Yearly
    lines += [
        "## 测试段内分年（同一段测试模拟的日历年切片；年初不重新注入现金）",
        "",
    ]
    yshow = years_df[years_df["strategy"].isin(["hs300_buy_hold", "equal_weight"] + winner_keys)].copy()
    if not yshow.empty:
        cols = ["strategy", "period", "start", "end", "ann_return", "max_drawdown", "sharpe", "total_return"]
        lines.append(_md_table(yshow, [c for c in cols if c in yshow.columns]))
        lines.append("")

    lines += [
        "## 参数冻结后的按年滚动（每一年从 10 万元现金重新开仓，用训练段选定的参数，不再重估）",
        "",
    ]
    if not wf_df.empty:
        cols = ["strategy", "period", "start", "end", "ann_return", "max_drawdown", "sharpe", "total_return"]
        lines.append(_md_table(wf_df, [c for c in cols if c in wf_df.columns]))
        lines.append("")

    # Peek: best test among ALL configs, with a warning.
    peek = all_df[all_df["period"] == "test"].sort_values(["sharpe", "ann_return"], ascending=False)
    lines += [
        "## 多重检验提醒（测试段偷看，不用于选参）",
        "",
        f"一共跑了 {summary['n_new_configs']} 组新配置的测试段。若直接在测试段里挑夏普最高的一组，很容易把运气当成能力。",
    ]
    if not peek.empty:
        p0 = peek.iloc[0]
        # Was this also the train winner of its family?
        train_same = all_df[(all_df["period"] == "train") & (all_df["strategy"] == p0["strategy"])]
        lines.append(
            f"偷看最高夏普是 `{p0['strategy']}`：测试年化 {_pct(p0['ann_return'])}，回撤 {_pct(p0['max_drawdown'])}，夏普 {_num(p0['sharpe'])}。"
            " 这组如果不是其策略族在训练段的胜出者，更不能当作可上线的策略。"
        )
        lines.append("")

    lines += [
        "## LLM 新闻情绪因子：可行性",
        "",
        "2026-11-22 实际探查 AKShare 1.20.65 的结果：",
        "",
        "- `stock_news_em`（东方财富个股新闻）对 510300 只返回最近约 10 条，时间范围大约是 2026-11-02 至 11-19，是「当日最近 N 条」，不是 2018 年以来的完整历史。",
        "- `stock_news_global_em` 大约最近 200 条，时间只到 2026-11-21 至 11-22。",
        "- 新浪、财联社电报同样只有当天或最近几十条。",
        "- `alternative_news_cctv` 可以按日取到 2016 年以来的新闻联播文本，但一天一次请求、没有现成的市场情绪分；用今天的模型去给旧新闻打分，会带入模型的世界知识，构成未来函数。当前环境也没有配置 LLM API 密钥。",
        "",
        "**结论：现有公开新闻接口不足以对 2018–2023 做可信的情绪因子回测。没有编造历史情绪分。**",
        "",
        "已另外写好只向前记录的框架：`paperbot/sentiment.py`。它每天拉取当前能拿到的标题，写入 `results/sentiment/forward_headlines.csv`，`sentiment_score` 栏留空，并且明确标记 `used_in_trading=no`。以后若有密钥和足够长的前瞻样本，再验证，不把未验证的分数接入交易。",
        "",
        "为避免新闻接口超时或失败影响到每个工作日 16:30 的 `python run_daily.py`，情绪日志**没有**挂进每日任务；需要时单独运行：",
        "",
        "```bash",
        "cd /workspace/trading && .venv/bin/python -c \"from paperbot.sentiment import append_daily_snapshot; append_daily_snapshot()\"",
        "```",
        "",
        status_for_report(),
        "",
        "## 文件与如何切换（尚未改默认）",
        "",
        f"- 本报告：`{output_dir / 'improve_report.md'}`",
        "- 全部训练/测试数字：`all_configs_metrics.csv`、`family_winners.csv`、`yearly_test_slices.csv`、`walkforward_frozen.csv`",
        "- 默认纸面策略仍是 `equal_weight`。`python run_daily.py` 的行为没有改成新策略。",
        "- 若以后要试用已注册的可选策略（仅当本实验有过门槛的策略并生成了 `optional_strategies.json`），把 `paperbot/config.py` 中 `PAPER_STRATEGY` 改成对应 key 即可；请先另开纸面账户或接受持仓会按新目标再平衡。",
        "",
        "## 风险说明",
        "",
        "历史模拟不是对未来收益的承诺。扩展标的（黄金、国债、纳指 QDII）有自身的溢价、外汇和流动性风险；QDII 申购有额度限制，本模拟按二级市场价格成交，未建模溢价大幅偏离。多重检验会让「最好的那一组」看起来好于真实的样本外期望。",
        "",
    ]
    path = output_dir / "improve_report.md"
    path.write_text("\n".join(lines))
    return path
