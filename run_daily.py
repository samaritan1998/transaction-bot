#!/usr/bin/env python3
"""Run the paper account for any missing trading days and write a Chinese daily report.

Simulation only: no brokers, exchanges, wallets, or live orders.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paperbot import config
from paperbot.data import aligned_close_panel, update_universe
from paperbot.paper import (
    connect,
    load_strategy_from_chosen,
    process_days,
    snapshot_frame,
    trades_frame,
)
from paperbot.report import write_daily_report


def _ensure_backtest() -> None:
    chosen = config.BACKTEST_DIR / "chosen.json"
    if chosen.exists():
        return
    print("未找到回测结果，先运行 run_backtest.py …")
    from paperbot.backtest import run_all

    run_all()


def _export_csv(snapshots, trades) -> None:
    snap_path = config.PAPER_DIR / "snapshots.csv"
    trade_path = config.PAPER_DIR / "trades.csv"
    if snapshots is not None and not snapshots.empty:
        out = snapshots.copy()
        out.to_csv(snap_path, index=False)
    if trades is not None and not trades.empty:
        trades.to_csv(trade_path, index=False)
    else:
        # still write a header so the path exists
        if not trade_path.exists():
            with trade_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        "id",
                        "date",
                        "symbol",
                        "side",
                        "shares",
                        "price",
                        "notional",
                        "commission",
                        "stamp_tax",
                        "transfer_fee",
                        "cost",
                    ]
                )


def main() -> int:
    print("这是 A 股纸面交易程序，只模拟、不实盘。")
    print("更新本地行情缓存…")
    try:
        data_status = update_universe(force=False)
    except Exception as exc:
        print(f"数据更新失败，尝试使用缓存：{type(exc).__name__}: {exc}")
        data_status = {"ok": False, "sources": {}, "errors": [str(exc)]}

    _ensure_backtest()
    key, name, signal_fn = load_strategy_from_chosen()
    print(f"纸面策略：{name}（{key}）")

    panel = aligned_close_panel(adjust="qfq")
    raw_panel = aligned_close_panel(adjust="raw")
    # raw panel may not inner-join the same if dates differ slightly; use qfq for engine.
    conn = connect()
    processed = process_days(conn, panel, raw_panel, signal_fn, key, name)
    snaps = snapshot_frame(conn)
    trades = trades_frame(conn)
    _export_csv(snaps, trades)

    if snaps.empty:
        print("没有可处理的交易日（检查纸面起始日与数据缓存）。")
        return 1

    asof = snaps["date"].max()
    qfq_prices = panel.loc[asof].astype(float).to_dict() if asof in panel.index else {}
    report = write_daily_report(
        snapshots=snaps,
        trades=trades,
        strategy_key=key,
        strategy_name=name,
        asof=asof,
        data_status=data_status,
        raw_prices=qfq_prices,
        out_path=config.REPORT_DIR / f"daily_{asof.date()}.md",
    )
    if processed:
        print(f"新处理 {len(processed)} 个交易日：{processed[0].date()} 至 {processed[-1].date()}")
    else:
        print(f"没有新的交易日，已用最新快照刷新日报（{asof.date()}）。")
    print(f"日报：{report}")
    print(f"最新日报副本：{config.REPORT_DIR / 'latest.md'}")
    print(f"账户库：{config.PAPER_DB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
