#!/usr/bin/env python3
"""Run phase-1 baseline backtests. Simulation only."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paperbot import config
from paperbot.data import update_universe
from paperbot.backtest import format_metrics_table, run_all


def main() -> int:
    print("更新本地行情缓存（新浪 + 腾讯，不调用券商）…")
    status = update_universe(force=False)
    if status["errors"]:
        print("数据源警告：")
        for err in status["errors"]:
            print(" -", err)
        if not status["ok"]:
            print("将尽量使用已有缓存继续。")
    for key, val in status.get("sources", {}).items():
        print(f"  {key}: {val}")

    print("开始回测（参数只在训练段上选择）…")
    out = run_all()
    metrics = out["metrics"]
    print("\n回测结果：")
    print(format_metrics_table(metrics).to_string(index=False))
    best = out["chosen"]["best_on_test"]
    print(f"\n测试段表现最好的策略：{best['name']}（{best['key']}）")
    print(f"报告：{config.BACKTEST_DIR / 'backtest_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
