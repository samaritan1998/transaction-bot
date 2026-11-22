#!/usr/bin/env python3
"""Run the strategy-improvement study. Simulation only. Does not change the daily default."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paperbot.improve import run_improve
from paperbot.sentiment import append_daily_snapshot


def main() -> int:
    print("策略改进实验：只模拟，不实盘。结果写到 results/improve/，不覆盖回测 chosen.json。")
    try:
        path = append_daily_snapshot()
        print(f"已写入前瞻新闻日志：{path}")
    except Exception as exc:
        print(f"新闻前瞻日志未能写入（不影响回测）：{type(exc).__name__}: {exc}")
    summary = run_improve()
    print(f"新配置数量：{summary['n_new_configs']}，过门槛：{summary['n_promoted']} -> {summary['promoted_keys']}")
    print("报告：results/improve/improve_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
