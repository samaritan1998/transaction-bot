"""Chinese daily report for the paper account."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from paperbot import config
from paperbot.engine import metrics_from_equity
from paperbot.llm import status_text


def _pct(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2%}"


def _yuan(x) -> str:
    return f"{x:,.2f}"


def write_daily_report(
    snapshots: pd.DataFrame,
    trades: pd.DataFrame,
    strategy_key: str,
    strategy_name: str,
    asof: pd.Timestamp,
    data_status: dict,
    raw_prices: dict[str, float],
    out_path: Path | None = None,
) -> Path:
    asof = pd.Timestamp(asof).normalize()
    out_path = Path(out_path or config.REPORT_DIR / f"daily_{asof.date()}.md")
    today = snapshots[snapshots["date"] == asof]
    if today.empty:
        raise ValueError(f"no snapshot on {asof.date()}")
    row = today.iloc[-1]
    prev = snapshots[snapshots["date"] < asof]
    prev_eq = float(prev["equity"].iloc[-1]) if not prev.empty else float(config.INITIAL_CASH)
    equity = float(row["equity"])
    cash = float(row["cash"])
    day_ret = equity / prev_eq - 1.0 if prev_eq else 0.0
    positions = json.loads(row["positions_json"] or "{}")
    pending = json.loads(row["pending_json"] or "{}")

    eq_series = snapshots.set_index("date")["equity"].sort_index()
    met = metrics_from_equity(eq_series, trades, strategy_key, "paper")

    day_trades = trades[trades["date"] == asof] if trades is not None and not trades.empty else pd.DataFrame()

    lines = [
        f"# 纸面交易日报 {asof.date()}",
        "",
        "**这是模拟账户，不是实盘。没有连接券商、交易所或钱包，也没有真实成交。**",
        "",
        f"- 策略：{strategy_name}（`{strategy_key}`）",
        f"- 起始资金：{_yuan(config.INITIAL_CASH)} 元（起始日 {config.PAPER_START}）",
        f"- 当日权益：{_yuan(equity)} 元",
        f"- 现金：{_yuan(cash)} 元；持仓市值：{_yuan(float(row['position_value']))} 元",
        f"- 当日涨跌：{_pct(day_ret)}（相对上一交易日权益）",
        f"- 区间累计收益：{_pct(met['total_return'])}；年化 {_pct(met['ann_return'])}；最大回撤 {_pct(met['max_drawdown'])}；夏普 {_num(met['sharpe'])}",
        f"- {status_text()}",
        "",
        "## 当前持仓",
        "",
    ]
    if not positions:
        lines.append("空仓。")
    else:
        lines.append("| 代码 | 名称 | 数量 | 前复权收盘价（元） | 市值（元） |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for code, shares in positions.items():
            px = float(raw_prices.get(code, 0))
            name = config.all_etfs().get(code, {}).get("name", code)
            lines.append(f"| {code} | {name} | {int(shares)} | {px:.3f} | {_yuan(px * int(shares))} |")

    lines += ["", "## 当日模拟成交", ""]
    if day_trades is None or day_trades.empty:
        lines.append("当日无模拟成交。")
    else:
        lines.append("| 代码 | 方向 | 数量 | 滑点后价格 | 成交金额 | 费用合计 |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
        for _, t in day_trades.iterrows():
            lines.append(
                f"| {t['symbol']} | {t['side']} | {int(t['shares'])} | {t['price']:.4f} | {_yuan(t['notional'])} | {_yuan(t['cost'])} |"
            )

    note = str(row["signal_note"]) if "signal_note" in row.index and pd.notna(row["signal_note"]) else ""
    lines += ["", "## 已生成、待下一交易日收盘成交的目标权重", ""]
    if note == "hold":
        lines.append("今日不发出新的调仓指令，维持现有持仓。")
    elif not pending:
        lines.append("目标为空仓。若当前有持仓，将在下一交易日收盘模拟卖出。")
    else:
        for code, w in pending.items():
            name = config.all_etfs().get(code, {}).get("name", code)
            lines.append(f"- {code} {name}：目标权重 {w:.0%}")
        lines.append("")
        lines.append("这些权重会在下一个交易日收盘价上模拟成交，并遵守 T+1 与一手 100 份限制。")

    lines += ["", "## 数据更新", ""]
    if data_status:
        if data_status.get("errors"):
            lines.append("数据源问题：")
            for err in data_status["errors"]:
                lines.append(f"- {err}")
        else:
            lines.append("数据更新未报错。")
        for key, val in (data_status.get("sources") or {}).items():
            lines.append(f"- `{key}`：{val}")
    else:
        lines.append("本次未重新下载，使用本地缓存。")

    lines += [
        "",
        "## 风险提示",
        "",
        "纸面成绩不等于未来实盘成绩。回测和纸面都计入了佣金、印花税、过户费、滑点和 T+1，但仍可能低估冲击成本、停牌或无法成交的情况。本报告不构成投资建议。",
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    # also copy to a stable latest path
    latest = config.REPORT_DIR / "latest.md"
    latest.write_text(out_path.read_text())
    return out_path


def _num(x) -> str:
    if x is None or pd.isna(x):
        return "n/a"
    return f"{x:.2f}"
