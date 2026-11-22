"""Forward-only news snapshot for a future sentiment study.

AKShare news endpoints checked on 2026-11-22 only return the most recent
items (often the last one or two days, at most a few weeks for a single
ticker). That is not enough history to backtest an LLM or news factor
from 2018-2023 without look-ahead. CCTV transcripts exist back to 2016,
but scoring them would need an LLM API key and would still not be a
contemporaneous market-sentiment series.

This module therefore only records today's (or the latest available)
headlines with their original timestamps. It does not score them and
does not change any trade.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from paperbot import config

TZ = timezone(timedelta(hours=8))


def _log_path() -> Path:
    return config.SENTIMENT_DIR / "forward_headlines.csv"


def fetch_latest_headlines(limit: int = 40) -> pd.DataFrame:
    """Pull whatever short-window headlines AKShare will give us now."""
    import akshare as ak

    rows = []
    # Global East Money flash news (~200 most recent).
    try:
        df = ak.stock_news_global_em()
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": "stock_news_global_em",
                    "published_at": str(r.get("发布时间", "")),
                    "title": str(r.get("标题", "")),
                    "summary": str(r.get("摘要", ""))[:500],
                }
            )
    except Exception as exc:
        rows.append(
            {
                "source": "stock_news_global_em",
                "published_at": "",
                "title": f"[fetch_failed] {type(exc).__name__}: {exc}",
                "summary": "",
            }
        )
    # Cailian Press telegraph, last ~20 items.
    try:
        df = ak.stock_news_global_cls(symbol="全部")
        for _, r in df.iterrows():
            rows.append(
                {
                    "source": "stock_news_global_cls",
                    "published_at": f"{r.get('发布日期', '')} {r.get('发布时间', '')}".strip(),
                    "title": str(r.get("标题", "")),
                    "summary": str(r.get("内容", ""))[:500],
                }
            )
    except Exception as exc:
        rows.append(
            {
                "source": "stock_news_global_cls",
                "published_at": "",
                "title": f"[fetch_failed] {type(exc).__name__}: {exc}",
                "summary": "",
            }
        )
    out = pd.DataFrame(rows)
    if len(out) > limit * 2:
        out = out.head(limit * 2)
    return out


def append_daily_snapshot(asof: pd.Timestamp | None = None) -> Path:
    """Append a dated snapshot. sentiment_score is left blank on purpose."""
    asof = pd.Timestamp(asof or datetime.now(TZ)).normalize()
    collected_at = datetime.now(TZ).isoformat(timespec="seconds")
    headlines = fetch_latest_headlines()
    headlines.insert(0, "asof_date", str(asof.date()))
    headlines.insert(1, "collected_at", collected_at)
    headlines["sentiment_score"] = ""  # reserved; no API key, do not invent
    headlines["used_in_trading"] = "no"
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        # Do not duplicate the same asof+source+title.
        key = ["asof_date", "source", "title"]
        merged = pd.concat([old, headlines], ignore_index=True)
        merged = merged.drop_duplicates(subset=key, keep="first")
        merged.to_csv(path, index=False)
    else:
        headlines.to_csv(path, index=False)
    note = {
        "asof": str(asof.date()),
        "collected_at": collected_at,
        "n_rows_written": int(len(headlines)),
        "path": str(path),
        "note": "Scores are intentionally empty. This log is not a trading input.",
    }
    (config.SENTIMENT_DIR / "last_snapshot.json").write_text(
        json.dumps(note, ensure_ascii=False, indent=2)
    )
    return path


def status_for_report() -> str:
    path = _log_path()
    if not path.exists():
        return (
            "新闻情绪：已确认 AKShare 公开新闻接口没有足够长的带时间戳历史，"
            "无法做 2018–2023 的可信回测；今日未写入前瞻日志。"
        )
    df = pd.read_csv(path)
    n = int(df["asof_date"].nunique()) if "asof_date" in df.columns else 0
    return (
        f"新闻情绪：仅前瞻记录标题，不评分、不下单。"
        f"日志 {path}，已覆盖 {n} 个日期。历史样本不足，尚未纳入任何策略。"
    )
