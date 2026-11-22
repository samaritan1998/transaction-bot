"""Download and cache A-share ETF and index daily bars.

Primary source is Sina (raw OHLCV) plus Tencent (forward-adjusted close).
East Money is a last-resort fallback and is not called unless explicitly asked,
because it rate-limits easily.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from paperbot import config

TZ_SHANGHAI = timezone(timedelta(hours=8))


def _now_shanghai() -> datetime:
    return datetime.now(TZ_SHANGHAI)


def _meta_path() -> Path:
    return config.DATA_DIR / "download_meta.json"


def load_meta() -> dict:
    p = _meta_path()
    if p.exists():
        return json.loads(p.read_text())
    return {}


def save_meta(meta: dict) -> None:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _meta_path().write_text(json.dumps(meta, ensure_ascii=False, indent=2))


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["date", "close"]).sort_values("date").drop_duplicates("date")
    return out.reset_index(drop=True)


def fetch_etf_raw_sina(symbol_sina: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.fund_etf_hist_sina(symbol=symbol_sina)
    df = raw.rename(columns=str.lower)
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in df.columns]
    return _normalize_bars(df[keep])


def fetch_etf_qfq_tx(symbol_tx: str, start: str, end: str) -> pd.DataFrame:
    import akshare as ak

    raw = ak.stock_zh_a_hist_tx(symbol=symbol_tx, start_date=start, end_date=end, adjust="qfq")
    df = raw.rename(columns=str.lower)
    keep = [c for c in ("date", "open", "high", "low", "close", "volume", "amount") if c in df.columns]
    return _normalize_bars(df[keep])


def fetch_index_sina(symbol: str = config.INDEX_SINA) -> pd.DataFrame:
    import akshare as ak

    raw = ak.index_zh_daily_sina(symbol=symbol)
    df = raw.rename(columns=str.lower)
    keep = [c for c in ("date", "open", "high", "low", "close", "volume") if c in df.columns]
    return _normalize_bars(df[keep])


def _cache_path(kind: str, code: str) -> Path:
    return config.DATA_DIR / f"{kind}_{code}.csv"



def symbols_needed_for_daily() -> dict:
    """Core five ETFs, plus any extra names required by the *current* paper strategy.

    The default equal-weight job therefore still only refreshes the original pool.
    """
    import json

    out = dict(config.ETFS)
    key = config.PAPER_STRATEGY
    if not key or key in ("auto", "equal_weight", "hs300_buy_hold", "dual_ma", "momentum"):
        return out
    path = config.IMPROVE_DIR / "optional_strategies.json"
    if not path.exists():
        return out
    records = json.loads(path.read_text())
    wanted = []
    for rec in records:
        if rec.get("key") != key:
            continue
        params = rec.get("params") or {}
        if "symbols" in params:
            wanted.extend(params["symbols"] or [])
        for field in ("symbol", "leftover", "defensive"):
            if params.get(field):
                wanted.append(params[field])
        if "weights" in params:
            wanted.extend((params["weights"] or {}).keys())
    for code in wanted:
        if code not in out:
            out[code] = config.etf_info(code)
    return out


def update_universe(force: bool = False, sleep_s: float = 1.2) -> dict:
    """Refresh local caches. Returns a status dict for the daily report."""
    meta = load_meta()
    status = {"ok": True, "sources": {}, "errors": []}
    today = _now_shanghai().strftime("%Y%m%d")
    start = pd.Timestamp(config.WARMUP_START).strftime("%Y%m%d")

    # Index first (Sina).
    idx_path = _cache_path("index", "000300")
    try:
        if force or not idx_path.exists():
            idx = fetch_index_sina()
            _write_csv(idx_path, idx)
            status["sources"]["index_000300"] = f"sina {idx['date'].min().date()}..{idx['date'].max().date()} n={len(idx)}"
        else:
            idx = _normalize_bars(pd.read_csv(idx_path))
            last = idx["date"].max()
            fresh = fetch_index_sina()
            merged = _normalize_bars(pd.concat([idx, fresh], ignore_index=True))
            _write_csv(idx_path, merged)
            status["sources"]["index_000300"] = (
                f"sina cached+refresh {merged['date'].min().date()}..{merged['date'].max().date()} n={len(merged)} prev_last={last.date()}"
            )
        time.sleep(0.3)
    except Exception as exc:
        status["ok"] = False
        status["errors"].append(f"index_000300: {type(exc).__name__}: {exc}")
        if not idx_path.exists():
            raise

    for code, info in symbols_needed_for_daily().items():
        raw_path = _cache_path("raw", code)
        qfq_path = _cache_path("qfq", code)
        try:
            raw = fetch_etf_raw_sina(info["sina"])
            _write_csv(raw_path, raw)
            time.sleep(sleep_s)
            qfq = fetch_etf_qfq_tx(info["sina"], start=start, end=today)
            _write_csv(qfq_path, qfq)
            status["sources"][code] = {
                "raw": f"sina {raw['date'].min().date()}..{raw['date'].max().date()} n={len(raw)} last_close={float(raw['close'].iloc[-1]):.3f}",
                "qfq": f"tx {qfq['date'].min().date()}..{qfq['date'].max().date()} n={len(qfq)} last_close={float(qfq['close'].iloc[-1]):.3f}",
            }
            time.sleep(sleep_s)
        except Exception as exc:
            status["ok"] = False
            msg = f"{code}: {type(exc).__name__}: {exc}"
            status["errors"].append(msg)
            if not (raw_path.exists() and qfq_path.exists()):
                # If one of the two files already exists, keep going with cache.
                if not raw_path.exists() and not qfq_path.exists():
                    continue

    meta["last_update"] = _now_shanghai().isoformat(timespec="seconds")
    meta["status"] = status
    save_meta(meta)
    return status


def load_etf_raw(code: str) -> pd.DataFrame:
    path = _cache_path("raw", code)
    if not path.exists():
        raise FileNotFoundError(f"missing raw cache for {code}: {path}")
    return _normalize_bars(pd.read_csv(path))


def load_etf_qfq(code: str) -> pd.DataFrame:
    path = _cache_path("qfq", code)
    if not path.exists():
        raise FileNotFoundError(f"missing qfq cache for {code}: {path}")
    return _normalize_bars(pd.read_csv(path))


def load_index() -> pd.DataFrame:
    path = _cache_path("index", "000300")
    if not path.exists():
        raise FileNotFoundError(f"missing index cache: {path}")
    return _normalize_bars(pd.read_csv(path))


def aligned_close_panel(adjust: str = "qfq", symbols: list[str] | None = None) -> pd.DataFrame:
    """Wide close-price panel on the 510300 trading calendar.

    Isolated missing bars (for example 159915 on 2021-02-08, absent in both
    Sina and Tencent) are forward-filled from the previous close and recorded
    in ``download_meta.json`` under ``filled_bars``.
    Default ``symbols`` is the original five-ETF pool so the daily job is unchanged.
    """
    symbols = list(symbols or config.ETFS.keys())
    if config.BENCHMARK_SYMBOL not in symbols:
        # Always load the calendar ETF, then drop it if the caller did not ask.
        calendar_needed = True
        load_codes = [config.BENCHMARK_SYMBOL] + symbols
    else:
        calendar_needed = False
        load_codes = symbols
    frames = {}
    for code in load_codes:
        df = load_etf_qfq(code) if adjust == "qfq" else load_etf_raw(code)
        frames[code] = df.set_index("date")["close"].rename(code)
    calendar = frames[config.BENCHMARK_SYMBOL].index
    start = pd.Timestamp(config.WARMUP_START)
    end = pd.Timestamp(config.SAMPLE_END) if config.SAMPLE_END else calendar.max()
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    panel = pd.concat(frames.values(), axis=1).sort_index().reindex(calendar)
    filled = {}
    for col in panel.columns:
        missing = panel.index[panel[col].isna()]
        if len(missing):
            filled[col] = [str(d.date()) for d in missing]
    if filled:
        panel = panel.ffill()
        still = panel.columns[panel.isna().any()].tolist()
        if still:
            raise ValueError(f"NaNs remain after forward-fill: {still}")
        meta = load_meta()
        key = f"{adjust}:{','.join(symbols)}"
        meta.setdefault("filled_bars", {})[key] = filled
        save_meta(meta)
    if calendar_needed and config.BENCHMARK_SYMBOL not in symbols:
        panel = panel[symbols]
    else:
        panel = panel[symbols]
    return panel


def raw_close_on(code: str, day) -> float:
    df = load_etf_raw(code)
    row = df[df["date"] == pd.Timestamp(day).normalize()]
    if row.empty:
        raise KeyError(f"no raw close for {code} on {day}")
    return float(row["close"].iloc[0])
