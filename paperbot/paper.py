"""Paper account persisted in SQLite. Simulation only."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd

from paperbot import config
from paperbot.engine import Account, Lot, execute_target, shares_for_weight
from paperbot.llm import LLMHypothesisLayer, status_text


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    date TEXT PRIMARY KEY,
    cash REAL NOT NULL,
    equity REAL NOT NULL,
    position_value REAL NOT NULL,
    positions_json TEXT NOT NULL,
    pending_json TEXT NOT NULL,
    signal_note TEXT
);
CREATE TABLE IF NOT EXISTS lots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    shares INTEGER NOT NULL,
    acquire_date TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    shares INTEGER NOT NULL,
    price REAL NOT NULL,
    notional REAL NOT NULL,
    commission REAL NOT NULL,
    stamp_tax REAL NOT NULL,
    transfer_fee REAL NOT NULL,
    cost REAL NOT NULL
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = Path(path or config.PAPER_DB)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def _get_meta(conn: sqlite3.Connection, key: str, default=None):
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row["value"] if row else default


def _set_meta(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def init_account(conn: sqlite3.Connection, strategy_key: str, strategy_name: str) -> None:
    if _get_meta(conn, "initialized"):
        return
    _set_meta(conn, "initialized", "1")
    _set_meta(conn, "strategy_key", strategy_key)
    _set_meta(conn, "strategy_name", strategy_name)
    _set_meta(conn, "initial_cash", config.INITIAL_CASH)
    _set_meta(conn, "paper_start", config.PAPER_START)
    _set_meta(conn, "pending_json", "{}")
    conn.commit()


def load_account(conn: sqlite3.Connection) -> Account:
    last = conn.execute("SELECT cash FROM snapshots ORDER BY date DESC LIMIT 1").fetchone()
    cash = float(last["cash"]) if last else float(config.INITIAL_CASH)
    account = Account(cash=cash, lots={})
    for row in conn.execute("SELECT symbol, shares, acquire_date FROM lots WHERE shares>0"):
        account.lots.setdefault(row["symbol"], []).append(
            Lot(shares=int(row["shares"]), acquire_date=pd.Timestamp(row["acquire_date"]))
        )
    return account


def save_lots(conn: sqlite3.Connection, account: Account) -> None:
    conn.execute("DELETE FROM lots")
    for symbol, lots in account.lots.items():
        for lot in lots:
            if lot.shares > 0:
                conn.execute(
                    "INSERT INTO lots(symbol, shares, acquire_date) VALUES(?,?,?)",
                    (symbol, int(lot.shares), str(pd.Timestamp(lot.acquire_date).date())),
                )


def last_snapshot_date(conn: sqlite3.Connection) -> pd.Timestamp | None:
    row = conn.execute("SELECT date FROM snapshots ORDER BY date DESC LIMIT 1").fetchone()
    return pd.Timestamp(row["date"]) if row else None


def pending_weights(conn: sqlite3.Connection) -> dict[str, float]:
    raw = _get_meta(conn, "pending_json", "{}")
    data = json.loads(raw or "{}")
    return {k: float(v) for k, v in data.items()}


def process_days(
    conn: sqlite3.Connection,
    panel: pd.DataFrame,
    raw_panel: pd.DataFrame,
    signal_fn,
    strategy_key: str,
    strategy_name: str,
) -> list[pd.Timestamp]:
    """Replay any trading days from paper start (or last snapshot+1) through last close."""
    init_account(conn, strategy_key, strategy_name)
    last = last_snapshot_date(conn)
    start = pd.Timestamp(config.PAPER_START)
    if last is not None:
        # Continue from the next bar after the last snapshot.
        later = panel.index[panel.index > last]
        if later.empty:
            return []
        days = later
    else:
        days = panel.index[panel.index >= start]
    processed = []
    account = load_account(conn)
    pending = pending_weights(conn)
    has_order = _get_meta(conn, "has_order", "0") == "1"

    for day in days:
        qfq_prices = panel.loc[day].astype(float).to_dict()
        # Fills and marks use forward-adjusted prices so P&L is consistent
        # with the backtest. On the latest day, qfq close equals the raw close.
        # has_order=True even when pending=={} means "go to cash".
        if has_order:
            equity_before = account.equity(qfq_prices)
            target_shares = {
                s: shares_for_weight(equity_before, w, qfq_prices[s]) for s, w in pending.items()
            }
            for s in list(account.lots):
                target_shares.setdefault(s, 0)
            fills = execute_target(account, target_shares, qfq_prices, day)
            for t in fills:
                conn.execute(
                    """INSERT INTO trades(date, symbol, side, shares, price, notional,
                       commission, stamp_tax, transfer_fee, cost)
                       VALUES(?,?,?,?,?,?,?,?,?,?)""",
                    (
                        str(t.date.date()),
                        t.symbol,
                        t.side,
                        t.shares,
                        t.price,
                        t.notional,
                        t.commission,
                        t.stamp_tax,
                        t.transfer_fee,
                        t.cost,
                    ),
                )
            has_order = False
            pending = {}

        equity = account.equity(qfq_prices)
        pos = {s: account.shares_of(s) for s in panel.columns if account.shares_of(s)}
        hist = panel.loc[:day]
        weights = signal_fn(hist, day)
        if weights is None:
            has_order = False
            note = "hold"
        else:
            pending = {s: float(w) for s, w in weights.items() if float(w) != 0.0}
            has_order = True
            note = "order"

        conn.execute(
            """INSERT INTO snapshots(date, cash, equity, position_value, positions_json, pending_json, signal_note)
               VALUES(?,?,?,?,?,?,?)
               ON CONFLICT(date) DO UPDATE SET
                 cash=excluded.cash, equity=excluded.equity,
                 position_value=excluded.position_value,
                 positions_json=excluded.positions_json,
                 pending_json=excluded.pending_json,
                 signal_note=excluded.signal_note
            """,
            (
                str(day.date()),
                account.cash,
                equity,
                account.position_value(qfq_prices),
                json.dumps(pos, ensure_ascii=False),
                json.dumps(pending, ensure_ascii=False),
                note,
            ),
        )
        save_lots(conn, account)
        _set_meta(conn, "pending_json", json.dumps(pending, ensure_ascii=False))
        _set_meta(conn, "has_order", "1" if has_order else "0")
        processed.append(day)

    conn.commit()
    return processed


def snapshot_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT date, cash, equity, position_value, positions_json, pending_json, signal_note FROM snapshots ORDER BY date"
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def trades_frame(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute("SELECT * FROM trades ORDER BY date, id").fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_strategy_from_chosen() -> tuple[str, str, object]:
    from paperbot.catalog import resolve_strategy

    path = config.BACKTEST_DIR / "chosen.json"
    if not path.exists():
        raise FileNotFoundError("尚未找到回测结果 results/backtest/chosen.json，请先运行 python run_backtest.py")
    spec = resolve_strategy(config.PAPER_STRATEGY)
    return spec.key, spec.name, spec.signal_fn
