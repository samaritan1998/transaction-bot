"""Paper-trading bot configuration. Simulation only: no brokers, wallets, or live orders."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "cache"
RESULT_DIR = ROOT / "results"
BACKTEST_DIR = RESULT_DIR / "backtest"
PAPER_DIR = RESULT_DIR / "paper"
REPORT_DIR = RESULT_DIR / "reports"
IMPROVE_DIR = RESULT_DIR / "improve"
SENTIMENT_DIR = RESULT_DIR / "sentiment"

for _p in (DATA_DIR, BACKTEST_DIR, PAPER_DIR, REPORT_DIR, IMPROVE_DIR, SENTIMENT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# Liquid ETF pool (secondary-market tickers). Histories all cover 2018+.
ETFS = {
    "510300": {"name": "沪深300ETF", "market": "sh", "sina": "sh510300"},
    "510050": {"name": "上证50ETF", "market": "sh", "sina": "sh510050"},
    "510500": {"name": "中证500ETF", "market": "sh", "sina": "sh510500"},
    "159915": {"name": "创业板ETF", "market": "sz", "sina": "sz159915"},
    "510880": {"name": "红利ETF", "market": "sh", "sina": "sh510880"},
}

# Research / optional-strategy universe. Not part of the default equal-weight pool,
# so the daily paper job and original baselines stay unchanged.
EXTENDED_ETFS = {
    "511010": {"name": "国债ETF", "market": "sh", "sina": "sh511010"},
    "518880": {"name": "黄金ETF", "market": "sh", "sina": "sh518880"},
    "513100": {"name": "纳指ETF", "market": "sh", "sina": "sh513100"},
    "159920": {"name": "恒生ETF", "market": "sz", "sina": "sz159920"},
    "512010": {"name": "医药ETF", "market": "sh", "sina": "sh512010"},
    "511220": {"name": "城投债ETF", "market": "sh", "sina": "sh511220"},
}

BENCHMARK_SYMBOL = "510300"
INDEX_SINA = "sh000300"
INDEX_NAME = "沪深300价格指数"

# Backtest window (inclusive). 2017-07 used only as indicator warmup.
WARMUP_START = "2017-07-01"
SAMPLE_START = "2018-01-02"
SAMPLE_END = None  # latest cached trading day
TRAIN_END = "2023-12-29"
TEST_START = "2024-01-02"

# Cost assumptions (A-share secondary market, documented in README).
COMMISSION_RATE = 0.00025  # 万2.5 each side
COMMISSION_MIN = 5.0  # yuan per fill
STAMP_TAX_OLD = 0.001  # sell only, until 2023-08-27
STAMP_TAX_NEW = 0.0005  # sell only, from 2023-08-28
STAMP_TAX_CUT_DATE = "2023-08-28"
TRANSFER_FEE_RATE = 0.00001  # 0.001%, both sides, China Clearing rate since 2022-04-29
TRANSFER_FEE_START = "2022-04-29"
TRANSFER_FEE_SH_OLD = 0.00002  # Shanghai only, before the cut
SLIPPAGE = 0.001  # 10 bps each side, applied to fill price
LOT_SIZE = 100  # one round lot

INITIAL_CASH = 100_000.0
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0  # cash in the paper account earns nothing

# Execution: after close T, signal uses data through T, fill at next close.
FILL_DELAY_BARS = 1

# Paper account
PAPER_DB = PAPER_DIR / "paper_account.sqlite"
PAPER_START = "2026-01-05"  # first trading day processed on a fresh account
PAPER_STRATEGY = "rp3_lb20"  # switched 2026-11-22 per user: stock/bond/gold risk parity

# LLM hypothesis layer is off by default and cannot place orders.
LLM_ENABLED = False

# Parameter grids (fit on the training window only).
MA_FAST_GRID = [5, 10, 20]
MA_SLOW_GRID = [60, 120, 250]
MOMENTUM_LOOKBACK_GRID = [20, 60, 120]
MOMENTUM_REBALANCE_GRID = [20]  # trading days between rebalances
MOMENTUM_ABS_FILTER = True  # go to cash if the leader's lookback return is <= 0


def etf_info(code: str) -> dict:
    if code in ETFS:
        return ETFS[code]
    if code in EXTENDED_ETFS:
        return EXTENDED_ETFS[code]
    raise KeyError(code)


def all_etfs() -> dict:
    out = dict(ETFS)
    out.update(EXTENDED_ETFS)
    return out
