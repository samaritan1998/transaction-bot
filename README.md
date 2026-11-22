# A 股 ETF 纸面交易 Bot

这是一个**只做模拟**的研究系统，用来练习 A 股 ETF 策略回测和收盘后纸面记账。不涉及真实资金。

- 不连接任何券商、交易所或钱包
- 不提交任何真实订单
- 不保证赚钱，也不构成投资建议
- 所有回测数字都由程序根据本地缓存的行情计算，没有手写收益

虚拟账户起始资金为 10 万元人民币。建议先看 `交易系统学习方案.md` 里的合规和风险说明。

## 环境

需要 Python 3.11+。在项目根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

虚拟环境只留在本机，不会提交到仓库。

## 一键运行

```bash
# 1. 下载/更新本地行情，并做训练段/测试段回测（参数只在训练段上选）
python run_backtest.py

# 2. 每个交易日收盘后，处理尚未入账的交易日，写中文日报
python run_daily.py
```

第一次运行 `run_daily.py` 时，如果还没有回测结果，它会先跑一遍回测。之后再运行只会补齐新的交易日。

日报路径：

- 当天：`results/reports/daily_YYYY-MM-DD.md`
- 最新副本：`results/reports/latest.md`
- 账户明细写在本机的 `results/paper/paper_account.sqlite`，并导出 `snapshots.csv`、`trades.csv`。这些文件不纳入版本库；仓库里的 `results/paper/` 只保留空目录

回测报告：`results/backtest/backtest_report.md`，数字表：`results/backtest/metrics.csv`。

## 行情与股票池

本地缓存目录是 `data/cache/`。

- 未复权行情：新浪 `ak.fund_etf_hist_sina`
- 前复权行情（用于回测与纸面盈亏）：腾讯 `ak.stock_zh_a_hist_tx(..., adjust="qfq")`
- 沪深 300 价格指数：新浪 `ak.index_zh_daily_sina("sh000300")`

默认不调用东方财富日线接口，因为它很容易限流。池子是流动性较好的 5 只 ETF：

| 代码 | 名称 |
| --- | --- |
| 510300 | 沪深 300ETF |
| 510050 | 上证 50ETF |
| 510500 | 中证 500ETF |
| 159915 | 创业板 ETF |
| 510880 | 红利 ETF |

回测使用**前复权收盘价**，以近似计入分红。纸面账户与回测同一套价格，避免两套口径。在最近一个交易日，前复权收盘价与市价相同。

## 模块说明

| 路径 | 作用 |
| --- | --- |
| `paperbot/config.py` | 标的、费用、训练/测试区间、纸面起始日 |
| `paperbot/data.py` | 下载与本地缓存 |
| `paperbot/costs.py` | 佣金、印花税、过户费、滑点 |
| `paperbot/engine.py` | 共享模拟引擎：T+1、一手 100 份、费用、下一根 K 线收盘成交 |
| `paperbot/strategies.py` | 买入持有、双均线择时、ETF 动量轮动、等权对照 |
| `paperbot/backtest.py` | 训练段选参、测试段评估、按年滚动前推 |
| `paperbot/paper.py` | SQLite 纸面账户 |
| `paperbot/report.py` | 中文日报 |
| `paperbot/llm.py` | 预留的 LLM 新闻判断接口，**默认关闭**，不能下单 |
| `paperbot/catalog.py` | 把策略 key 解析成可调用函数；支持改进实验注册的可选项 |
| `paperbot/improve.py` / `run_improve.py` | 扩展标的与风控方案的训练段选参、测试段评估（结果只写到 `results/improve/`） |
| `paperbot/sentiment.py` | 只向前记录新闻标题的框架，不评分、不接入交易 |

## 策略与回测规则

1. **沪深 300ETF 买入持有**：基准。
2. **双均线择时**：在 510300 上，短均线高于长均线则满仓，否则空仓。快/慢周期只在训练段网格搜索。
3. **ETF 动量轮动**：在五只 ETF 里持有回看期涨幅最高的一只；若最高涨幅为负则空仓。回看期只在训练段选择。
4. **五只 ETF 等权**：每月第一个交易日再平衡到约 20% 权重。不参与网格选参，但是测试段比较时一视同仁。

样本划分（收盘日）：

- 训练段：2018-01-02 至 2023-12-29
- 测试段：2024-01-02 至缓存中最近一个交易日
- 2017 年下半年数据只用于均线预热，不计入收益

成交与费用假设：

- 信号在 T 日收盘后、只用到 T 日及以前的数据
- 在 **T+1 日收盘价** 模拟成交（已经带 10 bp 滑点）
- 当天买入的份额，至少要到下一个交易日才能卖（T+1）
- 数量向下取整到 100 份
- 佣金双边万 2.5，每笔最低 5 元
- 印花税只在卖出收取：2023-08-28 前 0.1%，之后 0.05%
- 过户费：2022-04-29 起双边 0.001%；此前仅沪市 0.002%，深市 0
- 纸面现金不计利息，夏普计算的无风险利率为 0，按 252 个交易日年化

这些假设仍然可能低估大额成交的冲击，也不能处理停牌或涨跌停无法成交。


## 当前纸面策略

每日纸面任务使用 `paperbot/config.py` 里的 `PAPER_STRATEGY = "rp3_lb20"`：沪深300ETF、国债ETF、黄金ETF 按逆波动率加权，不加杠杆。请保持这个值。`run_daily.py` 和 `run_backtest.py` 都只做模拟，不连接券商，也不使用真实资金。

`python run_improve.py` 把对比写到 `results/improve/`，**不会覆盖** `results/backtest/chosen.json`，也不会改 `PAPER_STRATEGY`。研究报告在 `results/improve/improve_report.md`。

`results/improve/optional_strategies.json` 里另外登记了这些研究方案，供对照，不是当前纸面策略：

- `perm_lite`：固定 40% 沪深300 + 40% 国债 + 20% 黄金
- `rp4_lb20`：沪深300 / 国债 / 黄金 / 纳指 ETF 风险平价
- `vol_ew_bond_tv8`：五只 ETF 等权并做目标波动率，剩余进国债

换策略会让已有纸面持仓按新目标再平衡。若要更换，先备份或删除 `results/paper/paper_account.sqlite`，再运行 `python run_daily.py`。

扩展研究池：511010 国债、518880 黄金、513100 纳指，以及缓存中的 159920 / 512010 / 511220。

## LLM 接口

`paperbot/llm.py` 里预留了 `LLMHypothesisLayer`。配置项 `LLM_ENABLED` 默认为 `False`。即使以后打开，它也只能往日报里附加研究假设，**不能修改目标权重，不能下单**。

公开新闻接口目前没有足够长的带时间戳历史，无法做 2018–2023 的可信情绪回测，因此没有编造历史分数。前瞻日志用：

```bash
python -c "from paperbot.sentiment import append_daily_snapshot; append_daily_snapshot()"
```

该命令不在每日任务里自动运行，以免新闻接口失败影响记账。

## 风险提示

- 回测和好的纸面曲线都不等于未来能赚钱。
- 训练段表现好、测试段变差，是常见现象，说明可能过拟合。
- 若测试段还跑不赢沪深 300 买入持有，就不要把这个策略当成「赚钱机器」。
- 作为中国大陆居民，境外加密交易受监管限制，本项目也不提供任何加密或钱包功能。
- 若以后考虑 A 股程序化实盘，必须通过持牌券商，并遵守证监会关于程序化交易「先报告、后交易」的规定。本项目不实现这一步。

## 常见问题

**数据下载失败怎么办？**  
程序会尽量使用 `data/cache/` 里已有的 CSV。把报错原文留下即可。不要短时间反复请求东方财富接口。

**周末或节假日运行？**  
会用最近一个已收盘的交易日刷新日报，不会虚构新的交易日。

**我想换策略或起始资金？**  
改 `paperbot/config.py` 后重新运行回测。若纸面账户已经存在，换策略需要删掉（或备份）`results/paper/paper_account.sqlite` 再运行 `python run_daily.py`。
