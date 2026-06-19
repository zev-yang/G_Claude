# Lab Fundamental Factors — 交接说明

ADD-only。两个文件 + 本说明：
- `lab_factors_fundamental.py` — 7 个 `Lab_` 因子（PIT 无前视）。
- `run_lab_gate.py` — 跑预注册 gate（|ICIR|≥0.25 且 max|corr|<0.60）+ 诚实显著性诊断。
- 零选股影响：只供 gate 评估，不接生产选股。不覆盖 factors.py / config.py。

---

## 1. 我替你堵死的三个前视漏洞（fundamental 因子最常见的死法）
1. **报告期 vs 公告日**：一律按 `f_ann_date/ann_date` 最早值对齐，`merge_asof(backward)`，只用当日已知。
2. **YTD 累计 → TTM**：flow 项(NI/CFO/capex) 统一 `TTM = 上年年报 + 本期累计 − 上年同期累计`，去季节性。
3. **重述泄漏**：同 (ts_code,end_date) 保留最早披露，不用更正后的数。

## 2. ⚠️ gate 怎么读（重要，demo 已暴露）
逐日 ICIR 对**季度更新**的因子会**虚高**：因子一季度才动一次 → 逐日 IC 序列高度自相关
(demo 里 `IC_ac1`≈0.6–0.85) → `std(IC)` 被低估 → ICIR 被放大。**随机噪声都能冲到 |ICIR|≈0.5 且"PASS"。**
你原来的门槛是在 lab_smart_intraday / lab_upper_shadow 这类**日内/日频**因子上校准的（它们每天更新，
IC 自相关低，ICIR 才可信），直接套到慢因子上是错配。

**所以：**
- 别只看 `ICIR_full`，看 **`IC_t_NW`**（Newey-West 调整后的 IC t 值）。
- 但 `NW_LAG=horizon` 会**欠修正**（季度持续性的自相关延伸到 ~60 lag）——demo 里仍有 2/6 随机因子假阳。
- **真正的修法**：传 `rebalance_dates`，只在**潜伏调仓日**评估 IC，去掉前向收益重叠。
  代价是诚实地暴露了独立观测数上限 ≈ 年数×调仓次数——慢因子的统计 power 本就有结构性天花板，造不出来。
- `IC_t_NW`（在调仓频率的 IC 上算）**就是我第一条消息给你的 Fama-MacBeth + Newey-West t 值**。同一把工具，接进了 gate。

> `HORIZON / NW_LAG / rebalance_dates` 都是参数，留给你 pre-register。我**没有**偷改你的 0.25。

## 3. ⚠️ 你必须核对的两个假设（我看不到你的 schema）
- **FIELD_MAP**：Tushare 字段名，尤其 `contract_liab` / `f_ann_date` / `c_pay_acq_const_fiolta`。
- **`total_mv` 单位 = 万元**，代码已 `*1e4` 转元；你 lake 已是元就把 `MV_UNIT_SCALE=1.0`。错了 yield 差 1e4。

## 4. 符号方向（构造方向，非结果。gate 用 |ICIR|，符号只影响读方向）
| 因子 | 经济含义 | 构造（越高越好） |
|---|---|---|
| Lab_contract_liab_yoy | 合同负债同比，backlog | 正 |
| Lab_accruals_cf | (NI−CFO)/资产，盈利现金支撑(含AR+现金流质量) | **取负**(高accruals差) |
| Lab_fcf_yield | FCF/市值，现金流 yield | 正（大概率与 value 相关）|
| Lab_net_cash | (现金−有息负债)/资产，财务安全 | 正 |
| Lab_goodwill_ratio | 商誉/资产，减值风险 | **取负** |
| Lab_insider_net_buy | 内部人净增持(365d)/市值 | 正（稀疏）|
| Lab_holder_chg | 股东户数同比变化，集中度 | **取负**(户数增=分散差)|

## 5. 怎么跑
```python
from run_lab_gate import run_gate, forward_return
raw = dict(balancesheet=..., income=..., cashflow=..., daily_basic=...,
           holdernumber=..., holdertrade=...)         # 从你 lake 加载
panel = ...            # [trade_date, ts_code] 票池
fwd   = forward_return(close_wide, horizon=潜伏的horizon)   # 或直接传你的 fwd_ret 长表
report = run_gate(raw, panel, fwd,
                  existing_wide=你的已有因子宽表,          # index=[trade_date,ts_code]
                  rebalance_dates=潜伏的月度调仓日)         # 强烈建议传
print(report)
```

## 6. 我的预测（**预测，不是结果**——gate 说了算）
- 最可能同时过 ICIR+正交：**Lab_accruals_cf、Lab_contract_liab_yoy**（最正交）。
- 大概率死在 `max|corr|<0.60`：fcf_yield / net_cash（与 value/quality 簇撞）。
- 大概率 ICIR/NW 不达标：insider_net_buy（太稀疏）、holder_chg（弱+reverse causality）。

## 7. 诚实声明
以上代码**未**在你的数据上跑过。任何 ICIR/corr/t 值都由**你**在你的 lake 上跑出来——
我不替你产出回测数字。这是铁律，也是这套装置存在的意义。
