# -*- coding: utf-8 -*-
"""
lab_factors_fundamental.py  ——  ADD-only Lab_ factor module.

定位
----
- ADD 片段：本文件独立存在，不覆盖 factors.py / config.py。
- 零选股影响：这 7 个因子只供 Factor Lab 的 gate 评估，绝不接入生产选股路径。
- 你的 hfq in-memory join 只作用于"价格/收益"。下面 7 个里：
    * 纯财报项（合同负债/商誉/货币资金/借款/应收/NI/CFO）是绝对金额，不做 hfq；
    * 涉及市值的 yield 类因子用 daily_basic.total_mv（市值，PIT 正确），不用 hfq 后的 close。

本模块替你堵死的三个前视漏洞（fundamental 因子最常见的死法）
----------------------------------------------------------------
1) 报告期 vs 公告日 (PIT)：财报 end_date=2025-03-31，但要到 ~2025-04-底 才公告。
   用 end_date 对齐 = 用了还没披露的数 = 前视。本模块一律按 **f_ann_date / ann_date
   的最早值 (avail_date)** 对齐，merge_asof(direction='backward')，只用"当日已知"的数。
2) YTD 累计 → TTM：A 股利润表/现金流表是年内累计（Q1=3月、H1=6月…）。直接用会把
   Q1 当成"很小"= 季节性伪信号。本模块对 flow 项 (NI / CFO / capex) 统一算 TTM：
   TTM = 上年年报 + 本期累计 − 上年同期累计。
3) 重述泄漏：同一 end_date 可能有原始公告 + 更正公告。本模块按 (ts_code,end_date)
   保留 **最早一次披露 (first disclosure)**，避免用到当时还不存在的更正数。

输入
----
每个因子吃"你从 lake 里加载好的原始 DataFrame"（不替你联网、不假设你的 fetch 层）。
所需 Tushare 接口与字段见 FIELD_MAP。你只需把列名对齐到你 lake 的 schema。

输出
----
统一返回 long 格式 [trade_date, ts_code, <FactorName>]，已 PIT 对齐到你传入的 panel。
（你的 Lab_ harness 若要 wide [date×code]，最后 pivot 一下即可。）

⚠️ 两个你必须核对的假设（我看不到你的 schema，不替它们背书）
    A. FIELD_MAP 里的 Tushare 字段名（尤其 contract_liab / f_ann_date / capex 那几个）。
    B. daily_basic.total_mv 单位是 **万元**，本模块已 *1e4 转成元；若你 lake 已转过，
       把 MV_UNIT_SCALE 改成 1.0。单位错了 yield 会差 1e4 量级。
"""

import numpy as np
import pandas as pd

# ============================================================================
# 0. 字段映射 —— 看不到你的 schema，集中放这里，方便你一处核对/改名
# ============================================================================
FIELD_MAP = {
    # balancesheet (Tushare: balancesheet)
    "contract_liab": "contract_liab",       # 合同负债（新规；旧期可能 NaN）
    "adv_receipts":  "adv_receipts",        # 预收款项（旧期 fallback）
    "goodwill":      "goodwill",            # 商誉
    "money_cap":     "money_cap",           # 货币资金
    "total_assets":  "total_assets",        # 资产总计
    "accounts_recv": "accounts_receiv",     # 应收账款（accruals 子信号，已并入 accruals_cf）
    # 有息负债成分（present 的相加；缺的自动跳过）
    "ib_debt_parts": ["st_borr", "lt_borr", "non_cur_liab_due_1y", "bond_payable"],

    # income (Tushare: income)  —— flow，需 TTM
    "n_income":      "n_income",            # 净利润（用 n_income_attr_p 归母亦可，下面可改）

    # cashflow (Tushare: cashflow) —— flow，需 TTM
    "cfo":           "n_cashflow_act",      # 经营活动现金流净额
    "capex":         "c_pay_acq_const_fiolta",  # 购建固定/无形/长期资产支付现金（capex 代理）

    # daily_basic (Tushare: daily_basic)
    "total_mv":      "total_mv",            # 总市值（万元！见 MV_UNIT_SCALE）

    # stk_holdernumber
    "holder_num":    "holder_num",          # 股东户数

    # stk_holdertrade
    "ht_in_de":      "in_de",               # IN增持 / DE减持
    "ht_vol":        "change_vol",          # 变动数量（股）
    "ht_price":      "avg_price",           # 增减持均价

    # 公告日候选（取最早 = 最保守的可得日）
    "ann_candidates": ["f_ann_date", "ann_date"],
    "end_date":      "end_date",
    "trade_date":    "trade_date",
    "ts_code":       "ts_code",
}

MV_UNIT_SCALE = 1e4          # total_mv 万元 -> 元；你 lake 已是元则改 1.0
INSIDER_WINDOW_DAYS = 365    # 增持回看窗（按"一个披露周期"先验定，写死，不在 gate 上调）
EPS = 1e-9


# ============================================================================
# 1. 通用 PIT 工具（三个漏洞都在这里堵）
# ============================================================================
def _to_dt(s):
    return pd.to_datetime(s, errors="coerce")


def _avail(df):
    """加一列 avail_date = 最早可得公告日（漏洞#1）。"""
    df = df.copy()
    cands = [c for c in FIELD_MAP["ann_candidates"] if c in df.columns]
    if not cands:
        raise KeyError(f"无公告日列，候选={FIELD_MAP['ann_candidates']}，请核对 schema")
    for c in cands:
        df[c] = _to_dt(df[c])
    df["avail_date"] = df[cands].min(axis=1)
    df[FIELD_MAP["end_date"]] = _to_dt(df[FIELD_MAP["end_date"]])
    return df.dropna(subset=["avail_date", FIELD_MAP["end_date"]])


def _first_disclosure(df):
    """同 (ts_code,end_date) 保留最早披露，避免重述泄漏（漏洞#3）。"""
    return (df.sort_values("avail_date")
              .drop_duplicates([FIELD_MAP["ts_code"], FIELD_MAP["end_date"]], keep="first"))


def _prep(df):
    return _first_disclosure(_avail(df))


def _ttm(df, col):
    """YTD 累计 -> TTM（漏洞#2）。TTM = 上年年报 + 本期累计 − 上年同期累计。
    年报行(month==12)自动退化为当年年报值。缺上年数据则 NaN（优雅降级）。"""
    tc, ed = FIELD_MAP["ts_code"], FIELD_MAP["end_date"]
    b = df[[tc, ed, col]].dropna(subset=[ed]).drop_duplicates([tc, ed]).copy()
    b["year"] = b[ed].dt.year
    b["month"] = b[ed].dt.month

    same = b[[tc, "year", "month", col]].rename(columns={col: "_ytd_py"})
    same["year"] = same["year"] + 1                      # 对齐到"今年"
    out = b.merge(same, on=[tc, "year", "month"], how="left")

    ann = b.loc[b["month"] == 12, [tc, "year", col]].rename(columns={col: "_ann_py"})
    ann["year"] = ann["year"] + 1
    out = out.merge(ann, on=[tc, "year"], how="left")

    out["ttm"] = out["_ann_py"] + out[col] - out["_ytd_py"]
    return out[[tc, ed, "ttm"]]


def _yoy_level(df, col):
    """资产负债表 stock 项的同比（同季对同季，处理季节性）。"""
    tc, ed = FIELD_MAP["ts_code"], FIELD_MAP["end_date"]
    b = df[[tc, ed, col]].dropna(subset=[ed]).drop_duplicates([tc, ed]).copy()
    b["year"] = b[ed].dt.year
    b["month"] = b[ed].dt.month
    prev = b[[tc, "year", "month", col]].rename(columns={col: "_py"})
    prev["year"] = prev["year"] + 1
    out = b.merge(prev, on=[tc, "year", "month"], how="left")
    out["yoy"] = out[col] / out["_py"] - 1.0
    out.loc[out["_py"].abs() < EPS, "yoy"] = np.nan       # 防 0/负基数翻转
    return out[[tc, ed, "yoy"]]


def _pit_asof(panel, stmt, value_cols):
    """把 stmt（含 avail_date）PIT 展开到 panel 的每个 (trade_date, ts_code)。
    merge_asof backward：只取 avail_date <= trade_date 的最近一条。"""
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    p = panel[[td, tc]].copy()
    p[td] = _to_dt(p[td])
    p = p.sort_values(td)
    s = stmt[[tc, "avail_date"] + value_cols].dropna(subset=["avail_date"]).sort_values("avail_date")
    out = pd.merge_asof(p, s, left_on=td, right_on="avail_date", by=tc, direction="backward")
    return out[[td, tc] + value_cols]


# 三个 PIT 包装器，让因子本体保持极短 ----------------------------------------
def _pit_level(df, panel, col, name):
    d = _prep(df)
    q = d[[FIELD_MAP["ts_code"], FIELD_MAP["end_date"], "avail_date", col]].rename(columns={col: name})
    return _pit_asof(panel, q, [name])


def _pit_ttm(df, panel, col, name):
    d = _prep(df)
    t = (_ttm(d, col)
         .merge(d[[FIELD_MAP["ts_code"], FIELD_MAP["end_date"], "avail_date"]],
                on=[FIELD_MAP["ts_code"], FIELD_MAP["end_date"]], how="left")
         .rename(columns={"ttm": name}))
    return _pit_asof(panel, t, [name])


def _pit_yoy(df, panel, col, name):
    d = _prep(df)
    y = (_yoy_level(d, col)
         .merge(d[[FIELD_MAP["ts_code"], FIELD_MAP["end_date"], "avail_date"]],
                on=[FIELD_MAP["ts_code"], FIELD_MAP["end_date"]], how="left")
         .rename(columns={"yoy": name}))
    return _pit_asof(panel, y, [name])


def _mktcap(daily_basic, panel):
    """daily_basic 已是日频 PIT，直接按 (trade_date,ts_code) 合并；万元->元。"""
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    db = daily_basic[[tc, FIELD_MAP["trade_date"], FIELD_MAP["total_mv"]]].copy()
    db = db.rename(columns={FIELD_MAP["trade_date"]: td})
    db[td] = _to_dt(db[td])
    db["_mktcap"] = db[FIELD_MAP["total_mv"]] * MV_UNIT_SCALE
    p = panel[[td, tc]].copy()
    p[td] = _to_dt(p[td])
    return p.merge(db[[td, tc, "_mktcap"]], on=[td, tc], how="left")


# ============================================================================
# 2. 七个因子
#    符号约定：一律构造成"**因子值越高 = 预期收益越高**"的假设方向。
#    gate 用 |ICIR|，符号不影响过不过，只影响你读方向；ICIR 正=假设成立。
# ============================================================================
def lab_contract_liab_yoy(balancesheet, panel):
    """[T1] 合同负债同比增速 —— backlog/在手订单代理。高=订单饱满=假设利好。"""
    bs = _avail(balancesheet).copy()
    cl = FIELD_MAP["contract_liab"]
    fb = FIELD_MAP["adv_receipts"]
    bs["_cl"] = bs[cl] if cl in bs.columns else np.nan
    if fb in bs.columns:                                   # 旧期用预收款项兜底
        bs["_cl"] = bs["_cl"].fillna(bs[fb])
    out = _pit_yoy(bs, panel, "_cl", "Lab_contract_liab_yoy")
    return out


def lab_accruals_cf(income, cashflow, balancesheet, panel):
    """[T1] 现金流量表 accruals (Hribar-Collins)：(NI − CFO)/总资产，TTM。
    高 accruals=盈利没有现金支撑=earnings quality 差→收益低。故取负号。
    这一项同时覆盖了 视频里的'应收账款健康'(AR 是 accruals 的成分) 与 '现金流质量'。"""
    ni = _pit_ttm(income,   panel, FIELD_MAP["n_income"], "_ni")
    co = _pit_ttm(cashflow, panel, FIELD_MAP["cfo"],      "_cfo")
    ta = _pit_level(balancesheet, panel, FIELD_MAP["total_assets"], "_ta")  # 注：教科书用平均总资产，这里用 PIT 期末值，足够 rank
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    m = ni.merge(co, on=[td, tc]).merge(ta, on=[td, tc])
    m["Lab_accruals_cf"] = np.where(m["_ta"].abs() > EPS,
                                    -((m["_ni"] - m["_cfo"]) / m["_ta"]), np.nan)
    return m[[td, tc, "Lab_accruals_cf"]]


def lab_fcf_yield(cashflow, daily_basic, panel):
    """[T1] 自由现金流收益率：FCF_ttm / 市值。FCF=CFO−capex。高=便宜=假设利好。
    （现金流的 'yield' 表达；大概率与 value 簇相关，gate 会告诉你。）"""
    co = _pit_ttm(cashflow, panel, FIELD_MAP["cfo"],   "_cfo")
    cx = _pit_ttm(cashflow, panel, FIELD_MAP["capex"], "_capex")
    mc = _mktcap(daily_basic, panel)
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    m = co.merge(cx, on=[td, tc]).merge(mc, on=[td, tc])
    m["_fcf"] = m["_cfo"] - m["_capex"].fillna(0.0)        # capex 缺则按 0（保守，少减）
    m["Lab_fcf_yield"] = np.where(m["_mktcap"].abs() > EPS, m["_fcf"] / m["_mktcap"], np.nan)
    return m[[td, tc, "Lab_fcf_yield"]]


def lab_net_cash(balancesheet, panel):
    """[T1] 净现金/总资产 = (货币资金 − 有息负债)/总资产。高=财务安全=假设利好。
    （低杠杆溢价学术上偏弱/有争议，构造干净，让 gate 裁。）"""
    bs = balancesheet.copy()
    parts = [c for c in FIELD_MAP["ib_debt_parts"] if c in bs.columns]
    bs["_ib_debt"] = bs[parts].fillna(0.0).sum(axis=1) if parts else 0.0
    cash = _pit_level(bs, panel, FIELD_MAP["money_cap"], "_cash")
    debt = _pit_level(bs, panel, "_ib_debt", "_debt")
    ta   = _pit_level(bs, panel, FIELD_MAP["total_assets"], "_ta")
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    m = cash.merge(debt, on=[td, tc]).merge(ta, on=[td, tc])
    m["Lab_net_cash"] = np.where(m["_ta"].abs() > EPS,
                                 (m["_cash"] - m["_debt"]) / m["_ta"], np.nan)
    return m[[td, tc, "Lab_net_cash"]]


def lab_goodwill_ratio(balancesheet, panel):
    """[T2] 商誉/总资产，负向 quality（减值风险）。高商誉=风险=收益假设差→取负。"""
    gw = _pit_level(balancesheet, panel, FIELD_MAP["goodwill"], "_gw")
    ta = _pit_level(balancesheet, panel, FIELD_MAP["total_assets"], "_ta")
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    m = gw.merge(ta, on=[td, tc])
    m["_gw"] = m["_gw"].fillna(0.0)                        # 无并购=商誉 0
    m["Lab_goodwill_ratio"] = np.where(m["_ta"].abs() > EPS, -(m["_gw"] / m["_ta"]), np.nan)
    return m[[td, tc, "Lab_goodwill_ratio"]]


def lab_insider_net_buy(holdertrade, daily_basic, panel):
    """[T2] 内部人净增持金额(回看 365d)/市值。高=内部人看多=假设利好。
    稀疏：多数票当期无交易→净额 0。注意——这是'净增持作为 factor'，
    不是'拿增持价当估值锚'（后者循环，已砍）。"""
    tc, td = FIELD_MAP["ts_code"], "trade_date"
    ht = _avail(holdertrade).copy()                        # avail_date = 公告日
    sign = np.where(ht[FIELD_MAP["ht_in_de"]].astype(str).str.upper().str.startswith("IN"), 1.0, -1.0)
    ht["_amt"] = sign * ht[FIELD_MAP["ht_vol"]].astype(float) * ht[FIELD_MAP["ht_price"]].astype(float)

    # 每 ts_code：按可得日聚合 -> 日历重采样 -> 365D 滚动和 -> asof 到 panel
    daily = (ht.groupby([tc, "avail_date"])["_amt"].sum().reset_index()
               .rename(columns={"avail_date": td}))
    pieces = []
    for code, g in daily.groupby(tc):
        s = g.set_index(td)["_amt"].sort_index()
        s = s.resample("D").sum().fillna(0.0)
        roll = s.rolling(f"{INSIDER_WINDOW_DAYS}D").sum()
        pieces.append(pd.DataFrame({tc: code, td: roll.index, "_net": roll.values}))
    if not pieces:
        # 无任何增减持数据：返回全 NaN，gate 会判 ICIR 不达标
        out = panel[[td, tc]].copy(); out["Lab_insider_net_buy"] = np.nan; return out
    rolled = pd.concat(pieces, ignore_index=True).sort_values(td)

    p = panel[[td, tc]].copy(); p[td] = _to_dt(p[td]); p = p.sort_values(td)
    m = pd.merge_asof(p, rolled.sort_values(td), on=td, by=tc, direction="backward")
    m["_net"] = m["_net"].fillna(0.0)                      # 窗口内无交易 = 0 净额
    mc = _mktcap(daily_basic, panel)
    m = m.merge(mc, on=[td, tc], how="left")
    m["Lab_insider_net_buy"] = np.where(m["_mktcap"].abs() > EPS, m["_net"] / m["_mktcap"], np.nan)
    return m[[td, tc, "Lab_insider_net_buy"]]


def lab_holder_chg(holdernumber, panel):
    """[T3] 股东户数同比变化的负值。户数减少=筹码集中=假设利好→取负。
    注意区分：这是户数'变化'(不含价格)，不是九刀那个含价格的'人均持股金额'。
    已知风险：季度滞后、与价格 reverse causality、可能只是 reversal/size 代理。"""
    y = _pit_yoy(holdernumber, panel, FIELD_MAP["holder_num"], "_hy")
    y["Lab_holder_chg"] = -y["_hy"]
    return y.drop(columns=["_hy"])


# ============================================================================
# 3. 注册表（喂给 run_lab_gate.py）。value=(callable, 需要的原始表名)
# ============================================================================
LAB_FUNDAMENTAL_FACTORS = {
    "Lab_contract_liab_yoy": (lab_contract_liab_yoy, ["balancesheet"]),
    "Lab_accruals_cf":       (lab_accruals_cf,       ["income", "cashflow", "balancesheet"]),
    "Lab_fcf_yield":         (lab_fcf_yield,         ["cashflow", "daily_basic"]),
    "Lab_net_cash":          (lab_net_cash,          ["balancesheet"]),
    "Lab_goodwill_ratio":    (lab_goodwill_ratio,    ["balancesheet"]),
    "Lab_insider_net_buy":   (lab_insider_net_buy,   ["holdertrade", "daily_basic"]),
    "Lab_holder_chg":        (lab_holder_chg,        ["holdernumber"]),
}

# 符号方向备忘（构造方向，非结果）：
#   越高越好(正向假设): contract_liab_yoy, fcf_yield, net_cash, insider_net_buy
#   已取负(原始越高越差): accruals_cf(高accruals差), goodwill_ratio(高商誉差),
#                         holder_chg(户数增=分散差)
