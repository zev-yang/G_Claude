# -*- coding: utf-8 -*-
"""
run_lab_gate_real.py —— 在你的真 lake 上给 7 个 Lab_ 因子评分。

用法：填 3 个 TODO（从你 lake 加载）+ 设 2 个预注册常量，然后
    python run_lab_gate_real.py
读输出时信 IC_t_NW，不是 ICIR_full。把结果回给 Claude 做 survivor triage。
"""
import numpy as np
import pandas as pd
from run_lab_gate import run_gate, forward_return

# ============== 你拥有的预注册决定（按经济先验定，别按哪个好看定）==============
# 潜伏持有期对应的【交易日数】。value 6–18mo 重估 -> 贴近你的持有期。
#   ~3mo≈63   ~6mo≈126   ~12mo≈252
SCORE_HORIZON_DAYS = 126          # TODO 先设这个，否则下面 assert 拦住
# NW 滞后，单位=评估观测(调仓)数。月度调仓 + 多月 horizon 有重叠，
# 设成 ≈ horizon 的月数修残余（6mo→6, 12mo→12）。
NW_LAG_MONTHS      = 6          # TODO
USE_REBALANCE_DATES = True         # 强烈建议 True：只在调仓日评估，去前向收益重叠
# ===========================================================================

# ---- TODO 1：从你 lake 加载六张原始表（你已有 fetch/loader 基建）----
# ⚠️ holdernumber=stk_holdernumber、holdertrade=stk_holdertrade 是【独立 endpoint】。
#    lake 里没有就给 **空 DataFrame**——run_gate 的 per-factor try/except 会接住，
#    只让 Lab_holder_chg / Lab_insider_net_buy 报错行，其余 5 个照跑。
balancesheet = ...   # 列: ts_code,end_date,f_ann_date(或ann_date),contract_liab/adv_receipts,
                     #     goodwill,money_cap,total_assets,st_borr,lt_borr,(non_cur_liab_due_1y,bond_payable)
income       = ...   # ts_code,end_date,f_ann_date,n_income
cashflow     = ...   # ts_code,end_date,f_ann_date,n_cashflow_act,c_pay_acq_const_fiolta
daily_basic  = ...   # ts_code,trade_date,total_mv   (⚠️ total_mv 单位见 lab_factors 的 MV_UNIT_SCALE)
holdernumber = ...   # ts_code,end_date,ann_date,holder_num        | 没有 -> pd.DataFrame()
holdertrade  = ...   # ts_code,f_ann_date,in_de,change_vol,avg_price | 没有 -> pd.DataFrame()

raw = dict(balancesheet=balancesheet, income=income, cashflow=cashflow,
           daily_basic=daily_basic, holdernumber=holdernumber, holdertrade=holdertrade)

# ---- TODO 2：票池 panel + 前向收益 fwd（用 SCORE_HORIZON_DAYS，【不是】V25 的 8）----
panel      = ...     # DataFrame[trade_date, ts_code]：你回测期的票池
close_wide = ...     # DataFrame index=trade_date, columns=ts_code：hfq 后收盘价
assert SCORE_HORIZON_DAYS, "先把 SCORE_HORIZON_DAYS 设成潜伏的持有期(交易日数)"
assert NW_LAG_MONTHS,      "先把 NW_LAG_MONTHS 设成 ≈ horizon 的月数"
fwd = forward_return(close_wide, horizon=SCORE_HORIZON_DAYS)
#   或：你已有 fwd_ret 长表就跳过上面，直接 fwd = your_fwd_long[[trade_date,ts_code,'fwd_ret']]

# ---- TODO 3：已有因子宽表（正交检验；不传则跳过 corr，但那样判不了正交）----
existing_wide = ...  # index=MultiIndex[trade_date,ts_code], columns=已有因子名
                     # 把你 factors.py 的因子矩阵 pivot 成这个形状

# ---- 调仓日：默认每月首个交易日；换成潜伏真实调仓日历更好 ----
rebalance_dates = None
if USE_REBALANCE_DATES:
    td = pd.to_datetime(pd.Series(panel["trade_date"]).drop_duplicates()).sort_values()
    rebalance_dates = td.groupby([td.dt.year, td.dt.month]).first().values
    # 注意：月度调仓 + 多月 horizon 仍重叠 -> 靠 NW(nw_lag) 修残余。
    #       要【彻底】独立就把这里改成 >=horizon 间隔的日子（观测更少，但更诚实）。

# ---- 跑 ----
report = run_gate(raw, panel, fwd,
                  existing_wide=existing_wide,
                  rebalance_dates=rebalance_dates,
                  nw_lag=NW_LAG_MONTHS)
print(report.to_string(index=False))
report.to_csv("lab_gate_report.csv", index=False)
print("\n>>> 读 IC_t_NW，不是 ICIR_full。把 lab_gate_report.csv 回给 Claude 做 survivor triage。")
