# -*- coding: utf-8 -*-
"""
run_lurking.py — 潜伏模式【总装线】. 一键运行: 读共享数据 -> 四层因子 -> 合成中性化 -> 月度回测.

与 V25 完全隔离: 不 import V25 的 factors/backtest, 不改 V25 任何文件; 只【只读】共享数据湖
  (日线/复权/moneyflow/融资融券) + 潜伏专属数据 (财务/行业)。

流程:
  1. load_universe_audit  -> hfq 日线 panel (复用 V25 的 data_loader, 只读)
  2. 截取回测区间 (默认 2018+, 受财务数据起点限制)
  3. 在交易日×股票网格上:
       Layer1 quality_mask         (pit 财务)
       Layer2 valuation_score      (daily_basic 5年分位)
       Layer3 mispricing_score     (回撤 + moneyflow主力 + 融资余额, 条件门)
       Layer4 exhaustion_score     (RSI/缩量/ATR + 资金底部)
  4. Layer5 hidden_alpha 合成 -> 行业市值中性化
  5. lurking_backtest 月度调仓回测 -> 3/6/12月 全样本+留出OOS

运行: python run_lurking.py
"""
import sys
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit

from lurking_fundamentals import pit_fundamentals
from lurking_quality_value import quality_mask, valuation_score, load_daily_basic_ts
from lurking_mispricing import mispricing_score
from lurking_exhaustion import exhaustion_score
from lurking_synthesis import hidden_alpha, neutralize
from lurking_backtest import backtest

import moneyflow_factors as MF
import margin_factors as MG
from fetch_industry import load_industry

START = '20180101'          # 财务数据起点; 估值分位在此之前不足5年但 min_periods=250 仍可用
FUND_FIELDS_ROE = 'roe'     # 质量软分用


def _section(msg):
    print(f"\n{'='*78}\n>>> {msg}")


def main():
    # ── 1. 共享日线湖 (hfq) ──
    _section("1/5 加载共享日线湖 (hfq, 只读 — 与 V25 共用)")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    panel = panel[panel.index.get_level_values('date') >= pd.Timestamp(START)]
    dates = panel.index.get_level_values('date').unique().sort_values()
    codes = panel.index.get_level_values('code').unique().tolist()
    print(f"  回测区间: {dates.min().date()} ~ {dates.max().date()} | "
          f"{len(dates)} 交易日 | {len(codes)} 股票")
    close = panel['close']                          # hfq close
    high, low, vol = panel['high'], panel['low'], panel['volume']

    # ── 2. 时点对齐财务 -> Layer1 质量 ──
    _section("2/5 财务时点对齐 (ann_date) + Layer1 质量过滤")
    fund = pit_fundamentals(dates, codes, src=CONFIG.get('fundamentals_q_path',
                            './tushare_cache/_partial/fundamentals'))
    fund = fund.reindex(panel.index)               # 对齐到日线网格
    q_mask = quality_mask(fund)
    print(f"  质量池: 平均每日 {q_mask.groupby(level='date').sum().mean():.0f} 只入池")

    # ── 3. 估值 / 错杀 / 衰竭 ──
    _section("3/5 Layer2 估值分位 + Layer3 错杀验证 + Layer4 衰竭确认")
    # 估值: daily_basic 5年分位
    try:
        db = load_daily_basic_ts(CONFIG.get('fundamentals_path',
                                 './tushare_cache/_partial/daily_basic'))
        db = db.reindex(panel.index)
        val = valuation_score(db)
        lncap = np.log(db['circ_mv'].where(db['circ_mv'] > 0)).reindex(panel.index)
    except FileNotFoundError:
        print("  ⚠️ daily_basic 缺失 -> 估值层 NaN, 市值中性化降级")
        val = pd.Series(np.nan, index=panel.index)
        lncap = pd.Series(np.nan, index=panel.index)

    # 资金面板 (只读共享数据; 缺失则降级)
    mf_raw_panel, mf_main_daily = None, None
    try:
        mf_raw = MF.load_moneyflow(CONFIG['moneyflow_path'])
        mf_raw['date'] = pd.to_datetime(mf_raw['trade_date'].astype(str), format='%Y%m%d')
        mf_raw['code'] = mf_raw['ts_code'].astype(str).str[:6]
        mf_raw_panel = mf_raw.set_index(['date', 'code']).sort_index()
        # Layer4 用日频主力净流入(正/负)数连续流入天数
        mf_main_daily = ((mf_raw_panel['buy_lg_amount'] + mf_raw_panel['buy_elg_amount'])
                         - (mf_raw_panel['sell_lg_amount'] + mf_raw_panel['sell_elg_amount']))
    except Exception as e:
        print(f"  ⚠️ moneyflow 加载失败 ({e!r}) -> 资金子项降级")

    try:
        marg = MG.margin_panel(CONFIG.get('margin_detail_path',
                               './tushare_cache/_partial/margin_detail'))
    except Exception as e:
        print(f"  ⚠️ margin 加载失败 ({e!r}) -> 杠杆子项降级")
        marg = None

    mis = mispricing_score(close, margin_panel=marg, mf_raw_panel=mf_raw_panel)
    exh = exhaustion_score(close, high, low, vol, margin_panel=marg, mf_main_net=mf_main_daily)

    # ── 4. 合成 + 行业市值中性化 ──
    _section("4/5 Layer5 hidden_alpha 合成 + 行业市值中性化 (选股前)")
    roe = fund['roe'] if 'roe' in fund.columns else pd.Series(np.nan, index=panel.index)
    alpha = hidden_alpha(q_mask, val, mis, exh, roe)
    try:
        ind_map = load_industry(CONFIG.get('industry_path',
                                './tushare_cache/_partial/industry'))
        ind_dict = ind_map.to_dict()
    except FileNotFoundError:
        print("  ⚠️ 行业数据缺失 -> 仅做市值中性化")
        ind_dict = {}
    alpha_neut = neutralize(alpha, ind_dict, lncap)
    print(f"  中性化后可选股票: 平均每日 {alpha_neut.groupby(level='date').count().mean():.0f} 只")

    # ── 5. 月度调仓回测 ──
    _section("5/5 月度调仓回测 (3/6/12月, 自带留出期 OOS)")
    backtest(close, alpha_neut, top_n=CONFIG.get('lurking_top_n', 30), verbose=True)

    print(f"\n{'='*78}\n潜伏模式运行完毕。判读: 看留出期 OOS 是否为正且 3/6/12月方向稳健。")


if __name__ == '__main__':
    main()
