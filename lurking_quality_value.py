# -*- coding: utf-8 -*-
"""
lurking_quality_value.py — 潜伏模式 第2块: 质量过滤(Layer1) + 深度价值(Layer2).

依赖地基:
  · lurking_fundamentals.pit_fundamentals  -> 时点对齐(ann_date) 的财务指标 (roe/roa/负债率...)
  · daily_basic 分片 (共享 V25 的)          -> pe_ttm / pb / circ_mv (估值分位 + 市值)

设计要点(均为潜伏模式纪律):
  1) 所有判断只用【时点可得】数据 —— 财务走 ann_date as-of, 估值用当日及之前的历史分位。
  2) 质量过滤产出布尔 mask (优质池); 后续因子只在池内计算/排序。
  3) 估值分位 = 个股自身 PE/PB 在过去 ~5 年(1250 交易日)的滚动百分位, 极值截尾, 剔负 PE。
     用【个股纵向历史分位】而非横截面 —— "便宜"指相对它自己的历史便宜, 更稳健。
  4) daily_basic 缺失/历史不足时【优雅降级】: 估值分位返回 NaN, 不杀掉整个流程。

接口:
  quality_mask(fund_panel)                  -> Series[bool], index=(date,code)
  valuation_score(daily_basic_panel)        -> Series[float] 0~1, index=(date,code)
  其中 fund_panel 来自 pit_fundamentals; daily_basic_panel 来自 load_daily_basic_ts (下)。
"""
import os
import glob

import numpy as np
import pandas as pd

_ENGINE = 'fastparquet'
DB_SRC_DEFAULT = 'tushare_cache/_partial/daily_basic'

# ── 质量过滤阈值 (写死, 不在回测上调; 来自方案 + A股常识) ───────────────────────────
Q_ROE_MIN      = 8.0     # ROE > 8%  (放宽方案的10, A股优质股门槛)
Q_DEBT_MAX     = 70.0    # 资产负债率 < 70%
Q_OCF_MIN      = 0.0     # 经营现金流(ocfps) > 0 (利润含金量的下限)
Q_NPROF_YOY_MIN = -20.0  # 净利润增速 > -20% (允许周期波动, 但排除崩塌)


def load_daily_basic_ts(src=DB_SRC_DEFAULT):
    """读 daily_basic 分片 -> (date, code) 长表, 列含 pe_ttm/pb/circ_mv。缺失则抛 FileNotFoundError。"""
    files = sorted(glob.glob(os.path.join(src, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"no daily_basic parquet in {src}")
    cols = None
    frames = []
    for f in files:
        try:
            d = pd.read_parquet(f, engine=_ENGINE)
        except Exception:
            continue
        frames.append(d)
    db = pd.concat(frames, ignore_index=True)
    db['date'] = pd.to_datetime(db['trade_date'].astype(str), format='%Y%m%d')
    db['code'] = db['ts_code'].astype(str).str[:6]
    for c in ('pe_ttm', 'pb', 'circ_mv'):
        if c in db.columns:
            db[c] = pd.to_numeric(db[c], errors='coerce')
    keep = ['date', 'code'] + [c for c in ('pe_ttm', 'pb', 'circ_mv') if c in db.columns]
    return db[keep].sort_values(['code', 'date']).set_index(['date', 'code'])


def quality_mask(fund_panel):
    """Layer1 质量过滤 -> 优质池布尔 mask。NaN(财报未公告) -> False (不可得即不入池)。"""
    f = fund_panel
    roe   = f.get('roe')
    debt  = f.get('debt_to_assets')
    ocf   = f.get('ocfps')
    npyoy = f.get('netprofit_yoy')
    mask = pd.Series(True, index=f.index)
    if roe is not None:   mask &= (roe > Q_ROE_MIN)
    if debt is not None:  mask &= (debt < Q_DEBT_MAX)
    if ocf is not None:   mask &= (ocf > Q_OCF_MIN)
    if npyoy is not None: mask &= (npyoy > Q_NPROF_YOY_MIN)
    return mask.fillna(False)


def _rolling_pct_rank(s, window=1250, min_periods=250):
    """个股自身值在过去 window 天里的分位 (0~1); 当前值的历史百分位。min_periods 前为 NaN。"""
    # 当前观测在滚动窗口内的排名百分位: rank of last element among the window
    def _last_pct(x):
        last = x[-1]
        valid = x[~np.isnan(x)]
        if len(valid) < min_periods or np.isnan(last):
            return np.nan
        return (valid <= last).mean()
    return s.rolling(window, min_periods=min_periods).apply(_last_pct, raw=True)


def valuation_score(db_panel, pe_window=1250, pb_window=1250):
    """Layer2 深度价值 -> 0~1 (越高越低估)。

    pe_pct/pb_pct = 个股 PE/PB 的 ~5年滚动历史分位 (剔负PE, 极值截尾)。
    score = 0.5*(1-pe_pct) + 0.5*(1-pb_pct)。某项缺失则用另一项; 都缺则 NaN。
    """
    db = db_panel.sort_index()
    inv = {}
    for col, win in (('pe_ttm', pe_window), ('pb', pb_window)):
        if col not in db.columns:
            continue
        s = db[col].copy()
        if col == 'pe_ttm':
            s = s.where(s > 0)                          # 剔除负PE(亏损)
        lo, hi = s.quantile(0.01), s.quantile(0.99)     # 极值截尾
        s = s.clip(lo, hi)
        # 逐股滚动历史分位; 用 numpy 数组运算避开 MultiIndex 对齐, 结果按原 index 拼回
        parts = []
        for code, g in s.groupby(level='code', sort=False):
            vals = g.to_numpy()
            pct = _rolling_pct_rank(pd.Series(vals), win).to_numpy()
            parts.append(pd.Series(1.0 - pct, index=g.index))
        inv[col] = pd.concat(parts)
    if not inv:
        return pd.Series(np.nan, index=db_panel.index, name='valuation_score')
    inv_df = pd.DataFrame(inv).reindex(db_panel.index)
    return inv_df.mean(axis=1).rename('valuation_score').astype('float32')
