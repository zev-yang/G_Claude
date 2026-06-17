# -*- coding: utf-8 -*-
"""
lurking_fundamentals.py — 财务数据的【时点对齐】加载器 (潜伏模式).

fetcher 负责"存对" (ann_date), 这里负责"用对" (as-of join):
  对任意交易日 D 和股票 S, 只能取到【ann_date <= D 的、最近一期】财务数据。
  这就是 point-in-time / as-of 对齐 —— 回测里任何一天都只看见那天之前已公告的财报,
  彻底杜绝未来函数。用 pandas merge_asof (按时间向后匹配) 实现, 是业界标准做法。

接口:
  load_fundamentals_quarterly(src)         -> 原始 (ts_code, ann_date, end_date, 指标...) 长表
  pit_fundamentals(trade_dates, codes, src)-> 在给定交易日×股票网格上 as-of 对齐后的 panel
                                              index=(date, code), 列=财务指标 + _ann_date/_end_date
注意:
  · 任何股票在其首份财报公告日之前, 财务列为 NaN (那时市场确实不知道) — 正确行为。
  · _ann_date/_end_date 一并返回, 便于核查"这天用的是哪期财报、何时公告"。
"""
import os
import glob

import numpy as np
import pandas as pd

_ENGINE = 'fastparquet'
SRC_DEFAULT = 'tushare_cache/_partial/fundamentals'


def load_fundamentals_quarterly(src=SRC_DEFAULT):
    files = sorted(glob.glob(os.path.join(src, '*.parquet')))
    if not files:
        raise FileNotFoundError(f"no fundamentals parquet in {src} — 先跑 fetch_fundamentals.py")
    df = pd.concat((pd.read_parquet(f, engine=_ENGINE) for f in files), ignore_index=True)
    df['ann_date'] = pd.to_datetime(df['ann_date'].astype(str), format='%Y%m%d')
    df['end_date'] = pd.to_datetime(df['end_date'].astype(str), format='%Y%m%d')
    df['code'] = df['ts_code'].astype(str).str[:6]
    # 同 (code,end_date) 已在 fetch 层按最早 ann_date 去重; 这里再保险一次
    df = (df.sort_values(['code', 'ann_date', 'end_date'])
            .drop_duplicates(subset=['code', 'end_date'], keep='first'))
    return df


def pit_fundamentals(trade_dates, codes, src=SRC_DEFAULT):
    """As-of 对齐: 在 (trade_dates × codes) 网格上, 每格取 ann_date<=该交易日 的最近一期财务。

    trade_dates: DatetimeIndex (交易日); codes: list[str] (6位代码)。
    返回 index=(date, code) 的 panel。
    """
    fund = load_fundamentals_quarterly(src)
    fund = fund[fund['code'].isin(set(codes))]
    metric_cols = [c for c in fund.columns
                   if c not in ('ts_code', 'code', 'ann_date', 'end_date')]

    trade_dates = pd.DatetimeIndex(sorted(pd.DatetimeIndex(trade_dates).unique()))
    out_parts = []
    # 逐股 merge_asof: 把交易日"向后"匹配到最近一次已公告财报 (ann_date <= date)
    for code, g in fund.groupby('code'):
        g = g.sort_values('ann_date')
        left = pd.DataFrame({'date': trade_dates})
        merged = pd.merge_asof(left, g.rename(columns={'ann_date': '_ann_date'}),
                               left_on='date', right_on='_ann_date',
                               direction='backward')
        merged['code'] = code
        merged = merged.rename(columns={'end_date': '_end_date'})
        out_parts.append(merged[['date', 'code', '_ann_date', '_end_date'] + metric_cols])

    if not out_parts:
        return pd.DataFrame()
    panel = pd.concat(out_parts, ignore_index=True).set_index(['date', 'code']).sort_index()
    for c in metric_cols:
        panel[c] = panel[c].astype('float32')
    return panel
