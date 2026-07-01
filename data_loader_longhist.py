# -*- coding: utf-8 -*-
"""
data_loader_longhist.py — V25 的 16年干净数据加载器: 读 _longhist 完整OHLCV + 16年adj_factor, 输出与 load_universe_audit 同格式的 panel。

为什么: V25 原读 stock_data_all(仅~4年, 且 load_single_robust 的 max_history_days=1000 截断到~4年),
        审计窗口 auto_start=now-730 又只~2年OOS -> park 判决数据不足。本加载器喂满16年, 公平重判。
口径对齐(与 load_single_robust 一致): vol(手)->volume; amount(千元)->×1000=元; close_raw 由 attach_adjustment 生成。
复权: 用 _longhist/adj_factor (16年, 非 _partial 的~4年), 全程后复权 hfq。全程 fastparquet 引擎。
用法: run.py 把 load_universe_audit(...) 换成 load_universe_longhist()。
"""
import glob

import numpy as np
import pandas as pd

from data_loader import attach_adjustment   # 复用 V25 同一个后复权逻辑

LH_DAILY = './tushare_cache/_longhist/daily'
LH_ADJ = './tushare_cache/_longhist/adj_factor/*.parquet'   # 16年复权因子 (关键: 非 _partial 的~4年)
IND_FILE = './tushare_cache/_partial/industry/stock_industry.parquet'


def load_universe_longhist(max_history_days=None):
    files = sorted(glob.glob(f'{LH_DAILY}/*.parquet'))
    if not files:
        raise ValueError(f"无 {LH_DAILY} — 先补拉完整OHLCV (fetch_longhist)")
    print(f"\n📋 [1] Loading _longhist OHLCV ({len(files)} 天分片) ...")
    df = pd.concat([pd.read_parquet(f, engine='fastparquet') for f in files], ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['date'] = pd.to_datetime(df['trade_date'].astype(str), errors='coerce')

    # 口径对齐 load_single_robust: vol(手)->volume; amount(千元)->元
    df = df.rename(columns={'vol': 'volume'})
    if 'amount' in df.columns:
        df['amount'] = (pd.to_numeric(df['amount'], errors='coerce') * 1000.0)   # 千元 -> 元
    else:
        df['amount'] = df['close'] * df['volume'] * 100.0
    for c in ['open', 'high', 'low', 'close', 'volume', 'amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')

    # name (V25 panel 需要; _longhist 无 -> 用行业表名, 缺则用代码)
    try:
        nm = pd.read_parquet(IND_FILE, engine='fastparquet').drop_duplicates('code').set_index('code')['name']
        df['name'] = df['code'].map(nm)
    except Exception:
        df['name'] = np.nan
    df['name'] = df['name'].fillna(df['code']).astype(str)

    df = df.dropna(subset=['date', 'code', 'close'])
    df = df[['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount', 'name']]
    panel = (df.drop_duplicates(subset=['date', 'code'], keep='last')
               .set_index(['date', 'code']).sort_index())

    # 可选: 限制每股最近 N 天 (默认 None = 全16年; 传数值则与 max_history_days 同义截断)
    if max_history_days is not None:
        panel = (panel.groupby(level='code', group_keys=False)
                      .apply(lambda g: g.iloc[-int(max_history_days):]))

    print(f"  > Loaded: {len(panel):,} rows | {panel.index.get_level_values('date').min().date()}"
          f"~{panel.index.get_level_values('date').max().date()} | {panel.index.get_level_values('code').nunique()} 股")
    panel = attach_adjustment(panel, adj_glob=LH_ADJ)   # 16年复权因子
    return panel


if __name__ == '__main__':
    p = load_universe_longhist()
    print("\n列:", list(p.columns))
    print("close_raw 在?", 'close_raw' in p.columns)
    print(p.head(3).to_string())
