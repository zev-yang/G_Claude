# -*- coding: utf-8 -*-
"""
diag_data_source.py — 直接对比 stock_data_all(TDX+Tushare混源) vs _longhist(纯Tushare) 的 raw close, 验证污染。

V25 读 stock_data_all(混源), Value 读 _longhist(纯Tushare)。怀疑: 混源污染导致 V25 结果解释不通。
本脚本: 同(日期,股票)逐一比 raw close。对得上->无污染; 对不上(尤其早期TDX段系统性偏离)->污染坐实+定位接缝。
不重构 V25、不删库 —— 先用最小代价证伪/坐实假设。读法与 V25 完全一致(load_single_robust), 全程 fastparquet。
"""
import glob
import random
import re

import numpy as np
import pandas as pd

from data_loader import load_single_robust   # 与 V25 完全一致的 CSV 读法 (raw, 复权前)

LH_DAILY = './tushare_cache/_longhist/daily'
SDA_GLOB = './stock_data_all/*.csv'
N_SAMPLE = 150
MATCH_TOL = 0.005   # |比值-1|<0.5% 算匹配


def _code6(s):
    s = s.astype(str).str.extract(r'(\d{6})', expand=False)
    return s


def main():
    files = sorted(glob.glob(SDA_GLOB))
    if not files:
        raise SystemExit(f"无 {SDA_GLOB}")
    random.seed(42)
    sample = random.sample(files, min(N_SAMPLE, len(files)))
    print(f"抽样 {len(sample)}/{len(files)} 只股票对比 ...")

    parts = []
    for f in sample:
        try:
            d = load_single_robust(f)
            if d is not None and len(d):
                parts.append(d[['date', 'code', 'close']])
        except Exception as e:
            print(f"  [warn] {f}: {e!r}")
    sda = pd.concat(parts, ignore_index=True)
    sda['date'] = pd.to_datetime(sda['date'], errors='coerce')
    sda['code'] = _code6(sda['code'])
    sda = sda.dropna(subset=['date', 'code', 'close']).rename(columns={'close': 'sda'})
    codes = set(sda['code'].unique())

    lh_parts = []
    for f in sorted(glob.glob(f'{LH_DAILY}/*.parquet')):
        df = pd.read_parquet(f, engine='fastparquet')
        df['code'] = _code6(df['ts_code'])
        lh_parts.append(df[df['code'].isin(codes)][['trade_date', 'code', 'close']])
    lh = pd.concat(lh_parts, ignore_index=True)
    lh['date'] = pd.to_datetime(lh['trade_date'].astype(str), errors='coerce')
    lh = lh.dropna(subset=['date', 'code', 'close']).rename(columns={'close': 'lh'})

    m = sda.merge(lh[['date', 'code', 'lh']], on=['date', 'code'], how='inner')
    if m.empty:
        raise SystemExit("两份数据无重叠 (检查 code/date 格式)")
    m['ratio'] = m['sda'] / m['lh']
    m['match'] = (m['ratio'] - 1).abs() < MATCH_TOL
    m['year'] = m['date'].dt.year

    print(f"\n对齐 {len(m):,} 个 (日期×股) | 覆盖 {m['code'].nunique()} 股 | {m['date'].min().date()}~{m['date'].max().date()}")
    print(f"★ 总体匹配率 (|sda/lh - 1|<0.5%): {m['match'].mean():.1%}\n")

    print("逐年匹配率 + 中位比值 (看 TDX/Tushare 接缝在哪):")
    by = m.groupby('year').agg(对齐数=('match', 'size'), 匹配率=('match', 'mean'),
                                中位比值=('ratio', 'median'), 比值std=('ratio', 'std'))
    by['匹配率'] = (by['匹配率'] * 100).round(1).astype(str) + '%'
    by['中位比值'] = by['中位比值'].round(4)
    by['比值std'] = by['比值std'].round(4)
    print(by.to_string())

    bad = m[~m['match']]
    print(f"\n不匹配 {len(bad):,} 个 ({len(bad)/len(m):.1%})")
    if len(bad):
        print("最大偏离 5 例:")
        worst = bad.reindex((bad['ratio'] - 1).abs().sort_values(ascending=False).index).head(5)
        for _, r in worst.iterrows():
            print(f"  {r['code']} {r['date'].date()}: sda={r['sda']:.2f} vs lh={r['lh']:.2f} (比值 {r['ratio']:.3f})")

    print("\n判读:")
    print("  · 总体匹配率高(>98%)、逐年都接近100% -> 两份raw一致, 污染假设不成立, V25问题在别处。")
    print("  · 早期年份匹配率低/中位比值≠1、近年高 -> TDX那段raw与Tushare口径不同 = 污染坐实, 接缝即匹配率跳变处。")
    print("  · 若比值是个稳定常数(如恒2.0或0.5) -> 单位/小数口径差; 若杂乱 -> 复权/除权口径混乱。")


if __name__ == '__main__':
    main()
