# -*- coding: utf-8 -*-
"""
check_holdings_return.py — 算【当期 value 持仓】从调仓日买入持有到最新的收益 (用你本地数据, 准)。

读 value_portfolio_current.csv (build_value_portfolio 导出的当期持仓) + _longhist 后复权价格,
算: 每只 调仓日->最新 收益, 等权(按weight)组合收益, 同期沪深300, 超额, 胜率, 最好/最差几只。

BUY_DATE 默认取持仓文件里的调仓日(没有则 2026-04-01); 卖出取数据最新日。
跑: python check_holdings_return.py
"""
import os
import glob

import numpy as np
import pandas as pd

LH = './tushare_cache/_longhist'
HOLD_FILES = ['./value_portfolio_current.csv', './value_portfolio_holdings.csv']
BUY_DATE_FALLBACK = '2026-04-01'
ENGINE = 'fastparquet'


def _read_lh(sub, cols):
    fs = sorted(glob.glob(f'{LH}/{sub}/*.parquet'))
    if not fs:
        raise FileNotFoundError(f"无 {LH}/{sub}, 先更新长历史")
    df = pd.concat([pd.read_parquet(f, engine=ENGINE) for f in fs], ignore_index=True)[cols]
    df['date'] = pd.to_datetime(df['trade_date']); df['code'] = df['ts_code'].astype(str).str[:6]
    return df


def main():
    hf = next((f for f in HOLD_FILES if os.path.exists(f)), None)
    if hf is None:
        print("找不到持仓文件, 先跑 build_value_portfolio.py"); return
    h = pd.read_csv(hf, dtype={'code': str})
    h['code'] = h['code'].str.zfill(6)
    buy_date = pd.Timestamp(h['rebal_date'].iloc[0]) if 'rebal_date' in h.columns else pd.Timestamp(BUY_DATE_FALLBACK)
    w = h.set_index('code')['weight'] if 'weight' in h.columns else pd.Series(1.0 / len(h), index=h['code'])

    # 后复权价格面板
    d = _read_lh('daily', ['ts_code', 'trade_date', 'close'])
    a = _read_lh('adj_factor', ['ts_code', 'trade_date', 'adj_factor'])
    m = d.merge(a[['date', 'code', 'adj_factor']], on=['date', 'code'])
    m['hfq'] = m['close'] * m['adj_factor']
    px = m.set_index(['date', 'code'])['hfq'].unstack('code').sort_index()

    d0 = px.index[px.index >= buy_date][0]                 # 实际买入交易日
    d1 = px.index[-1]                                       # 最新交易日
    codes = [c for c in h['code'] if c in px.columns]
    p0, p1 = px.loc[d0, codes], px.loc[d1, codes]
    rets = (p1 / p0 - 1.0).dropna()
    wv = w.reindex(rets.index).fillna(0)
    port = float((rets * wv).sum() / wv.sum())             # 按权重(等权)组合收益

    # 沪深300 同期
    bench = np.nan
    ipath = f'{LH}/index/000300.parquet'
    if os.path.exists(ipath):
        idx = pd.read_parquet(ipath, engine=ENGINE); idx['date'] = pd.to_datetime(idx['trade_date'])
        ic = idx.set_index('date')['close'].sort_index()
        bench = float(ic.asof(d1) / ic.asof(d0) - 1)

    n_miss = len(h) - len(rets)
    print(f"=== 当期 value 持仓 收益复盘 ===")
    print(f"买入 {d0.date()} -> 最新 {d1.date()} ({(d1 - d0).days} 天) | 持仓 {len(h)} 只 (有价 {len(rets)}{f', 缺价 {n_miss}' if n_miss else ''})")
    print(f"\n组合收益(按权重)      : {port:+.2%}")
    if not np.isnan(bench):
        print(f"同期沪深300          : {bench:+.2%}")
        print(f"超额                 : {port - bench:+.2%}")
    print(f"\n个股: 胜率 {(rets > 0).mean():.0%} | 中位 {rets.median():+.1%} | 均值 {rets.mean():+.1%} | 最好 {rets.max():+.1%} / 最差 {rets.min():+.1%}")
    top = rets.sort_values(ascending=False)
    nm = h.set_index('code')['name'] if 'name' in h.columns else pd.Series('', index=h['code'])
    def fmt(s): return '  '.join(f"{c}({nm.get(c,'')}) {v:+.0%}" for c, v in s.items())
    print(f"  最好5: {fmt(top.head(5))}")
    print(f"  最差5: {fmt(top.tail(5))}")
    print("\n注: 后复权收益(含分红除权调整); 缺价多为停牌/退市, 已按有价部分等权。这是一次性持有的实际表现, 非策略年化。")


if __name__ == '__main__':
    main()
