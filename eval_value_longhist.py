# -*- coding: utf-8 -*-
"""
eval_value_longhist.py — 路径甲【最终判决】: value 因子在长样本上的 OOS 确认。

唯一假设 (预注册, 不在数据上调): value = 5年 PE/PB 历史分位(逐股自比, 越便宜分越高),
  经【行业 + size 中性化】(逐日对行业哑变量 + ln(circ_mv) 回归取残差) -> 剥掉行业/市值 tilt,
  再做【多空】(顶 20% - 底 20%, dollar-neutral -> 剥 beta)。问: 它的 IC 在长样本上稳不稳。

★ PASS (现在定死): 中性化后 value 的 NW-IC ——
    在 {63,126,252} full 全部 t>=2  AND  每个时间子期(均分3段) 126d IC 同为正  AND  多空年化价差>0。
  这是为抵消"在 4.3 年样本上发现 t=4.5"的选择性偏差而设的硬关; 全过 = value 是真因子(可部署起点);
  任一条破 = 那 t=4.5 是 regime 侥幸 -> 路径乙(拿市场收益)。
  quality(负)/accruals/gm_mom 一律搁置, 不在此调权重、不复活凑分。

数据: tushare_cache/_longhist (fetch_longhist) + _partial/industry/stock_industry.parquet。
"""
import glob
import os

import numpy as np
import pandas as pd

LH = './tushare_cache/_longhist'
IND_FILE = './tushare_cache/_partial/industry/stock_industry.parquet'
ENGINE = 'fastparquet'

HORIZONS = (63, 126, 252)
QTILE = 0.20
PCT_WIN = 1250          # 5年(~252*5)滚动分位
PCT_MINP = 252          # 至少1年才出分位
N_SUBPERIODS = 3
T_MIN = 2.0


def _read(sub, cols):
    fs = sorted(glob.glob(f'{LH}/{sub}/*.parquet'))
    if not fs:
        raise FileNotFoundError(f"无 {LH}/{sub} 分片, 先跑 fetch_longhist.py")
    df = pd.concat([pd.read_parquet(f, engine=ENGINE) for f in fs], ignore_index=True)
    df['date'] = pd.to_datetime(df['trade_date'])
    df['code'] = df['ts_code'].astype(str).str[:6]
    return df[['date', 'code'] + cols]


def _xs_rank(s):
    return s.groupby(level='date').rank(pct=True)


def _nw_t(ic, lag):
    x = ic.dropna().to_numpy(float); n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean(); var = (e @ e) / n
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(var / n)
    return x.mean() / se if se > 0 else np.nan


def _ic_at(fac, fwd, dates):
    df = pd.concat([fac.rename('f'), fwd.rename('r')], axis=1).dropna()
    out = {}
    for d in dates:
        try:
            g = df.xs(d, level='date')
        except KeyError:
            continue
        if len(g) >= 30:
            out[d] = g['f'].corr(g['r'], method='spearman')
    return pd.Series(out).sort_index()


def build_hfq(daily, adj):
    m = daily.merge(adj, on=['date', 'code'], how='inner')
    m['hfq'] = m['close'] * m['adj_factor']
    return m.set_index(['date', 'code'])['hfq'].sort_index()


def build_value(db):
    """5年 PE/PB 历史分位: value = 0.5*(1-pe分位) + 0.5*(1-pb分位), 越便宜越高。"""
    db = db.set_index(['date', 'code']).sort_index()
    parts = []
    for col in ('pe_ttm', 'pb'):
        w = db[col].where(db[col] > 0).unstack('code').sort_index()
        pct = w.rolling(PCT_WIN, min_periods=PCT_MINP).rank(pct=True)   # 当前值在自身5年内的分位
        parts.append((1.0 - pct).stack())
    return (0.5 * parts[0] + 0.5 * parts[1]).rename('value')


def neutralize(fac, ind_map, lncap, dates):
    """逐调仓日: fac 对 [行业哑变量 + ln(circ_mv)] OLS 取残差 -> 剥行业/size tilt。"""
    out = []
    for d in dates:
        try:
            y = fac.xs(d, level='date').dropna()
        except KeyError:
            continue
        if len(y) < 50:
            continue
        codes = y.index
        ind = ind_map.reindex(codes).fillna('UNK').astype('category')
        try:
            lc = lncap.xs(d, level='date').reindex(codes)
        except KeyError:
            continue
        D = pd.get_dummies(ind, drop_first=True).to_numpy(float)
        lc = lc.fillna(lc.median()).to_numpy(float).reshape(-1, 1)
        lc = (lc - np.nanmean(lc)) / (np.nanstd(lc) + 1e-9)
        X = np.column_stack([np.ones(len(y)), D, lc])
        yv = y.to_numpy(float)
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        resid = yv - X @ beta
        out.append(pd.Series(resid, index=pd.MultiIndex.from_arrays(
            [np.repeat(d, len(codes)), codes.to_numpy()], names=['date', 'code'])))
    return pd.concat(out).rename('value_neu') if out else pd.Series(dtype=float)


def _ls_spread(fac, fwd, dates, q=QTILE):
    df = pd.concat([fac.rename('c'), fwd.rename('r')], axis=1).dropna()
    out = {}
    for d in dates:
        try:
            g = df.xs(d, level='date')
        except KeyError:
            continue
        if len(g) < 50:
            continue
        out[d] = g.loc[g['c'] >= g['c'].quantile(1 - q), 'r'].mean() - \
                 g.loc[g['c'] <= g['c'].quantile(q), 'r'].mean()
    return pd.Series(out).sort_index()


def main():
    print("① 读长历史 + 构建 后复权收盘 / value / lncap ...")
    daily = _read('daily', ['close'])
    adj = _read('adj_factor', ['adj_factor'])
    db = _read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    price = build_hfq(daily, adj)
    value_raw = build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    ind = pd.read_parquet(IND_FILE, engine=ENGINE).drop_duplicates('code').set_index('code')['industry']

    dates = price.index.get_level_values('date').unique().sort_values()
    s = pd.Series(dates, index=dates)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()
    wide = price.unstack('code')
    print(f"   样本: {dates.min().date()} ~ {dates.max().date()} | 调仓月 {len(rebal)} | 股票 {price.index.get_level_values('code').nunique()}")

    print("② 行业 + size 中性化 value (逐调仓日 OLS 残差) ...")
    value_neu = neutralize(value_raw, ind, lncap, rebal)
    value_neu = _xs_rank(value_neu)                # 残差再转横截面 rank
    print(f"   中性化后可用: {int(value_neu.notna().sum())}\n")

    # 子期切分
    rb = [d for d in rebal if d in value_neu.index.get_level_values('date')]
    bounds = np.array_split(np.array(rb), N_SUBPERIODS)

    print(f"NW-IC (raw vs 中性化) + 多空年化; 子期 t (126d); QTILE={QTILE:.0%}")
    hdr = f"{'':>14}" + "".join(f"{h:>8}d" for h in HORIZONS)
    print(hdr); print('-' * len(hdr))
    ic_neu = {}
    for name, fac in (('value_raw', _xs_rank(value_raw)), ('value_中性', value_neu)):
        row = []
        for H in HORIZONS:
            fwd = (wide.shift(-H) / wide - 1.0).stack()
            ic = _ic_at(fac, fwd, rebal)
            t = _nw_t(ic, max(1, round(H / 21)))
            row.append(t)
            if name == 'value_中性':
                ic_neu[H] = ic
        print(f"{name:>14}" + "".join(f"{t:>9.2f}" for t in row))

    # 子期 (126d) + 多空
    fwd126 = (wide.shift(-126) / wide - 1.0).stack()
    print(f"\n中性化 value @126d 子期稳健 + 多空:")
    sub_pos = []
    for i, chunk in enumerate(bounds, 1):
        icc = ic_neu[126].loc[ic_neu[126].index.isin(chunk)]
        m = icc.mean()
        sub_pos.append(m > 0)
        print(f"   子期{i} ({pd.Timestamp(chunk[0]).date()}~{pd.Timestamp(chunk[-1]).date()}): mean-IC={m:+.3f}  ({'正' if m>0 else '负'})")
    ls = _ls_spread(value_neu, fwd126, rebal)
    ls_ann = float(np.nanmean(ls) * (252.0 / 126))
    print(f"   多空(顶20%-底20%)年化价差: {ls_ann:+.1%}")

    # ── 预注册 PASS ──
    t_all = {H: _nw_t(ic_neu[H], max(1, round(H / 21))) for H in HORIZONS}
    all_t = all(t_all[H] >= T_MIN for H in HORIZONS)
    all_sub_pos = all(sub_pos)
    ls_pos = ls_ann > 0
    verdict = all_t and all_sub_pos and ls_pos
    print(f"\n预注册判定: 全horizon t>=2={all_t} ({', '.join(f'{H}d:{t_all[H]:.1f}' for H in HORIZONS)}) | "
          f"子期全正={all_sub_pos} | 多空年化>0={ls_pos}")
    print(f"  ->  {'PASS: value 是真因子, 长样本 OOS 站住 -> 可作部署起点' if verdict else '不过: value 的 t=4.5 含 regime 成分, 回路径乙'}")


if __name__ == '__main__':
    main()
