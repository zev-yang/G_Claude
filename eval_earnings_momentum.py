# -*- coding: utf-8 -*-
"""
eval_earnings_momentum.py — 测【盈利边际改善】(调整二) 有没有 cross-sectional 信号.

★ 不做组合回测 (留出 2~3 窗口无法裁决, 必过拟合); 用高 power 的 NW-IC + 正交, 与 eval_cash_quality 同法。

预注册 (看结果前定死):
  · PRIMARY 因子 = 单季度 ROE 环比改善 (q_roe[t] - q_roe[t-1]); 方向: 改善越大越好。
  · 诊断 = 单季度毛利率环比 (q_gsprofit_margin 差分); 仅参照, 不替换 PRIMARY (best-of-N 纪律)。
  · 通过 = ICIR>=0.25 且 IC_t_NW>=2.0 且 maxabs_corr<0.60 (对现有 ROE-quality / value 正交)。
  · multiple-testing 诚实话: 这是本轮第 N 个被测因子, 单个 t≈2 在全局多重检验下仍弱; 通过≠可重仓。

前置: fetch_fundamentals.py 的 FIELDS 末尾加 'q_roe,q_gsprofit_margin', 清空重拉 _partial/fundamentals/。
"""
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from lurking_fundamentals import load_fundamentals_quarterly, pit_fundamentals
from lurking_quality_value import load_daily_basic_ts, valuation_score

MOM_FIELDS = [('q_roe', 'PRIMARY'), ('q_gsprofit_margin', '诊断')]
HORIZON_DAYS, NW_LAG = 126, 6
ICIR_MIN, T_MIN, CORR_MAX = 0.25, 2.0, 0.60
FUND_SRC = 'tushare_cache/_partial/fundamentals'


def _xs_rank(s):
    return s.groupby(level='date').rank(pct=True)


def _nw_t(ic, lag):
    x = ic.dropna().to_numpy(float); n = len(x)
    if n < 3:
        return np.nan, np.nan
    e = x - x.mean(); g0 = (e @ e) / n; var = g0
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(var / n)
    ac1 = (e[1:] @ e[:-1]) / (e @ e) if (e @ e) > 0 else np.nan
    return (x.mean() / se if se > 0 else np.nan), ac1


def _ic_at(factor, fwd, dates):
    df = pd.concat([factor.rename('f'), fwd.rename('r')], axis=1).dropna()
    out = {}
    for d in dates:
        try:
            g = df.xs(d, level='date')
        except KeyError:
            continue
        if len(g) >= 20:
            out[d] = g['f'].corr(g['r'], method='spearman')
    return pd.Series(out).sort_index()


def _xs_corr_at(a, b, dates):
    df = pd.concat([a.rename('a'), b.rename('b')], axis=1).dropna()
    cs = []
    for d in dates:
        try:
            g = df.xs(d, level='date')
        except KeyError:
            continue
        if len(g) >= 20:
            cs.append(g['a'].corr(g['b'], method='spearman'))
    return float(np.nanmean(cs)) if cs else np.nan


def mom_pit(field, rebal_dates, src=FUND_SRC):
    """PIT 环比改善: 逐 code 按 end_date 排序取 field 季度差(本季-上季), 自本季 ann_date 起可得;
    merge_asof 到各调仓日 (取每股最新可得的环比值)。"""
    h = load_fundamentals_quarterly(src)
    h = h.dropna(subset=['ann_date', 'end_date']).copy()
    h['ann_date'] = pd.to_datetime(h['ann_date'])
    h['end_date'] = pd.to_datetime(h['end_date'])
    if 'code' not in h.columns:
        h['code'] = h['ts_code'].astype(str).str[:6]
    h = h.sort_values(['code', 'end_date'])
    h['mom'] = h.groupby('code')[field].diff()           # 本季 - 上季
    h = h.dropna(subset=['mom']).sort_values('ann_date')
    out = []
    for d in pd.DatetimeIndex(rebal_dates):
        avail = h[h['ann_date'] <= d].groupby('code').tail(1)   # 每股最新可得环比
        if avail.empty:
            continue
        s = pd.Series(avail['mom'].to_numpy(),
                      index=pd.MultiIndex.from_arrays(
                          [np.repeat(d, len(avail)), avail['code'].to_numpy()],
                          names=['date', 'code']))
        out.append(s)
    return pd.concat(out).rename(field + '_mom') if out else pd.Series(dtype=float)


def main():
    print("① 价格 + 现有 ROE/value (正交目标) ...")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']
    dates = price.index.get_level_values('date').unique().sort_values()
    codes = price.index.get_level_values('code').unique().tolist()
    s = pd.Series(dates, index=dates)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()

    fund = pit_fundamentals(pd.DatetimeIndex(rebal), codes)
    roe_rank = _xs_rank(fund['roe'])
    value = valuation_score(load_daily_basic_ts())

    wide = price.unstack('code')
    fwd = (wide.shift(-HORIZON_DAYS) / wide - 1.0).stack()

    print(f"② gate + 正交 @ horizon={HORIZON_DAYS}d, 月度调仓, NW_lag={NW_LAG}\n")
    hdr = f"{'因子(环比)':>20}{'IC样本':>7}{'ICIR':>8}{'IC_ac1':>8}{'IC_t_NW':>9}{'corr_ROE':>10}{'corr_val':>10}{'maxabs':>8}{'判定':>7}"
    print(hdr); print('-' * len(hdr))
    for fld, tag in MOM_FIELDS:
        try:
            mom = mom_pit(fld, rebal)
        except Exception as e:
            print(f"{fld+'('+tag+')':>20}   读取失败(检查FIELDS是否含该字段+重拉): {e!r}")
            continue
        if mom.empty:
            print(f"{fld+'('+tag+')':>20}   无数据 (字段缺失? 检查FIELDS+重拉)")
            continue
        fac = _xs_rank(mom)
        ic = _ic_at(fac, fwd, rebal)
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        t_nw, ac1 = _nw_t(ic, NW_LAG)
        c_roe = _xs_corr_at(fac, roe_rank, rebal)
        c_val = _xs_corr_at(fac, value, rebal)
        mx = np.nanmax(np.abs([c_roe, c_val]))
        ok = (icir >= ICIR_MIN) and (t_nw >= T_MIN) and (mx < CORR_MAX) and (tag == 'PRIMARY')
        print(f"{fld+'('+tag+')':>20}{len(ic):>7}{icir:>8.3f}{ac1:>8.2f}{t_nw:>9.2f}"
              f"{c_roe:>10.2f}{c_val:>10.2f}{mx:>8.2f}{'PASS' if ok else '—':>7}")

    print(f"\n判读: PRIMARY(q_roe环比) 通过 = ICIR>={ICIR_MIN} 且 IC_t_NW>={T_MIN} 且 maxabs_corr<{CORR_MAX}。")
    print("  · 诊断列(毛利环比)仅参照, 即便分更高也【不替换】PRIMARY (best-of-N 纪律)。")
    print("  · 通过 -> 才有资格作为一个新成分进一步评估; 不过 -> 盈利动量在此 universe/horizon 也无信号。")
    print("  · 全局诚实话: 本轮已测多个因子, 单个 t≈2 仍弱; 通过≠可重仓, 且不改变'组合无可证伪超额'的结论。")


if __name__ == '__main__':
    main()
