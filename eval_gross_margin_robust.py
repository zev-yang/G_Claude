# -*- coding: utf-8 -*-
"""
eval_gross_margin_robust.py — gross-margin momentum 的【更硬】确认测试 (re-pre-registered).

它原是【诊断】里冒出来的(选择性偏差) + 用的是 QoQ(单季毛利率有季节性), 所以这里过一道硬关:
  · 构造对比: QoQ(本季-上季, 季节污染) vs YoY去季节(本季-去年同季 = diff 4 季度)。
  · horizon {63,126,252} × 时间 {full/前半/后半}; NW 滞后随 horizon 缩放(≈H/21)。

★ PRE-REGISTERED PASS (只看 YoY 去季节, 现在定死):
    NW-t>=2 在 {63,126,252} full 全部成立  AND  126d 前后两半 mean-IC 同为正  AND  maxabs_corr<0.60。
  QoQ 列仅诊断季节性 (QoQ 强而 YoY 垮 = 那 t=3 大半是日历效应)。
  即便全过 -> 小卫星(accruals 待遇), 非核心; 组合无独立超额结论不变。

前置: FIELDS 已含 q_gsprofit_margin (上一步已加+重拉)。
"""
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from lurking_fundamentals import load_fundamentals_quarterly, pit_fundamentals
from lurking_quality_value import load_daily_basic_ts, valuation_score

FIELD = 'q_gsprofit_margin'
HORIZONS = (63, 126, 252)
CORR_MAX, T_MIN = 0.60, 2.0
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


def mom_pit(field, rebal_dates, lag, src=FUND_SRC):
    """PIT 动量: 逐 code 按 end_date 排序取 field 的 lag 阶差 (lag=1 QoQ / lag=4 YoY去季节),
    自该季 ann_date 起可得; 取每股最新可得值对齐到调仓日。"""
    h = load_fundamentals_quarterly(src)
    h = h.dropna(subset=['ann_date', 'end_date']).copy()
    h['ann_date'] = pd.to_datetime(h['ann_date'])
    h['end_date'] = pd.to_datetime(h['end_date'])
    if 'code' not in h.columns:
        h['code'] = h['ts_code'].astype(str).str[:6]
    h = h.sort_values(['code', 'end_date'])
    h['mom'] = h.groupby('code')[field].diff(lag)
    h = h.dropna(subset=['mom']).sort_values('ann_date')
    out = []
    for d in pd.DatetimeIndex(rebal_dates):
        avail = h[h['ann_date'] <= d].groupby('code').tail(1)
        if avail.empty:
            continue
        out.append(pd.Series(avail['mom'].to_numpy(),
                   index=pd.MultiIndex.from_arrays(
                       [np.repeat(d, len(avail)), avail['code'].to_numpy()], names=['date', 'code'])))
    return pd.concat(out).rename('mom') if out else pd.Series(dtype=float)


def main():
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']
    dates = price.index.get_level_values('date').unique().sort_values()
    codes = price.index.get_level_values('code').unique().tolist()
    s = pd.Series(dates, index=dates)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()
    mid = rebal[len(rebal) // 2]
    wide = price.unstack('code')
    roe_rank = _xs_rank(pit_fundamentals(pd.DatetimeIndex(rebal), codes)['roe'])
    value = valuation_score(load_daily_basic_ts())

    print(f"gross-margin momentum 确认测试 (NW-t; 前/后半为 mean-IC 符号)\n")
    hdr = (f"{'构造':>12}{'horizon':>8}{'NWlag':>6}{'ICIR':>7}{'IC_ac1':>7}"
           f"{'t_full':>8}{'IC前半':>8}{'IC后半':>8}{'maxabs':>8}")
    print(hdr); print('-' * len(hdr))
    yoy_cells = {}
    for cons, lag in (('QoQ(季节)', 1), ('YoY(去季节)', 4)):
        mom = mom_pit(FIELD, rebal, lag)
        if mom.empty:
            print(f"{cons:>12}   无数据 (检查 FIELDS 含 q_gsprofit_margin + 重拉)"); continue
        fac = _xs_rank(mom)
        mx = np.nanmax(np.abs([_xs_corr_at(fac, roe_rank, rebal), _xs_corr_at(fac, value, rebal)]))
        for H in HORIZONS:
            nwlag = max(1, round(H / 21))
            fwd = (wide.shift(-H) / wide - 1.0).stack()
            ic = _ic_at(fac, fwd, rebal)
            icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
            t_full, ac1 = _nw_t(ic, nwlag)
            m1, m2 = ic[ic.index < mid].mean(), ic[ic.index >= mid].mean()
            print(f"{cons:>12}{H:>8}{nwlag:>6}{icir:>7.2f}{ac1:>7.2f}{t_full:>8.2f}"
                  f"{m1:>+8.3f}{m2:>+8.3f}{mx:>8.2f}")
            if cons.startswith('YoY'):
                yoy_cells[H] = dict(t=t_full, m1=m1, m2=m2, mx=mx)

    # ── 预注册 PASS 判定 (只看 YoY) ──
    if yoy_cells:
        all_t = all(yoy_cells[H]['t'] >= T_MIN for H in HORIZONS)
        halves_pos = (yoy_cells[126]['m1'] > 0) and (yoy_cells[126]['m2'] > 0)
        orth = yoy_cells[126]['mx'] < CORR_MAX
        verdict = all_t and halves_pos and orth
        print(f"\n预注册判定 (YoY去季节): "
              f"全horizon t>=2={all_t} | 126d两半同正={halves_pos} | 正交={orth}  ->  "
              f"{'PASS (小卫星, 非核心)' if verdict else '不过 -> 动量线收掉'}")
    print("  · 无论过否, 组合层面'无可证伪独立超额'结论不变; 这只是诚实地把动量线测到底。")


if __name__ == '__main__':
    main()
