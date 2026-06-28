# -*- coding: utf-8 -*-
"""
eval_cash_quality.py — 评估 fina_indicator 现金质量因子能否当 潜伏 hidden_alpha 的第5成分.

只复用 潜伏 现成模块 (lurking_fundamentals / lurking_quality_value) + V25 的 data_loader,
不依赖任何平行管线 (那些要退役)。回答两个问题, 都按 潜伏 调仓口径:
  1) gate: 这个现金质量因子在月度调仓上有没有【自相关调整后】的 IC? -> 读 IC_t_NW (非 ICIR)。
  2) 正交: 它是不是只是把 潜伏 已有的 ROE-quality / value 换个说法? -> maxabs_corr < 0.60 才是真增量。

通过判据 (写死, 不在回测上调): ICIR >= 0.25  AND  IC_t_NW >= 2.0  AND  maxabs_corr < 0.60。

前置: 先在 fetch_fundamentals.py 的 FIELDS 末尾加 'ocf_to_profit,ocf_to_or',
      清空并重拉 tushare_cache/_partial/fundamentals/ (增量 fetcher 会跳过旧分片, 不清就拿不到新列)。
"""
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from lurking_fundamentals import pit_fundamentals
from lurking_quality_value import load_daily_basic_ts, valuation_score

# ── pre-registration 旋钮 ──
CASH_FIELDS  = ['ocf_to_profit', 'ocf_to_or']   # 主选 ocf_to_profit(CFO/营业利润, 最贴近已验证 accruals); ocf_to_or(CFO/营收) 稳健备选
HORIZON_DAYS = 126      # 潜伏持有期(~6mo), 与回测口径一致 (3mo=63 / 12mo=252)
NW_LAG       = 6        # Newey-West 滞后(单位=调仓数) ≈ horizon 月数
WINSOR       = 0.01     # 比率类双侧 1% 截尾
ICIR_MIN, T_MIN, CORR_MAX = 0.25, 2.0, 0.60
# ──────────────────────────


def _xs_rank(s):
    return s.groupby(level='date').rank(pct=True)


def _nw_t(ic, lag):
    """mean(IC) 的 Newey-West(Bartlett) HAC t 值, 兼返 lag-1 自相关。"""
    x = ic.dropna().to_numpy(float)
    n = len(x)
    if n < 3:
        return np.nan, np.nan
    e = x - x.mean()
    g0 = (e @ e) / n
    var = g0
    for L in range(1, min(lag, n - 1) + 1):
        var += 2.0 * (1.0 - L / (lag + 1.0)) * (e[L:] @ e[:-L]) / n
    se = np.sqrt(var / n)
    ac1 = (e[1:] @ e[:-1]) / (e @ e) if (e @ e) > 0 else np.nan
    t = x.mean() / se if se > 0 else np.nan
    return t, ac1


def _ic_at(factor, fwd, dates):
    """在给定调仓日上逐日截面 Spearman IC -> Series(index=date)。"""
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
    """两因子在调仓日上的截面 Spearman 相关, 取均值 (正交性)。"""
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


def main():
    print("① 加载价格面板 + 潜伏现成 PIT 基本面 / 估值分 ...")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']
    dates = price.index.get_level_values('date').unique().sort_values()
    codes = price.index.get_level_values('code').unique().tolist()

    fund  = pit_fundamentals(dates, codes)                       # (date,code) -> fina_indicator 指标 (含新字段)
    db    = load_daily_basic_ts()
    value = valuation_score(db)                                  # 潜伏 value 成分
    roe_rank = _xs_rank(fund['roe'])                             # 潜伏 quality 软分 (ROE 横截面 rank)

    # 前向收益 + 月度调仓日 (与 lurking_backtest 一致)
    wide = price.unstack('code')
    fwd = (wide.shift(-HORIZON_DAYS) / wide - 1.0).stack()
    s = pd.Series(dates, index=dates)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()

    print(f"② gate + 正交 @ horizon={HORIZON_DAYS}d, 月度调仓, NW_lag={NW_LAG}\n")
    hdr = f"{'字段':>14}{'IC样本':>7}{'ICIR':>8}{'IC_ac1':>8}{'IC_t_NW':>9}{'corr_ROE':>10}{'corr_val':>10}{'maxabs':>8}{'判定':>7}"
    print(hdr); print('-' * len(hdr))
    for fld in CASH_FIELDS:
        if fld not in fund.columns:
            print(f"{fld:>14}   缺失—检查 FIELDS 并清空重拉 _partial/fundamentals/")
            continue
        lo, hi = fund[fld].quantile([WINSOR, 1 - WINSOR])
        fac = _xs_rank(fund[fld].clip(lo, hi))
        ic = _ic_at(fac, fwd, rebal)
        icir = ic.mean() / ic.std() if ic.std() > 0 else np.nan
        t_nw, ac1 = _nw_t(ic, NW_LAG)
        c_roe = _xs_corr_at(fac, roe_rank, rebal)
        c_val = _xs_corr_at(fac, value, rebal)
        mx = np.nanmax(np.abs([c_roe, c_val]))
        ok = (icir >= ICIR_MIN) and (t_nw >= T_MIN) and (mx < CORR_MAX)
        print(f"{fld:>14}{len(ic):>7}{icir:>8.3f}{ac1:>8.2f}{t_nw:>9.2f}"
              f"{c_roe:>10.2f}{c_val:>10.2f}{mx:>8.2f}{'PASS' if ok else '—':>7}")

    print(f"\n判读: 通过 = ICIR>={ICIR_MIN} 且 IC_t_NW>={T_MIN} 且 maxabs_corr<{CORR_MAX}。")
    print("  · IC_t_NW 是准 (季度因子日频 ICIR 被自相关吹高); maxabs_corr 看是否只是重述 ROE/value。")
    print("  · 通过的那个 -> 才作为第5成分加进 lurking_synthesis (一处权重改动); 都不过 -> 别加。")


if __name__ == '__main__':
    main()
