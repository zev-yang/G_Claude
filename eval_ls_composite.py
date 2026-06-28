# -*- coding: utf-8 -*-
"""
eval_ls_composite.py — 路径甲: 多空 + beta中性 + 多因子聚合 的【最终判决】(预注册, 只跑一次).

不是再测单因子(那是多重检验跑步机), 是攻【病根】: beta(多空剥掉) + breadth(全截面双尾) + 对的尺子(IC)。
回答: 我们 vet 过的几个 faint 但正交的信号(value/quality/accruals/gm_mom), 聚合成多空 beta 中性组合后,
      到底有没有可部署的 alpha?  IR ≈ IC × √breadth —— 不需要单个强, 需要多个弱而正交 + breadth。

预注册 (看结果前定死):
  · 因子集(等权, 不调): value(5y估值分位) / quality(ROE) / accruals(ocf_to_profit) / gm_mom(毛利YoY去季节)
  · 构造: 全截面按 composite 排序, 多顶 Q / 空底 Q (dollar-neutral -> 剥 beta, breadth x2)
  · PASS = composite NW-IC t>=2 @126d  AND  {63,126,252} IC 同为正  AND  126d 前后两半同正  AND  多空年化价差>0
  · 同列单因子 IC 与 composite IC: 看聚合是否把信号抬上来。
  · 即便 PASS = 这是【modest 多因子 alpha】, 下一步用 长样本 + 行业/size 中性 做确认; 不是终点的"成了"。
"""
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from lurking_fundamentals import load_fundamentals_quarterly, pit_fundamentals
from lurking_quality_value import load_daily_basic_ts, valuation_score

HORIZONS = (63, 126, 252)
QTILE = 0.20            # 多顶/空底各 20%
T_MIN = 2.0
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
        if len(g) >= 30:
            out[d] = g['f'].corr(g['r'], method='spearman')
    return pd.Series(out).sort_index()


def _mom_pit(field, rebal_dates, lag, src=FUND_SRC):
    h = load_fundamentals_quarterly(src)
    h = h.dropna(subset=['ann_date', 'end_date']).copy()
    h['ann_date'] = pd.to_datetime(h['ann_date']); h['end_date'] = pd.to_datetime(h['end_date'])
    if 'code' not in h.columns:
        h['code'] = h['ts_code'].astype(str).str[:6]
    h = h.sort_values(['code', 'end_date'])
    h['mom'] = h.groupby('code')[field].diff(lag)
    h = h.dropna(subset=['mom']).sort_values('ann_date')
    out = []
    for d in pd.DatetimeIndex(rebal_dates):
        a = h[h['ann_date'] <= d].groupby('code').tail(1)
        if a.empty:
            continue
        out.append(pd.Series(a['mom'].to_numpy(), index=pd.MultiIndex.from_arrays(
            [np.repeat(d, len(a)), a['code'].to_numpy()], names=['date', 'code'])))
    return pd.concat(out) if out else pd.Series(dtype=float)


def _ls_spread(comp, fwd, dates, q=QTILE):
    """每调仓日: 顶 q 等权 - 底 q 等权 的前向价差 (dollar-neutral, 剥 beta)。"""
    df = pd.concat([comp.rename('c'), fwd.rename('r')], axis=1).dropna()
    out = {}
    for d in dates:
        try:
            g = df.xs(d, level='date')
        except KeyError:
            continue
        if len(g) < 50:
            continue
        hi = g.loc[g['c'] >= g['c'].quantile(1 - q), 'r'].mean()
        lo = g.loc[g['c'] <= g['c'].quantile(q), 'r'].mean()
        out[d] = hi - lo
    return pd.Series(out).sort_index()


def main():
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']
    dates = price.index.get_level_values('date').unique().sort_values()
    codes = price.index.get_level_values('code').unique().tolist()
    s = pd.Series(dates, index=dates)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()
    mid = rebal[len(rebal) // 2]
    wide = price.unstack('code')

    print("构建 4 个正交因子 (PIT) + 等权 composite ...")
    fund = pit_fundamentals(pd.DatetimeIndex(rebal), codes)
    F = pd.DataFrame(index=fund.index)
    F['value']    = _xs_rank(valuation_score(load_daily_basic_ts()).reindex(fund.index))
    F['quality']  = _xs_rank(fund['roe'])
    F['accruals'] = _xs_rank(fund['ocf_to_profit'])
    F['gm_mom']   = _xs_rank(_mom_pit('q_gsprofit_margin', rebal, lag=4)).reindex(fund.index)
    comp = F.mean(axis=1).where(F.count(axis=1) >= 3)          # 至少 3/4 个因子可得
    print("   因子非空率: " + (F.notna().mean() * 100).round(0).to_dict().__str__())
    print(f"   composite 可选: {int(comp.notna().sum())}\n")

    print(f"NW-IC (单因子 vs composite) + 多空年化价差; QTILE={QTILE:.0%}")
    hdr = f"{'横轴':>10}" + "".join(f"{h:>9}d" for h in HORIZONS)
    print(hdr); print('-' * len(hdr))
    # 单因子 + composite 的 IC-t (各 horizon)
    series = {**{c: F[c] for c in F.columns}, 'COMPOSITE': comp}
    ic_by = {}
    for name, fac in series.items():
        row = []
        for H in HORIZONS:
            fwd = (wide.shift(-H) / wide - 1.0).stack()
            ic = _ic_at(fac, fwd, rebal)
            t, _ = _nw_t(ic, max(1, round(H / 21)))
            row.append(t)
            if name == 'COMPOSITE':
                ic_by[H] = ic
        print(f"{name:>10}" + "".join(f"{t:>10.2f}" for t in row) + ("   <- IC_t_NW" if name == 'value' else ""))

    # composite 前后两半 + 多空价差
    print(f"\n{'(composite)':>12}{'IC前半':>9}{'IC后半':>9}{'LS年化':>10}")
    ls_ann = {}
    for H in HORIZONS:
        ic = ic_by[H]
        m1, m2 = ic[ic.index < mid].mean(), ic[ic.index >= mid].mean()
        fwd = (wide.shift(-H) / wide - 1.0).stack()
        ls = _ls_spread(comp, fwd, rebal)
        ann = float(np.nanmean(ls) * (252.0 / H))            # 年化(算术)多空价差
        ls_ann[H] = ann
        print(f"{H:>10}d{m1:>+9.3f}{m2:>+9.3f}{ann:>+10.1%}")

    # ── 预注册 PASS ──
    t126, _ = _nw_t(ic_by[126], 6)
    signs = []
    for H in HORIZONS:
        t, _ = _nw_t(ic_by[H], max(1, round(H / 21)))
        signs.append(t > 0)
    same_sign = all(signs)
    halves_pos = (ic_by[126][ic_by[126].index < mid].mean() > 0) and (ic_by[126][ic_by[126].index >= mid].mean() > 0)
    ls_pos = ls_ann[126] > 0
    verdict = (t126 >= T_MIN) and same_sign and halves_pos and ls_pos
    print(f"\n预注册判定: t126={t126:.2f}(>=2? {t126>=T_MIN}) | 三horizon同正={same_sign} | "
          f"126两半同正={halves_pos} | LS年化>0={ls_pos}  ->  "
          f"{'PASS: 存在可部署的多因子 beta中性 alpha (modest)' if verdict else '不过'}")
    print("  · PASS -> 下一步用 长样本(10-15y) + 行业/size 中性 做确认, 才谈部署。")
    print("  · 不过 -> 聚合也没把 faint 信号抬过线; 剩最后一根杠杆=拉长历史, 否则就是路径乙(拿市场收益)。")


if __name__ == '__main__':
    main()
