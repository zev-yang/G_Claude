# -*- coding: utf-8 -*-
"""
eval_value_trend.py — 卫星因子增量测试①: 给已验证 value 加 120日 trend, 看能否缓解左侧价值陷阱。

纪律(预注册, 跑前写死):
  · 基线 = value-decile (季度, 已验证 IR 0.51)。
  · Step1 先单独验 trend 的中性化 NW-IC (A股动量常反转, 先过闸; ≈0/负则加它帮不了, 属预期)。
  · Step2 两种加法都测(预注册):
      filter   : 剔动量最弱后40%(留trend前60%), 再在剩下取value最便宜同样只数 (对应"避开左侧陷阱")。
      composite: 0.5*value_rank + 0.5*trend_rank 等权(不调权重), 取 top-decile。
  · 保留条件: vs value-alone, IR升 或 超额回撤变浅, 且 TE不显著升、换手成本没吃掉增量。否则砍。
  · 特别盯【超额回撤】: 痛点是买在还要跌的位置, trend有用应让回撤变浅。
trend = 120日收益, 行业+size 中性化后 xs-rank (与 value 同构)。复用 eval_value_longhist。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V

EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
TREND_LB = 120
TREND_KEEP = 0.60          # filter: 保留 trend 前 60%
COSTS = [0.001, 0.003, 0.005]
STEP = 3                   # 季度
HORIZONS = (63, 126, 252)


def _turnover(prev_w, new_w):
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())


def _metrics(port_r, bench_r, ppy):
    pr, br = np.asarray(port_r, float), np.asarray(bench_r, float)
    n = len(pr)
    p_ann = np.prod(1 + pr) ** (ppy / n) - 1
    b_ann = np.prod(1 + br) ** (ppy / n) - 1
    exc = pr - br
    te = exc.std() * np.sqrt(ppy)
    ir = (exc.mean() * ppy) / (te + 1e-12)
    rel = np.cumprod(1 + exc)
    dd = (rel / np.maximum.accumulate(rel) - 1).min()
    return p_ann, p_ann - b_ann, te, ir, dd


def run(value_neu, trend_neu, circ, wide, st, rebal, mode):
    gross, ewb, turns = [], [], []
    prev_w = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            v = value_neu.xs(d, level='date').dropna()
            tr = trend_neu.xs(d, level='date').dropna()
            cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = tradeable[~tradeable.isin(st)]
        vt = v.reindex(tradeable).dropna()
        if len(vt) < 50:
            continue
        N = max(1, int(len(vt) * TOP_PCT))
        if mode == 'value':
            picks = vt.nlargest(N).index
        elif mode == 'filter':
            tt = tr.reindex(vt.index).dropna()
            keep = tt[tt >= tt.quantile(1 - TREND_KEEP)].index      # 留动量前60%
            picks = vt.reindex(keep).dropna().nlargest(N).index
        else:  # composite
            common = vt.index.intersection(tr.index)
            vr = vt.reindex(common).rank(pct=True)
            trr = tr.reindex(common).rank(pct=True)
            picks = (0.5 * vr + 0.5 * trr).nlargest(N).index
        if len(picks) < 1:
            continue
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        turns.append(_turnover(prev_w, new_w)); prev_w = new_w
        gross.append(r); ewb.append(ew)
    return (np.array(x, float) for x in (gross, ewb, turns))


def main():
    print("① 构建 value(中性) + trend120(中性) ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind = ind_df['industry']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    wide = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code')

    dates = wide.index
    s = pd.Series(dates, index=dates)
    rebal_all = s.groupby([s.index.year, s.index.month]).first().tolist()
    rebal = rebal_all[::STEP]
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, rebal))
    trend_raw = (wide / wide.shift(TREND_LB) - 1).stack().rename('trend')
    trend_neu = V._xs_rank(V.neutralize(trend_raw, ind, lncap, rebal))
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')]
    print(f"   样本 {dates.min().date()}~{dates.max().date()} | 季度调仓 {len(rebal)}\n")

    # ── Step1: trend 单独 NW-IC ──
    print("Step1 — trend120(中性) 单独 NW-IC (A股动量是否信号):")
    print(f"{'':>10}" + "".join(f"{h:>8}d" for h in HORIZONS))
    row = []
    for H in HORIZONS:
        fwd = (wide.shift(-H) / wide - 1).stack()
        ic = V._ic_at(trend_neu, fwd, rebal)
        row.append(V._nw_t(ic, max(1, round(H / 21))))
    print(f"{'trend t':>10}" + "".join(f"{t:>9.2f}" for t in row))
    print(f"  ({'trend 像有正信号' if all(t > 2 for t in row) else 'trend 信号弱/不稳 — 加它大概率难抬IR(等权还稀释value)'})\n")

    # ── Step2: value-alone vs +trend ──
    print("Step2 — value-alone vs value+trend (季度, decile, vsEW, 扣成本):")
    hdr = f"{'方案':>12}{'组合年化@.3':>11}{'超额@.3':>9}{'TE':>7}{'IR@.1':>7}{'IR@.3':>7}{'IR@.5':>7}{'超额回撤@.3':>12}{'年换手':>8}"
    print(hdr); print('-' * len(hdr))
    res = {}
    for mode, label in (('value', 'value单独'), ('filter', '+trend过滤'), ('composite', '+trend复合')):
        g, e, t = run(value_neu, trend_neu, circ, wide, st, rebal, mode)
        if len(g) < 2:
            print(f"{label:>12}  样本不足"); continue
        ppy = 12.0 / STEP
        m = {c: _metrics(g - t * c, e, ppy) for c in COSTS}
        p3, exc3, te, ir3, dd3 = m[0.003]
        res[mode] = (ir3, dd3, te)
        print(f"{label:>12}{p3:>10.1%}{exc3:>+9.1%}{te:>7.1%}{m[0.001][3]:>7.2f}{ir3:>7.2f}{m[0.005][3]:>7.2f}{dd3:>11.1%}{np.mean(t)*ppy:>7.0%}")

    # ── 预注册判定 ──
    if 'value' in res:
        b_ir, b_dd, b_te = res['value']
        print(f"\n预注册判定 (基线 value: IR@.3={b_ir:.2f}, 回撤={b_dd:.1%}, TE={b_te:.1%}):")
        for mode, label in (('filter', '+trend过滤'), ('composite', '+trend复合')):
            if mode not in res:
                continue
            ir, dd, te = res[mode]
            ir_up = ir > b_ir + 0.02
            dd_better = dd > b_dd + 0.02      # 回撤变浅(dd是负数, 更大=更浅)
            te_ok = te <= b_te + 0.01
            keep = (ir_up or dd_better) and te_ok
            print(f"  {label}: IR {ir:.2f}({'升' if ir_up else '未升'}) | 回撤 {dd:.1%}({'变浅' if dd_better else '未变浅'}) | "
                  f"TE {te:.1%}({'可接受' if te_ok else '升太多'}) -> {'保留' if keep else '砍'}")
        print("  (保留=IR升或回撤变浅 且 TE不显著升; 否则 trend 没增量, 维持纯 value。)")


if __name__ == '__main__':
    main()
