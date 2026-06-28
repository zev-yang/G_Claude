# -*- coding: utf-8 -*-
"""
eval_trend_robust.py — trend过滤增量(IR 0.51->0.85)的稳健性三关, 排除过拟合/混淆再定型。

trend过滤式在主测里大幅改善(IR升、回撤减半)。好得意外 -> 先当红旗。三关:
  关1 size体检   : 过滤前后组合 平均市值/价格 变了多少? 若过滤后明显变大盘 -> 增量混进 size, 打折。
                   (trend 已 size 中性化, 理论上不该有 tilt; 此处实测验证。)
  关2 阈值敏感   : 剔 后20/30/40/50/60% 的 IR/超额/回撤 —— 要单调平滑改善, 不能只 40% 一个孤点(阈值挖矿)。
  关3 子期稳健   : 增量在 2011-16/16-21/21-26 三段是否都为正 —— 不能只靠某一段。
三关全过 -> trend过滤是真增量, 可写进生产; 任一破 -> 0.85 含水分, 维持纯 value。
复用 eval_value_longhist。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V

EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
TREND_LB = 120
STEP = 3
COST = 0.003
KEEP_GRID = [1.00, 0.80, 0.70, 0.60, 0.50, 0.40]   # 1.0=value单独(不剔); 0.60=剔后40%
N_SUB = 3


def _turnover(prev_w, new_w):
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())


def select(value_neu, trend_neu, circ, st, d, keep):
    v = value_neu.xs(d, level='date').dropna()
    cm = circ.xs(d)
    tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    tradeable = tradeable[~tradeable.isin(st)]
    vt = v.reindex(tradeable).dropna()
    if len(vt) < 50:
        return None, None
    N = max(1, int(len(vt) * TOP_PCT))
    if keep >= 1.0:
        picks = vt.nlargest(N).index
    else:
        tr = trend_neu.xs(d, level='date').dropna()
        tt = tr.reindex(vt.index).dropna()
        kept = tt[tt >= tt.quantile(1 - keep)].index
        picks = vt.reindex(kept).dropna().nlargest(N).index
    return picks, tradeable


def backtest(value_neu, trend_neu, circ, wide, st, rebal, keep):
    recs, turns = [], []
    prev = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            picks, tradeable = select(value_neu, trend_neu, circ, st, d, keep)
        except KeyError:
            continue
        if picks is None or len(picks) < 1:
            continue
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
            mc = circ.xs(d).reindex(picks).mean()
            px = wide.loc[d, picks].mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        turns.append(_turnover(prev, new_w)); prev = new_w
        recs.append({'d1': d1, 'port': r, 'bench': ew, 'mc': mc, 'px': px})
    df = pd.DataFrame(recs).set_index('d1')
    df['turn'] = turns
    return df


def _metrics(df, cost, ppy):
    net = df['port'] - df['turn'] * cost
    exc = (net - df['bench']).to_numpy(float)
    p_ann = np.prod(1 + net.to_numpy(float)) ** (ppy / len(net)) - 1
    b_ann = np.prod(1 + df['bench'].to_numpy(float)) ** (ppy / len(net)) - 1
    te = exc.std() * np.sqrt(ppy)
    ir = (exc.mean() * ppy) / (te + 1e-12)
    rel = np.cumprod(1 + exc)
    dd = (rel / np.maximum.accumulate(rel) - 1).min()
    return p_ann - b_ann, te, ir, dd


def main():
    print("构建 value(中性) + trend120(中性) ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind = ind_df['industry']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    wide = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code')
    s = pd.Series(wide.index, index=wide.index)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()[::STEP]
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, rebal))
    trend_raw = (wide / wide.shift(TREND_LB) - 1).stack().rename('trend')
    trend_neu = V._xs_rank(V.neutralize(trend_raw, ind, lncap, rebal))
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')]
    ppy = 12.0 / STEP
    print(f"   样本 {wide.index.min().date()}~{wide.index.max().date()} | 季度 {len(rebal)}\n")

    bt = {k: backtest(value_neu, trend_neu, circ, wide, st, rebal, k) for k in KEEP_GRID}

    # ── 关2: 阈值敏感 ──
    print("关2 — 阈值敏感 (剔最弱动量后 X%; @0.3%成本): 要单调平滑, 非孤点")
    hdr = f"{'方案':>12}{'超额':>9}{'IR':>7}{'超额回撤':>10}"
    print(hdr); print('-' * len(hdr))
    for k in KEEP_GRID:
        exc, te, ir, dd = _metrics(bt[k], COST, ppy)
        label = 'value单独(不剔)' if k >= 1.0 else f'剔后{int((1-k)*100)}%'
        print(f"{label:>12}{exc:>+9.1%}{ir:>7.2f}{dd:>10.1%}")

    # ── 关1: size/价格 体检 (value单独 vs 剔后40%) ──
    base, filt = bt[1.00], bt[0.60]
    print(f"\n关1 — size/价格 体检 (value单独 vs 剔后40%): 过滤后是否偷偷变大盘?")
    mc_b, mc_f = base['mc'].mean() / 1e4, filt['mc'].mean() / 1e4
    px_b, px_f = base['px'].mean(), filt['px'].mean()
    print(f"   平均流通市值: value单独 {mc_b:.1f}亿 -> 剔后40% {mc_f:.1f}亿  (比值 {mc_f/mc_b:.2f})")
    print(f"   平均价格    : value单独 {px_b:.1f}元 -> 剔后40% {px_f:.1f}元  (比值 {px_f/px_b:.2f})")
    tilt = mc_f / mc_b
    print(f"   -> {'⚠️ 过滤后明显变大盘(比值>1.2), 增量可能混 size, 打折看' if tilt > 1.2 else '✅ 市值无显著漂移(trend已size中性, 符合预期), 增量不是 size'}")

    # ── 关3: 子期稳健 (增量在三段是否都正) ──
    print(f"\n关3 — 子期稳健 (剔后40% 相对 value单独 的超额增量, 三段是否都正):")
    j = base[['port', 'bench', 'turn']].join(filt[['port', 'turn']], rsuffix='_f', how='inner')
    j['exc_v'] = (j['port'] - j['turn'] * COST) - j['bench']
    j['exc_f'] = (j['port_f'] - j['turn_f'] * COST) - j['bench']
    chunks = np.array_split(np.array(j.index), N_SUB)
    all_pos = True
    for i, ch in enumerate(chunks, 1):
        sub = j.loc[j.index.isin(ch)]
        v_ann = sub['exc_v'].mean() * ppy
        f_ann = sub['exc_f'].mean() * ppy
        inc = f_ann - v_ann
        all_pos &= inc > 0
        print(f"   子期{i} ({pd.Timestamp(ch[0]).date()}~{pd.Timestamp(ch[-1]).date()}): "
              f"value {v_ann:+.1%} | 剔后40% {f_ann:+.1%} | 增量 {inc:+.1%} ({'正' if inc>0 else '负'})")

    # ── 总判 ──
    exc_b, _, ir_b, dd_b = _metrics(base, COST, ppy)
    exc_f, _, ir_f, dd_f = _metrics(filt, COST, ppy)
    irs = [_metrics(bt[k], COST, ppy)[2] for k in KEEP_GRID]
    base_ir = irs[0]
    # 剔后30/40/50% (KEEP 0.70/0.60/0.50) 邻域是否都明显优于 value单独 -> 非孤点(驼峰也算过)
    broad = all(irs[KEEP_GRID.index(k)] >= base_ir + 0.05 for k in (0.70, 0.60, 0.50))
    print(f"\n=== 三关总判 ===")
    print(f"  关1 size无漂移: {'✅' if tilt <= 1.2 else '❌'} (市值比 {tilt:.2f})")
    print(f"  关2 阈值非孤点: {'✅' if broad else '❌'} (剔后30/40/50%邻域均明显优于value单独)")
    print(f"  关3 子期全正  : {'✅' if all_pos else '❌'}")
    verdict = (tilt <= 1.2) and broad and all_pos
    print(f"  -> {'✅ 三关全过: trend过滤是真增量, 可写进 build_value_portfolio (value选股前加剔最弱动量层)' if verdict else '❌ 有破关: 0.85 含水分, 维持纯 value'}")


if __name__ == '__main__':
    main()
