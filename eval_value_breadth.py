# -*- coding: utf-8 -*-
"""
eval_value_breadth.py — breadth 敏感: 已验证 value 组合在不同持仓数下的净 IR (为 10万账户定 N)。

策略不变(已验证): 季度 / 中性化 value / 全市场剔ST剔微盘 / vsEW。只变【持仓数】:
  20 / 30 / 50 / 100 / decile(~266)。看 IR≈IC×√breadth 的代价: 只数↓ -> IR↓、TE↑、回撤↑。
这是【纯确认】, 不是新策略、不调参; 用来按账户大小挑 N, 并量化"降级"降多少。

判读: 现实成本(~0.3%/边)下, 看各 N 的 IR/超额/回撤。10万现实只能 20-30 只 ->
      看它相对 decile 丢了多少 IR, 心里有数这是降级版。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V

EXCL_SMALL_PCT = 0.30
N_LIST = [20, 30, 50, 100, 'decile']
TOP_PCT = 0.10
COSTS = [0.001, 0.003, 0.005]
FREQS = {'季度': 3, '月度': 1}


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


def run(fac, circ, wide, st, rebal, spec):
    gross, ewb, turns = [], [], []
    prev_w = pd.Series(dtype=float)
    cnts = []
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            v = fac.xs(d, level='date').dropna(); cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = tradeable[~tradeable.isin(st)]
        vin = v.reindex(tradeable).dropna()
        if len(vin) < 50:
            continue
        n = int(len(vin) * TOP_PCT) if spec == 'decile' else min(spec, len(vin))
        picks = vin.nlargest(max(1, n)).index
        cnts.append(len(picks))
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        turns.append(_turnover(prev_w, new_w)); prev_w = new_w
        gross.append(r); ewb.append(ew)
    g, e, t = (np.array(x, float) for x in (gross, ewb, turns))
    return g, e, t, (int(np.mean(cnts)) if cnts else 0)


def main():
    print("① 读长历史 + 构建 中性化 value / circ_mv / ST集 ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    dates = value_raw.index.get_level_values('date').unique().sort_values()
    s = pd.Series(dates, index=dates)
    rebal_all = s.groupby([s.index.year, s.index.month]).first().tolist()
    wide = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code')
    value_neu = V._xs_rank(V.neutralize(value_raw, ind_df['industry'], lncap, rebal_all))
    print(f"   样本 {dates.min().date()}~{dates.max().date()}\n")

    for fname, step in FREQS.items():
        rebal = [d for d in rebal_all[::step] if d in value_neu.index.get_level_values('date')]
        print(f"【{fname}调仓】净超额 vs 可交易域等权; 成本 0.1/0.3/0.5%/边")
        hdr = (f"{'持仓数':>10}{'组合年化@.3':>11}{'超额@.3':>9}{'TE':>7}"
               f"{'IR@.1':>7}{'IR@.3':>7}{'IR@.5':>7}{'超额回撤@.3':>12}{'年换手':>8}")
        print(hdr); print('-' * len(hdr))
        for spec in N_LIST:
            g, e, t, avgn = run(value_neu, circ, wide, st, rebal, spec)
            if len(g) < 2:
                continue
            ppy = 12.0 / step
            m = {c: _metrics(g - t * c, e, ppy) for c in COSTS}
            p3, exc3, te, ir3, dd3 = m[0.003]
            label = f"decile(~{avgn})" if spec == 'decile' else f"{spec}"
            print(f"{label:>10}{p3:>10.1%}{exc3:>+9.1%}{te:>7.1%}"
                  f"{m[0.001][3]:>7.2f}{ir3:>7.2f}{m[0.005][3]:>7.2f}{dd3:>11.1%}{np.mean(t)*ppy:>7.0%}")
        print()
    print("判读:")
    print("  · IR≈IC×√breadth: 持仓↓ -> IR 系统性↓、TE↑、回撤↑。看你能接受的 IR 下限对应多少只。")
    print("  · 10万现实 20-30 只: 看它 vs decile 的 IR 差 = 小账户的'降级代价'。")
    print("  · 选定 N 后我把 build_value_portfolio 的持仓数改成它; 注意小 N 单票仓位大、真实冲击成本高于这里的flat假设。")


if __name__ == '__main__':
    main()
