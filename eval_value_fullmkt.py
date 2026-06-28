# -*- coding: utf-8 -*-
"""
eval_value_fullmkt.py — 路径1: value 放回主场(全市场含中小盘, long-only)测净落地超额。

大盘 tilt 已证伪(vsEW 0成本即负)。value 的钱在全市场两端+中小盘。这里测唯一还没测的形态:
  全市场 long-only(剔ST+剔微盘) 选最便宜 decile, 对照可交易域等权, 扣【中小盘高成本】。

★ 预注册(定死, 一次跑完照单全收):
  · 可交易域 = 非ST 且 circ_mv >= 当期 30 分位 (剔微盘: 否则'最便宜'=壳/退市垃圾, 假收益且不可交易)。
  · 组合 = 域内最便宜 10%(top decile) 等权。
  · 选股信号两个都报: 中性化 value(纯) + raw value(value+size)。
  · 基准 = 可交易域等权 (size-clean -> 纯 value 归因)。
  · 成本 0.1/0.2/0.3/0.5%/边 (高端=中小盘真实冲击); 月度 vs 季度。
  · 成功 = ~0.3%/边下 vsEW 净超额>0 且 IR>~0.5 -> 可落地; 否则 value 主场也扛不住成本 -> 路径B。

复用 eval_value_longhist 的数据构建块。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V

EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
COSTS = [0.001, 0.002, 0.003, 0.005]
FREQS = {'月度': 1, '季度': 3}


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


def main():
    print("① 读长历史 + 构建 价格 / value(raw & 中性) / circ_mv / ST集 ...")
    daily = V._read('daily', ['close']); adj = V._read('adj_factor', ['adj_factor'])
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    price = V.build_hfq(daily, adj)
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind = ind_df['industry']
    st_codes = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])

    dates = price.index.get_level_values('date').unique().sort_values()
    s = pd.Series(dates, index=dates)
    rebal_all = s.groupby([s.index.year, s.index.month]).first().tolist()
    wide = price.unstack('code')
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, rebal_all))
    value_raw_r = V._xs_rank(value_raw)
    print(f"   样本 {dates.min().date()}~{dates.max().date()} | 剔ST {len(st_codes)} 只 + 剔微盘后 long-only top{TOP_PCT:.0%}\n")

    print("净超额 vs 可交易域等权 (size-clean); 中小盘高成本扫描")
    hdr = f"{'频率':>5}{'信号':>9}{'成本/边':>8}{'组合年化':>9}{'超额vsEW':>10}{'跟踪误差':>9}{'IR':>6}{'超额回撤':>9}{'年换手':>8}"
    print(hdr); print('-' * len(hdr))

    for fname, step in FREQS.items():
        rebal = [d for d in rebal_all[::step] if d in value_neu.index.get_level_values('date')]
        for sig_name, fac in (('中性value', value_neu), ('raw value', value_raw_r)):
            gross, ewb, turns = [], [], []
            prev_w = pd.Series(dtype=float)
            for i in range(len(rebal) - 1):
                d, d1 = rebal[i], rebal[i + 1]
                try:
                    v = fac.xs(d, level='date').dropna()
                    cm = circ.xs(d)
                except KeyError:
                    continue
                tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
                tradeable = tradeable[~tradeable.isin(st_codes)]
                vin = v.reindex(tradeable).dropna()
                if len(vin) < 50:
                    continue
                picks = vin.nlargest(max(1, int(len(vin) * TOP_PCT))).index
                try:
                    r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
                    ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
                except KeyError:
                    continue
                new_w = pd.Series(1.0 / len(picks), index=picks)
                turns.append(_turnover(prev_w, new_w)); prev_w = new_w
                gross.append(r); ewb.append(ew)
            gross, ewb, turns = map(lambda x: np.array(x, float), (gross, ewb, turns))
            if len(gross) < 2:
                print(f"{fname:>5}{sig_name:>9}   样本不足 (域太小/数据不足)"); continue
            ppy = 12.0 / step
            avg_turn = np.nanmean(turns) * ppy
            for c in COSTS:
                net = gross - turns * c
                p_ann, exc, te, ir, dd = _metrics(net, ewb, ppy)
                print(f"{fname:>5}{sig_name:>9}{c*100:>7.1f}%{p_ann:>8.1%}{exc:>+10.1%}{te:>8.1%}{ir:>6.2f}{dd:>8.1%}{avg_turn:>7.0%}")
    print("\n判读:")
    print("  · 看现实成本(~0.3%/边)那几行的 超额vsEW + IR。>0 且 IR>0.5 -> value 主场可落地。")
    print("  · 对比'中性value' vs 'raw value': 若只有 raw 为正 -> 可落地的钱靠小盘 tilt(承担小盘风险), 非纯 value。")
    print("  · vsEW 在现实成本下仍 ~0/负 -> value 在主场也扛不住中小盘成本 -> 学术真实但你拿不到, 转路径B。")


if __name__ == '__main__':
    main()
