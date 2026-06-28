# -*- coding: utf-8 -*-
"""
eval_value_weight.py — V2.0①非等权: 已验证 decile value 组合, 等权 vs rank加权 vs score加权, 看能否多榨 Alpha。

不动持仓数、不加新因子 —— 唯一变量是【权重】。回答 Zev 的问题"为什么一定等权"。
内在张力: 加权=集中=降低有效breadth(报1/Σw²); 而breadth正是decile打赢30只的原因。净效果是经验问题。
   且 score加权把权重压向最便宜极端票(扎堆困境票) -> 可能反伤。让数据说。

预注册(三种, 写死, 不挑最好):
  · ew    : 1/N 等权 (基线, IR 0.51)。
  · rank  : 按 value 分排名线性, 最便宜权重最高 (w ∝ N+1-rank)。
  · score : w ∝ (value分 - decile临界分), 按信号强度给权 (更激进)。
判定: 加权方案要 ① IR 明显>等权, ② 过三子期, ③ TE/有效breadth 没崩。否则等权已是最优, 维持。
季度, vsEW, 扣0.3%成本。复用 eval_value_longhist。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V

EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
STEP = 3
COST = 0.003
N_SUB = 3


def _turnover(prev_w, new_w):
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())


def _weights(picks, vt, N, scheme):
    """picks: top-N value分(降序 Series)。返回权重 Series(sum=1)。"""
    if scheme == 'ew':
        return pd.Series(1.0 / len(picks), index=picks.index)
    if scheme == 'rank':
        r = picks.rank(ascending=False)              # 最高分=1
        w = (len(picks) + 1 - r)                      # 最便宜得 N, 边际得 1
        return w / w.sum()
    # score: 锚定 decile 临界分(第N+1只)
    s_cut = vt.iloc[N] if len(vt) > N else picks.min()
    w = (picks - s_cut).clip(lower=1e-6)
    return w / w.sum()


def backtest(value_neu, circ, wide, st, rebal, scheme):
    recs, turns, effN = [], [], []
    prev_w = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            v = value_neu.xs(d, level='date').dropna()
            cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = tradeable[~tradeable.isin(st)]
        vt = v.reindex(tradeable).dropna().sort_values(ascending=False)
        if len(vt) < 50:
            continue
        N = max(1, int(len(vt) * TOP_PCT))
        picks = vt.iloc[:N]
        w = _weights(picks, vt, N, scheme)
        try:
            ret_i = wide.loc[d1, picks.index] / wide.loc[d, picks.index] - 1
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        r = float((w * ret_i).sum())
        turns.append(_turnover(prev_w, w)); prev_w = w
        effN.append(1.0 / float((w ** 2).sum()))
        recs.append({'d1': d1, 'port': r, 'bench': ew})
    df = pd.DataFrame(recs).set_index('d1'); df['turn'] = turns
    return df, float(np.mean(effN))


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
    print("构建 value(中性) ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    wide = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code')
    s = pd.Series(wide.index, index=wide.index)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()[::STEP]
    value_neu = V._xs_rank(V.neutralize(value_raw, ind_df['industry'], lncap, rebal))
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')]
    ppy = 12.0 / STEP
    print(f"   样本 {wide.index.min().date()}~{wide.index.max().date()} | 季度 {len(rebal)}\n")

    schemes = [('ew', '等权(基线)'), ('rank', 'rank线性加权'), ('score', 'score比例加权')]
    bt = {m: backtest(value_neu, circ, wide, st, rebal, m) for m, _ in schemes}

    print("全样本 (decile, 季度, vsEW, @0.3%成本):")
    hdr = f"{'权重方案':>14}{'有效持仓':>9}{'超额':>9}{'IR':>7}{'TE':>7}{'超额回撤':>10}"
    print(hdr); print('-' * len(hdr))
    for m, label in schemes:
        df, en = bt[m]
        exc, te, ir, dd = _metrics(df, COST, ppy)
        print(f"{label:>14}{en:>9.0f}{exc:>+9.1%}{ir:>7.2f}{te:>7.1%}{dd:>10.1%}")

    print(f"\n三子期 IR (季度, @0.3%): 加权方案要全段稳, 不能偏近期")
    idx0 = bt['ew'][0].index
    chunks = np.array_split(np.array(idx0), N_SUB)
    h = f"{'权重方案':>14}" + "".join(f"{'子期'+str(i):>10}" for i in range(1, N_SUB + 1))
    print(h); print('-' * len(h))
    sub_ok = {}
    for m, label in schemes:
        df = bt[m][0]; row = []; irs = []
        for ch in chunks:
            sub = df.loc[df.index.isin(ch)]
            _, _, ir, _ = _metrics(sub, COST, ppy)
            row.append(f"{ir:>10.2f}"); irs.append(ir)
        sub_ok[m] = all(x > 0 for x in irs)
        print(f"{label:>14}" + "".join(row))

    # 判定
    exc_e, te_e, ir_e, dd_e = _metrics(bt['ew'][0], COST, ppy)
    print(f"\n=== 判定 (非等权能否干净抬 IR) ===")
    for m, label in (('rank', 'rank线性'), ('score', 'score比例')):
        df, en = bt[m]
        exc, te, ir, dd = _metrics(df, COST, ppy)
        ir_up = ir > ir_e + 0.03
        cross = sub_ok[m]
        te_ok = te <= te_e + 0.01
        keep = ir_up and cross and te_ok
        print(f"  {label}: IR {ir:.2f} vs 等权 {ir_e:.2f}({'升' if ir_up else '未升'}) | 三子期全正 {'✅' if cross else '❌'} | "
              f"TE {te:.1%}({'稳' if te_ok else '升'}) | 有效持仓 {en:.0f} -> {'✅ 采用' if keep else '❌ 不采用'}")
    print("  (采用=IR明显升 且 三子期全正 且 TE没崩; 否则等权已最优, 维持。集中虽用更多信号, 但降breadth, 常打平。)")


if __name__ == '__main__':
    main()
