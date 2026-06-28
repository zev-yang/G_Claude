# -*- coding: utf-8 -*-
"""
eval_value_sector.py — 行业分层 value: 每行业取最便宜K只, 对比 naive top-N, 找小资金可管理(~30只)且不自欺的版本。

动机: breadth 表显示 naive top-30 超额从 decile +2.1% 掉到 +0.6% (不只 breadth, 更因"最便宜30只"
      扎堆在最惨行业+困境票, 高相关+偏价值陷阱)。行业分层(每行业取最便宜K只)打散相关、避扎堆,
      且顺着已验证的【行业中性】信号 —— 不是另起炉灶。

预注册(写死, 一次跑完照单全收):
  · 域: 全市场剔ST剔微盘; value = 行业+size 中性化(同已验证)。
  · 分层: 每行业取 value 最便宜 K 只 (K=1/2), 等权。
  · 对比: naive top-30 / 分层(K=1,~30) / 分层(K=2,~60) / decile(全量基线)。
  · 判定: 分层版要 ① IR 明显 > naive 同只数, 且 ② 过三子期关(不能偏近期, trend 就栽这)。
  · 季度, vsEW, 扣0.3%成本。
复用 eval_value_longhist。
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


def _tradeable(circ, st, d):
    cm = circ.xs(d)
    t = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    return t[~t.isin(st)]


def select(value_neu, circ, st, ind, d, mode):
    """mode: 'naive30' / 'sectorK1' / 'sectorK2' / 'decile'"""
    v = value_neu.xs(d, level='date').dropna()
    try:
        tradeable = _tradeable(circ, st, d)
    except KeyError:
        return None, None
    vt = v.reindex(tradeable).dropna()
    if len(vt) < 50:
        return None, None
    if mode == 'naive30':
        picks = vt.nlargest(30).index
    elif mode == 'decile':
        picks = vt.nlargest(max(1, int(len(vt) * TOP_PCT))).index
    elif mode == 'capped30':                       # top-30 by value, 每行业最多2只 (恰30只+分散, 不受行业数影响)
        sec = ind.reindex(vt.index).fillna('UNK')
        cnt = {}; chosen = []
        for code in vt.sort_values(ascending=False).index:
            sname = sec.get(code, 'UNK')
            if cnt.get(sname, 0) < 2:
                chosen.append(code); cnt[sname] = cnt.get(sname, 0) + 1
            if len(chosen) >= 30:
                break
        picks = pd.Index(chosen)
    else:  # 行业分层: 每行业取最便宜 K 只
        K = 1 if mode == 'sectorK1' else 2
        sec = ind.reindex(vt.index).fillna('UNK')
        df = pd.DataFrame({'v': vt, 'sec': sec})
        picks = df.sort_values('v', ascending=False).groupby('sec').head(K).index
    return picks, tradeable


def backtest(value_neu, circ, wide, st, ind, rebal, mode):
    recs, turns = [], []
    prev = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        picks, tradeable = select(value_neu, circ, st, ind, d, mode)
        if picks is None or len(picks) < 1:
            continue
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        turns.append(_turnover(prev, new_w)); prev = new_w
        recs.append({'d1': d1, 'port': r, 'bench': ew, 'n': len(picks)})
    df = pd.DataFrame(recs).set_index('d1'); df['turn'] = turns
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
    return p_ann - b_ann, te, ir, dd, df['n'].mean()


def main():
    print("构建 value(中性) ...")
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
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')]
    ppy = 12.0 / STEP
    print(f"   样本 {wide.index.min().date()}~{wide.index.max().date()} | 季度 {len(rebal)} | 行业数 {ind.nunique()}\n")

    modes = [('naive30', 'naive最便宜30'), ('capped30', '分散30(每业≤2)'),
             ('sectorK1', '行业分层K1'), ('sectorK2', '行业分层K2'),
             ('decile', 'decile全量(基线)')]
    bt = {m: backtest(value_neu, circ, wide, st, ind, rebal, m) for m, _ in modes}

    print("全样本 (季度, vsEW, @0.3%成本):")
    hdr = f"{'方案':>16}{'持仓数':>7}{'超额':>9}{'IR':>7}{'TE':>7}{'超额回撤':>10}"
    print(hdr); print('-' * len(hdr))
    for m, label in modes:
        exc, te, ir, dd, n = _metrics(bt[m], COST, ppy)
        print(f"{label:>16}{n:>7.0f}{exc:>+9.1%}{ir:>7.2f}{te:>7.1%}{dd:>10.1%}")

    # ── 三子期 (分层 vs naive30, 看分层是否普适且不偏近期) ──
    print(f"\n三子期 IR (季度, @0.3%): 分层版要全段稳, 不能像 trend 那样偏近期")
    idx_common = bt['naive30'].index
    chunks = np.array_split(np.array(idx_common), N_SUB)
    h = f"{'方案':>16}" + "".join(f"{'子期'+str(i):>10}" for i in range(1, N_SUB + 1))
    print(h); print('-' * len(h))
    sub_ok = {}
    for m, label in modes:
        df = bt[m]; row = []; irs = []
        for ch in chunks:
            sub = df.loc[df.index.isin(ch)]
            if len(sub) < 2:
                row.append('  n/a'); irs.append(np.nan); continue
            _, _, ir, _, _ = _metrics(sub, COST, ppy)
            row.append(f"{ir:>10.2f}"); irs.append(ir)
        sub_ok[m] = all((not np.isnan(x)) and x > 0 for x in irs)
        print(f"{label:>16}" + "".join(row))

    # ── 判定 ──
    exc_n, _, ir_n, dd_n, _ = _metrics(bt['naive30'], COST, ppy)
    print(f"\n=== 判定 (小资金~30只是否有可练手的版本) ===")
    for m, label in (('capped30', '分散30(每业≤2)'), ('sectorK1', '行业分层K1'), ('sectorK2', '行业分层K2(~60)')):
        exc, te, ir, dd, n = _metrics(bt[m], COST, ppy)
        better = ir > ir_n + 0.05
        cross = sub_ok[m]
        ok = better and cross
        print(f"  {label}: IR {ir:.2f} vs naive30 {ir_n:.2f} ({'更优' if better else '没更优'}) | "
              f"三子期全正 {'✅' if cross else '❌'} -> {'✅ 可作小资金练手版' if ok else '❌ 不达标'}")
    print("  (达标=IR明显>naive同档 且 三子期全正; 否则小资金跑不了这个因子, 老实模拟盘或换主动选股。)")


if __name__ == '__main__':
    main()
