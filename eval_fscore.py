# -*- coding: utf-8 -*-
"""
eval_fscore.py — V2.0: Piotroski F-score(7项财务健康) 作 value 的【避陷阱过滤】, 用基本面数据给集中价值选股避雷。

F-score 是为"在便宜股里筛掉财务垮掉的、留下健康的"而设计 -> 直击集中持有最怕的价值陷阱。
维度不同于已试过的: trend(价格,破关) / asset_growth(单一财务,无效); F-score 是9项综合健康度(此处7项)。

7项 (你 fundamentals 字段可算; 缺 current_ratio/股本 两项):
  盈利: ROA>0 | 经营现金流>0(ocfps) | ROA同比改善 | 现金流>净利(ocf_to_profit>1, 盈利质量)
  杠杆: 负债率下降(debt_to_assets YoY)
  效率: 毛利率改善(grossprofit_margin YoY) | 资产周转改善(assets_turn YoY)
F-score = 7项之和(0-7), 越高越健康。YoY 用同股 end_date 排序后 shift(4)(同季比, 避季节性)。

预注册(写死): 中性化 -> 正交性 + 单独NW-IC + 三子期; 两腿(复合/过滤)。
  判定: F-score 作过滤让 value 的 IR升或回撤变浅, 且三子期稳、TE不崩 -> 采用。否则砍。
PIT: F-score 按 ann_date as-of 每调仓日。复用 eval_value_longhist。
"""
import glob

import numpy as np
import pandas as pd

import eval_value_longhist as V

FUND_DIR = './tushare_cache/_partial/fundamentals'
EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
FSCORE_KEEP = 0.60          # 过滤: 保留 F-score 前60%(剔财务最差40%)
STEP = 3
COST = 0.003
HORIZONS = (63, 126, 252)
N_SUB = 3
T_MIN = 2.0


def _turnover(prev_w, new_w):
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())


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


def _subperiod_ir(df, ppy):
    chunks = np.array_split(np.array(df.index), N_SUB)
    return [(_metrics(df.loc[df.index.isin(ch)], COST, ppy)[2] if df.index.isin(ch).sum() >= 2 else np.nan) for ch in chunks]


def load_fscore(rebal):
    fs = sorted(glob.glob(f'{FUND_DIR}/*.parquet'))
    if not fs:
        raise SystemExit(f"无 {FUND_DIR}")
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, engine='fastparquet'))
        except Exception as e:
            print(f"  [warn] {f}: {e!r}")
    df = pd.concat(parts, ignore_index=True)
    need = ['roa', 'ocfps', 'ocf_to_profit', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"fundamentals 缺字段 {miss} — 需加入 fetch_fundamentals FIELDS 重拉。现有: {list(df.columns)}")
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
    df['end'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
    for c in need:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ann', 'end', 'roa']).sort_values(['code', 'end'])

    g = df.groupby('code')
    F1 = (df['roa'] > 0)
    F2 = (df['ocfps'] > 0)
    F3 = ((df['roa'] - g['roa'].shift(4)) > 0)
    F4 = (df['ocf_to_profit'] > 1)
    F5 = ((df['debt_to_assets'] - g['debt_to_assets'].shift(4)) < 0)
    F8 = ((df['grossprofit_margin'] - g['grossprofit_margin'].shift(4)) > 0)
    F9 = ((df['assets_turn'] - g['assets_turn'].shift(4)) > 0)
    df['fscore'] = (F1.astype(int) + F2.astype(int) + F3.astype(int) + F4.astype(int)
                    + F5.astype(int) + F8.astype(int) + F9.astype(int))

    df = df.sort_values('ann')
    panels = []
    for d in rebal:
        sub = df[df['ann'] <= d]
        if sub.empty:
            continue
        latest = sub.groupby('code')['fscore'].last()
        panels.append(pd.DataFrame({'date': d, 'code': latest.index, 'fscore': latest.values}))
    fsc = pd.concat(panels).set_index(['date', 'code'])['fscore']
    print(f"  [fscore] 构建完成 | F-score 分布: " +
          ", ".join(f"{k}:{int(v)}" for k, v in df['fscore'].value_counts().sort_index().items()))
    return fsc


def backtest(score, circ, wide, st, rebal, mode, value_neu=None, fscore_neu=None):
    recs, turns = [], []
    prev = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = tradeable[~tradeable.isin(st)]
        try:
            if mode == 'filter':
                v = value_neu.xs(d, level='date').dropna()
                fsc = fscore_neu.xs(d, level='date').dropna()
                ft = fsc.reindex(tradeable).dropna()
                keep = ft[ft >= ft.quantile(1 - FSCORE_KEEP)].index   # 留财务前60%
                vin = v.reindex(keep).dropna()
            else:
                sc = score.xs(d, level='date').dropna()
                vin = sc.reindex(tradeable).dropna()
        except KeyError:
            continue
        if len(vin) < 50 if mode != 'filter' else len(vin) < 10:
            continue
        N = max(1, int((len(tradeable) if mode == 'filter' else len(vin)) * TOP_PCT))
        picks = vin.nlargest(min(N, len(vin))).index
        if len(picks) < 1:
            continue
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        turns.append(_turnover(prev, new_w)); prev = new_w
        recs.append({'d1': d1, 'port': r, 'bench': ew})
    df = pd.DataFrame(recs).set_index('d1'); df['turn'] = turns
    return df


def main():
    print("① 构建 value(中性) + F-score(中性) ...")
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

    fsc = load_fscore(rebal)
    fscore_neu = V._xs_rank(V.neutralize(fsc.rename('fs'), ind, lncap, rebal))
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date') and d in fscore_neu.index.get_level_values('date')]
    ppy = 12.0 / STEP
    print(f"   样本 {wide.index.min().date()}~{wide.index.max().date()} | 季度 {len(rebal)}\n")

    # 正交性
    j = pd.concat([value_neu.rename('v'), fscore_neu.rename('f')], axis=1).dropna()
    corr = j.groupby(level='date').apply(lambda g: g['v'].corr(g['f'], method='spearman')).mean()
    print(f"② 正交性: F-score 与 value 截面相关 = {corr:+.2f} "
          f"({'✅ 够正交' if abs(corr) < 0.3 else '⚠️ 偏相关(健康股常更贵, 正常)'})\n")

    # 单独 IC + 子期
    print("③ F-score 单独 NW-IC (本身是不是收益信号; 注: 即便弱, 作'去陷阱过滤'仍可能有用):")
    print(f"{'':>10}" + "".join(f"{h:>8}d" for h in HORIZONS))
    ts = []
    for H in HORIZONS:
        ic = V._ic_at(fscore_neu, (wide.shift(-H) / wide - 1).stack(), rebal)
        ts.append(V._nw_t(ic, max(1, round(H / 21))))
    print(f"{'fscore t':>10}" + "".join(f"{t:>9.2f}" for t in ts))
    ic126 = V._ic_at(fscore_neu, (wide.shift(-126) / wide - 1).stack(), rebal)
    chunks = np.array_split(np.array([d for d in rebal if d in ic126.index]), N_SUB)
    print(f"   子期 mean-IC(126d): " + " | ".join(f"{ic126.loc[ic126.index.isin(ch)].mean():+.3f}" for ch in chunks) + "\n")

    # 两腿: value vs +Fscore复合 vs +Fscore过滤
    print("④ value vs +F-score (decile, 季度, vsEW, @0.3%): 重点看过滤式能否抬IR/降回撤且三子期稳")
    bt_v = backtest(value_neu, circ, wide, st, rebal, 'value')
    bt_c = backtest(0.5 * value_neu + 0.5 * fscore_neu, circ, wide, st, rebal, 'composite')
    bt_f = backtest(None, circ, wide, st, rebal, 'filter', value_neu=value_neu, fscore_neu=fscore_neu)
    hdr = f"{'方案':>16}{'超额':>9}{'IR':>7}{'TE':>7}{'超额回撤':>10}{'子期1':>8}{'子期2':>8}{'子期3':>8}"
    print(hdr); print('-' * len(hdr))
    rows = {}
    for df, label, key in ((bt_v, 'value单独', 'v'), (bt_c, '+Fscore复合', 'c'), (bt_f, '+Fscore过滤', 'f')):
        exc, te, ir, dd = _metrics(df, COST, ppy)
        sir = _subperiod_ir(df, ppy)
        rows[key] = (ir, dd, te, sir)
        print(f"{label:>16}{exc:>+9.1%}{ir:>7.2f}{te:>7.1%}{dd:>10.1%}" + "".join(f"{x:>8.2f}" for x in sir))

    # 判定
    ir_v, dd_v, te_v, sir_v = rows['v']
    print(f"\n=== 判定 (F-score 作避陷阱过滤是否有用) ===")
    for key, label in (('f', '+Fscore过滤'), ('c', '+Fscore复合')):
        ir, dd, te, sir = rows[key]
        better = (ir > ir_v + 0.03) or (dd > dd_v + 0.02)
        cross = all((not np.isnan(x)) and x > 0 for x in sir)
        te_ok = te <= te_v + 0.01
        keep = better and cross and te_ok
        print(f"  {label}: IR {ir:.2f} vs {ir_v:.2f} | 回撤 {dd:.1%} vs {dd_v:.1%} | 三子期全正 {'✅' if cross else '❌'} | "
              f"TE {'稳' if te_ok else '升'} -> {'✅ 采用' if keep else '❌ 不采用'}")
    print("  (采用=IR升或回撤变浅 且 三子期全正 且 TE没崩; 否则 F-score 没增量, 维持纯 value。)")
    print("  注: 即便整体不采用, 若过滤式三子期稳且回撤变浅, F-score 仍可作【集中选股的避雷短名单】用途。")


if __name__ == '__main__':
    main()
