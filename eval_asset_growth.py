# -*- coding: utf-8 -*-
"""
eval_asset_growth.py — V2.0②: asset growth(资产增长)因子, 找 value 的正交互补腿(靠分散扛逆风, 非择时)。

low asset growth anomaly: 低资产增长 -> outperform (海外稳, 独立于估值/盈利)。
预注册(写死):
  · asset growth = 总资产同比增长率(fina_indicator.assets_yoy), 低增长->高分; 行业+size 中性化。
  · 先验正交性: 与 value 的截面相关够低才有分散价值。
  · 单独 NW-IC + 三子期: 本身是不是信号、跨不跨regime (同 value 硬关)。
  · 过关才叠加: value+ag 两腿等权, 重点看【子期3(value逆风段)】组合是否被扛平滑。
PIT: assets_yoy 按 ann_date as-of 每个调仓日。复用 eval_value_longhist。
"""
import glob

import numpy as np
import pandas as pd

import eval_value_longhist as V

FUND_DIR = './tushare_cache/_partial/fundamentals'
EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
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


def load_asset_growth(rebal):
    fs = sorted(glob.glob(f'{FUND_DIR}/*.parquet'))
    if not fs:
        raise SystemExit(f"无 {FUND_DIR} — 先更新 fundamentals (fetch_fundamentals)")
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['ann_date'])
    if 'assets_yoy' in df.columns:
        df['ag'] = pd.to_numeric(df['assets_yoy'], errors='coerce')
        src = 'assets_yoy(总资产同比增长率)'
    elif 'total_assets' in df.columns:
        df['end'] = pd.to_datetime(df['end_date'])
        df = df.sort_values(['code', 'end'])
        df['ag'] = df.groupby('code')['total_assets'].transform(lambda x: x / x.shift(4) - 1) * 100
        src = 'total_assets 算 YoY'
    else:
        raise SystemExit("fundamentals 无 assets_yoy/total_assets —— 在 fetch_fundamentals FIELDS 加 'assets_yoy' 重拉 (或我改读 balancesheet)")
    df = df.dropna(subset=['ann', 'ag']).sort_values('ann')
    panels = []
    for d in rebal:
        sub = df[df['ann'] <= d]
        if sub.empty:
            continue
        latest = sub.groupby('code')['ag'].last()
        panels.append(pd.DataFrame({'date': d, 'code': latest.index, 'ag': latest.values}))
    ag = pd.concat(panels).set_index(['date', 'code'])['ag']
    return ag, src


def backtest(score, circ, wide, st, rebal):
    recs, turns = [], []
    prev = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            sc = score.xs(d, level='date').dropna(); cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = tradeable[~tradeable.isin(st)]
        sct = sc.reindex(tradeable).dropna()
        if len(sct) < 50:
            continue
        N = max(1, int(len(sct) * TOP_PCT))
        picks = sct.nlargest(N).index
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


def _subperiod_ir(df, ppy):
    chunks = np.array_split(np.array(df.index), N_SUB)
    out = []
    for ch in chunks:
        sub = df.loc[df.index.isin(ch)]
        out.append(_metrics(sub, COST, ppy)[2] if len(sub) >= 2 else np.nan)
    return out


def main():
    print("① 构建 value(中性) + asset_growth(中性) ...")
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

    ag, src = load_asset_growth(rebal)
    ag_neu = V._xs_rank(V.neutralize((-ag).rename('agsig'), ind, lncap, rebal))   # 低增长->高分
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date') and d in ag_neu.index.get_level_values('date')]
    ppy = 12.0 / STEP
    print(f"   源: {src} | 样本 {wide.index.min().date()}~{wide.index.max().date()} | 季度 {len(rebal)}\n")

    # ── 先验正交性 ──
    j = pd.concat([value_neu.rename('v'), ag_neu.rename('a')], axis=1).dropna()
    corr = j.groupby(level='date').apply(lambda g: g['v'].corr(g['a'], method='spearman')).mean()
    print(f"② 正交性: asset_growth 与 value 截面相关 = {corr:+.2f} "
          f"({'✅ 够正交, 有分散价值' if abs(corr) < 0.3 else '⚠️ 偏相关, 分散价值有限'})\n")

    # ── 单独 NW-IC + 三子期 ──
    print("③ asset_growth 单独 NW-IC (本身是不是信号):")
    print(f"{'':>10}" + "".join(f"{h:>8}d" for h in HORIZONS))
    ts = []
    for H in HORIZONS:
        fwd = (wide.shift(-H) / wide - 1).stack()
        ic = V._ic_at(ag_neu, fwd, rebal)
        ts.append(V._nw_t(ic, max(1, round(H / 21))))
    print(f"{'ag t':>10}" + "".join(f"{t:>9.2f}" for t in ts))
    ic126 = V._ic_at(ag_neu, (wide.shift(-126) / wide - 1).stack(), rebal)
    chunks = np.array_split(np.array([d for d in rebal if d in ic126.index]), N_SUB)
    sub_ic = [ic126.loc[ic126.index.isin(ch)].mean() for ch in chunks]
    print(f"   子期 mean-IC(126d): " + " | ".join(f"{m:+.3f}" for m in sub_ic))
    standalone_ok = all(t >= T_MIN for t in ts) and all(m > 0 for m in sub_ic)
    print(f"   -> 单独{'通过(全horizon t>=2 且 三子期正)' if standalone_ok else '不达硬关'}\n")

    # ── 两腿叠加: value vs value+ag ──
    print("④ 两腿叠加 (decile, 季度, vsEW, @0.3%): 重点看子期3(value逆风)是否被扛平滑")
    bt_v = backtest(value_neu, circ, wide, st, rebal)
    bt_v2 = backtest(0.5 * value_neu + 0.5 * ag_neu, circ, wide, st, rebal)
    hdr = f"{'方案':>16}{'超额':>9}{'IR':>7}{'TE':>7}{'超额回撤':>10}{'子期1':>8}{'子期2':>8}{'子期3':>8}"
    print(hdr); print('-' * len(hdr))
    for df, label in ((bt_v, 'value单独'), (bt_v2, 'value+ag两腿')):
        exc, te, ir, dd = _metrics(df, COST, ppy)
        sir = _subperiod_ir(df, ppy)
        print(f"{label:>16}{exc:>+9.1%}{ir:>7.2f}{te:>7.1%}{dd:>10.1%}" + "".join(f"{x:>8.2f}" for x in sir))

    # ── 判定 ──
    exc_v, te_v, ir_v, dd_v = _metrics(bt_v, COST, ppy)
    exc_2, te_2, ir_2, dd_2 = _metrics(bt_v2, COST, ppy)
    s3_v, s3_2 = _subperiod_ir(bt_v, ppy)[2], _subperiod_ir(bt_v2, ppy)[2]
    print(f"\n=== 判定 ===")
    print(f"  ag 单独有效: {'✅' if standalone_ok else '❌'} | 与value正交: {'✅' if abs(corr)<0.3 else '⚠️'}")
    print(f"  两腿 vs value单独: IR {ir_2:.2f} vs {ir_v:.2f} | 子期3 {s3_2:+.2f} vs {s3_v:+.2f} ({'逆风被扛平滑' if s3_2 > s3_v + 0.1 else '逆风未改善'})")
    adopt = standalone_ok and (abs(corr) < 0.3) and (ir_2 > ir_v + 0.03 or s3_2 > s3_v + 0.1) and te_2 <= te_v + 0.01
    print(f"  -> {'✅ 采用 asset_growth 作正交互补腿 (分散扛逆风)' if adopt else '❌ 不采用 (或单独无效, 或正交不足, 或两腿没改善)'}")
    print("  (采用=单独过硬关 且 正交 且 两腿IR升或子期3逆风被扛平滑 且 TE没崩。这是分散, 不是择时。)")


if __name__ == '__main__':
    main()
