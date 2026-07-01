# -*- coding: utf-8 -*-
"""
eval_derisk.py — B(F-score补流动比率->8项) + C(商誉排雷), 都从 balancesheet 取数, 隔离测各自效果。

纪律: 这俩基于财务先验(完整Piotroski有流动比率项; 商誉暴雷是A股头号雷), 非对着单季结果调参。
      每个都用16年+三子期+成本硬关测; 过关才采用。绝不因某季好看就采纳。

数据(零重拉, 来自 _partial/balancesheet, PIT用f_ann_date):
  B: 流动比率 = total_cur_assets/total_cur_liab; 第6项 Δ流动比率>0(YoY shift4)。F7+此=F8。
  C: 商誉率 = goodwill/total_hldr_eqy_exc_min_int(净资产); 剔除 > 阈值(预注册)。

测(都decile/季度/vsEW/PIT):
  ① value单独   ② value+F7过滤(基准, 已知)   ③ value+F8过滤(B: 补流动比率有没有更好)
  ④ value+剔高商誉(C单独)   ⑤ value+F7+剔高商誉(C叠加F-score)
  判定: ③比②好 => B采用; ④/⑤降回撤且三子期稳 => C采用。均看三子期+成本, 非单季。
复用 eval_value_longhist / eval_fscore。
"""
import glob

import numpy as np
import pandas as pd

import eval_value_longhist as V
import eval_fscore as FS

BS_DIR = './tushare_cache/_partial/balancesheet'
EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
KEEP = 0.60                 # F-score过滤保留比例(同eval_fscore)
GW_MAX = 0.30              # 商誉/净资产 > 30% 剔除(预注册; 暴雷高发线)
COST = 0.003
N_SUB = 3


def load_bs(rebal):
    """PIT(f_ann_date as-of): 每股最新报告期的 商誉率 + 流动比率YoY变化。"""
    cols = ['ts_code', 'end_date', 'f_ann_date', 'goodwill', 'total_assets',
            'total_hldr_eqy_exc_min_int', 'total_cur_assets', 'total_cur_liab']
    parts = []
    for f in sorted(glob.glob(f'{BS_DIR}/*.parquet')):
        try:
            d = pd.read_parquet(f, engine='fastparquet')
            parts.append(d[[c for c in cols if c in d.columns]])
        except Exception as e:
            print(f"  [warn] {f}: {e!r}")
    df = pd.concat(parts, ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['f_ann_date'].astype(str), errors='coerce')
    df['end'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
    for c in ['goodwill', 'total_assets', 'total_hldr_eqy_exc_min_int', 'total_cur_assets', 'total_cur_liab']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ann', 'end']).sort_values(['code', 'end'])
    df = df.drop_duplicates(['code', 'end'], keep='last')
    eq = df['total_hldr_eqy_exc_min_int']
    df['gw'] = (df['goodwill'].fillna(0) / eq.where(eq > 0)).clip(lower=0)   # 商誉率; 无商誉->0
    df['curr'] = df['total_cur_assets'] / df['total_cur_liab'].where(df['total_cur_liab'] > 0)
    df['curr_chg'] = df['curr'] - df.groupby('code')['curr'].shift(4)        # Δ流动比率 YoY
    df = df.sort_values('ann')
    gw_p, cc_p = [], []
    for dt in rebal:
        sub = df[df['ann'] <= dt]
        if sub.empty:
            continue
        last = sub.groupby('code').last()
        gw_p.append(pd.DataFrame({'date': dt, 'code': last.index, 'gw': last['gw'].values}))
        cc_p.append(pd.DataFrame({'date': dt, 'code': last.index, 'cc': last['curr_chg'].values}))
    gw = pd.concat(gw_p).set_index(['date', 'code'])['gw']
    cc = pd.concat(cc_p).set_index(['date', 'code'])['cc']
    return gw, cc


def _metrics(df, cost, ppy):
    return FS._metrics(df, cost, ppy)


def backtest(circ, wide, st, rebal, select_fn):
    recs, turns = [], []
    prev = pd.Series(dtype=float)
    for i in range(len(rebal) - 1):
        d, d1 = rebal[i], rebal[i + 1]
        try:
            cm = circ.xs(d)
        except KeyError:
            continue
        tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
        tradeable = [c for c in tradeable if c not in st]
        picks = select_fn(d, pd.Index(tradeable))
        if picks is None or len(picks) < 1:
            continue
        try:
            r = (wide.loc[d1, picks] / wide.loc[d, picks] - 1).mean()
            ew = (wide.loc[d1, tradeable] / wide.loc[d, tradeable] - 1).mean()
        except KeyError:
            continue
        new_w = pd.Series(1.0 / len(picks), index=picks)
        idx = prev.index.union(new_w.index)
        turns.append(float((new_w.reindex(idx).fillna(0) - prev.reindex(idx).fillna(0)).abs().sum()))
        prev = new_w
        recs.append({'d1': d1, 'port': r, 'bench': ew})
    df = pd.DataFrame(recs).set_index('d1'); df['turn'] = turns
    return df


def main():
    print("① 构建 value(中性) + F7/F8 + 商誉率 ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind = ind_df['industry']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    wide = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code')
    s = pd.Series(wide.index, index=wide.index)
    rebal = s.groupby([s.index.year, s.index.month]).first().tolist()[::FS.STEP]
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, rebal))

    f7 = FS.load_fscore(rebal)                                  # 7项 (0-7)
    gw, cc = load_bs(rebal)                                     # 商誉率, 流动比率YoY变化
    f8_raw = (f7 + (cc > 0).astype(int).reindex(f7.index).fillna(0))   # F8 = F7 + Δ流动比率>0
    f7_neu = V._xs_rank(V.neutralize(f7.rename('f'), ind, lncap, rebal))
    f8_neu = V._xs_rank(V.neutralize(f8_raw.rename('f'), ind, lncap, rebal))

    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')
             and d in f7_neu.index.get_level_values('date') and d in gw.index.get_level_values('date')]
    ppy = 12.0 / FS.STEP
    print(f"   季度 {len(rebal)} | 商誉率>30%剔除 | F-score过滤保留{KEEP:.0%}\n")

    # 选股函数 (都先在 tradeable 内)
    def cheap_decile(d, tradeable, pool=None):
        v = value_neu.xs(d, level='date').dropna()
        base = pool if pool is not None else tradeable
        vin = v.reindex(base).dropna()
        N = max(1, int(len(tradeable) * TOP_PCT))
        return vin.nlargest(min(N, len(vin))).index if len(vin) else pd.Index([])

    def keep_fscore(d, tradeable, fneu):
        f = fneu.xs(d, level='date').dropna().reindex(tradeable).dropna()
        return f[f >= f.quantile(1 - KEEP)].index if len(f) else pd.Index([])

    def excl_gw(d, codes):
        g = gw.xs(d, level='date') if d in gw.index.get_level_values('date') else pd.Series(dtype=float)
        bad = g[g > GW_MAX].index
        return pd.Index([c for c in codes if c not in bad])

    sel = {
        '①value单独': lambda d, t: cheap_decile(d, t),
        '②value+F7过滤': lambda d, t: cheap_decile(d, t, pool=keep_fscore(d, t, f7_neu)),
        '③value+F8过滤': lambda d, t: cheap_decile(d, t, pool=keep_fscore(d, t, f8_neu)),
        '④value+剔高商誉': lambda d, t: excl_gw(d, cheap_decile(d, t)),
        '⑤value+F7+剔商誉': lambda d, t: excl_gw(d, cheap_decile(d, t, pool=keep_fscore(d, t, f7_neu))),
    }

    print("② 五组对比 (decile, 季度, vsEW, @0.3%, PIT):")
    hdr = f"{'方案':>16}{'超额':>9}{'IR':>7}{'TE':>7}{'回撤':>9}{'子期1':>8}{'子期2':>8}{'子期3':>8}"
    print(hdr); print('-' * len(hdr))
    res = {}
    for name, fn in sel.items():
        bt = backtest(circ, wide, st, rebal, fn)
        exc, te, ir, dd = _metrics(bt, COST, ppy)
        sir = FS._subperiod_ir(bt, ppy)
        res[name] = (ir, dd, te, sir, bt)
        print(f"{name:>16}{exc:>+9.1%}{ir:>7.2f}{te:>7.1%}{dd:>9.1%}" + "".join(f"{x:>8.2f}" for x in sir))

    ir2, dd2 = res['②value+F7过滤'][0], res['②value+F7过滤'][1]
    print(f"\n=== 判定 (都对比基准②value+F7过滤; 看三子期稳健, 非单季) ===")
    # B: F8 vs F7
    ir3, dd3, _, sir3, _ = res['③value+F8过滤']
    b_ok = (ir3 > ir2 + 0.03 or dd3 > dd2 + 0.02) and all(not np.isnan(x) for x in sir3)
    print(f"  B(补流动比率 F8): IR {ir3:.2f} vs F7 {ir2:.2f} | 回撤 {dd3:.1%} vs {dd2:.1%} -> "
          f"{'✅ 8项更好, 采用' if b_ok else '❌ 补流动比率无稳健增量, 维持7项'}")
    # C: goodwill (④对比①value单独; ⑤对比②value+F7)
    ir1, dd1 = res['①value单独'][0], res['①value单独'][1]
    c_bases = {'④value+剔高商誉': ('①value单独', ir1, dd1),
               '⑤value+F7+剔商誉': ('②value+F7过滤', ir2, dd2)}
    for name, (base_name, base_ir, base_dd) in c_bases.items():
        ir_, dd_, _, sir_, _ = res[name]
        better = (dd_ > base_dd + 0.02) or (ir_ > base_ir + 0.03)
        cross = all((not np.isnan(x)) for x in sir_)
        print(f"  C({name} vs {base_name}): IR {ir_:.2f} vs {base_ir:.2f} | 回撤 {dd_:.1%} vs {base_dd:.1%} | "
              f"子期 {sir_[0]:.2f}/{sir_[1]:.2f}/{sir_[2]:.2f} -> "
              f"{'✅ 降回撤/稳, 采用' if (better and cross) else '⚠️ 看回撤是否稳定变浅'}")
    print("  注: 商誉排雷价值主要在'降尾部风险(回撤)', 未必抬IR; 三子期回撤都变浅才算稳。单季不作数。")


if __name__ == '__main__':
    main()
