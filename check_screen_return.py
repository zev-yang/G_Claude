# -*- coding: utf-8 -*-
"""
check_screen_return.py — 看 F-score 过滤【这一季】到底干了什么: 4/1选池 -> 持有到最新, 三组对比。

★定位(务必记住): 1个季度=1个样本=纯噪声, 说明不了策略好坏。F-score价值已由16年回测+稳健性三关
  定性(去陷阱、降回撤、组合增量弱)。本报告是看"过滤机制这季的表现", 不是重判策略。

★PIT(无未来函数): 池子按【买入日4/1当时】可得数据选(value as-of 4/1, F-score ann<=4/1), 持有到最新卖出点。

三组对比(都4/1选、持有到最新):
  ① 纯value池      = 最便宜decile (不加F-score)
  ② value+F过滤池  = decile ∩ F>=5 (加了过滤; 你实际用的下限)
  ③ 被剔除的票     = decile ∩ F<5 (F-score判定不健康, 过滤掉的)
  -> 看 ①vs② 过滤改变多少; 看③是否比②差(剔除的不健康票若跌更惨 = 去陷阱奏效的直接证据)。
复用 eval_value_longhist 构建块。跑: python check_screen_return.py
"""
import glob
import os

import numpy as np
import pandas as pd

import eval_value_longhist as V

FUND_DIR = './tushare_cache/_partial/fundamentals'
LH = './tushare_cache/_longhist'
EXCL_SMALL_PCT = 0.30
CHEAP_PCT = 0.10
FSCORE_HEALTHY = 5


def fscore_asof(buy_date):
    fs = sorted(glob.glob(f'{FUND_DIR}/*.parquet'))
    df = pd.concat([pd.read_parquet(f, engine='fastparquet') for f in fs], ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
    df['end'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
    for c in ['roa', 'ocfps', 'ocf_to_profit', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ann', 'end', 'roa']).sort_values(['code', 'end'])
    g = df.groupby('code')
    F = [(df['roa'] > 0), (df['ocfps'] > 0), ((df['roa'] - g['roa'].shift(4)) > 0),
         (df['ocf_to_profit'] > 1), ((df['debt_to_assets'] - g['debt_to_assets'].shift(4)) < 0),
         ((df['grossprofit_margin'] - g['grossprofit_margin'].shift(4)) > 0),
         ((df['assets_turn'] - g['assets_turn'].shift(4)) > 0)]
    df['fscore'] = sum(x.astype(int) for x in F)
    sub = df[df['ann'] <= buy_date]
    return sub.loc[sub.groupby('code')['end'].idxmax()].set_index('code')['fscore']


def _stat(rets, bench, name_map, label):
    if len(rets) == 0:
        print(f"  {label}: 空"); return np.nan
    port = float(rets.mean())
    line = f"  {label:<16} ({len(rets):>3}只): {port:+.2%}"
    if not np.isnan(bench):
        line += f" | 超额 {port - bench:+.2%}"
    line += f" | 胜率 {(rets > 0).mean():.0%} | 中位 {rets.median():+.1%}"
    print(line)
    return port


def main():
    print("① 构建 中性化 value (PIT 选池) ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind, name_map = ind_df['industry'], ind_df['name']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])

    dates = value_raw.index.get_level_values('date').unique().sort_values()
    latest = dates.max()
    s = pd.Series(dates, index=dates)
    monthly = s.groupby([s.index.year, s.index.month]).first()
    qd = monthly[pd.Index(monthly.index.get_level_values(1)).isin([1, 4, 7, 10])].tolist()
    buy_date = max(d for d in qd if d <= latest)
    print(f"   数据至 {latest.date()} | 买入日(最近季度调仓) {buy_date.date()}\n")

    # PIT 选池 (4/1 当时数据)
    vneu = V._xs_rank(V.neutralize(value_raw, ind, lncap, [buy_date])).xs(buy_date, level='date').dropna()
    cm = circ.xs(buy_date)
    domain = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    domain = [c for c in domain if c not in st and c in vneu.index]
    vd = vneu.reindex(domain).dropna()
    decile = list(vd[vd >= vd.quantile(1 - CHEAP_PCT)].index)      # 纯value池
    fsc = fscore_asof(buy_date)
    kept = [c for c in decile if c in fsc.index and fsc.loc[c] >= FSCORE_HEALTHY]   # 过滤后
    removed = [c for c in decile if c in fsc.index and fsc.loc[c] < FSCORE_HEALTHY] # 被剔除
    print(f"② 4/1 PIT池: 纯value decile {len(decile)} 只 -> 过滤后(F>={FSCORE_HEALTHY}) {len(kept)} 只 | "
          f"剔除(F<{FSCORE_HEALTHY}) {len(removed)} 只\n")

    # 后复权收益 买入->最新
    px = V.build_hfq(V._read('daily', ['close']), V._read('adj_factor', ['adj_factor'])).unstack('code').sort_index()
    d0 = px.index[px.index >= buy_date][0]; d1 = px.index[-1]
    def rets_of(codes):
        cc = [c for c in codes if c in px.columns]
        return (px.loc[d1, cc] / px.loc[d0, cc] - 1).dropna()
    r_all, r_kept, r_rm = rets_of(decile), rets_of(kept), rets_of(removed)

    bench = np.nan
    ipath = f'{LH}/index/000300.parquet'
    if os.path.exists(ipath):
        idx = pd.read_parquet(ipath, engine='fastparquet'); idx['date'] = pd.to_datetime(idx['trade_date'])
        ic = idx.set_index('date')['close'].sort_index()
        bench = float(ic.asof(d1) / ic.asof(d0) - 1)

    print(f"=== 持有 {d0.date()} -> {d1.date()} ({(d1 - d0).days}天) | F-score过滤这季干了什么 ===")
    if not np.isnan(bench):
        print(f"  同期沪深300: {bench:+.2%}\n")
    p_all = _stat(r_all, bench, name_map, "① 纯value池")
    p_kept = _stat(r_kept, bench, name_map, "② value+F过滤")
    p_rm = _stat(r_rm, bench, name_map, "③ 被剔除(不健康)")

    print(f"\n  ▶ 过滤效果 (②−①): {p_kept - p_all:+.2%}  "
          f"({'过滤这季改善了收益' if p_kept > p_all else '过滤这季略降收益' if p_kept < p_all else '几乎无差'})")
    if not np.isnan(p_rm):
        print(f"  ▶ 去陷阱验证 (③被剔除 vs ②保留): 剔除 {p_rm:+.2%} vs 保留 {p_kept:+.2%}  "
              f"({'✅ 剔除的不健康票确实更差, 去陷阱奏效' if p_rm < p_kept else '✗ 这季剔除的反而更好(单季噪声)'})")
        worst = r_rm.sort_values().head(5)
        print(f"  ▶ 被剔除票里最差5只(本该是要避的陷阱?): " +
              '  '.join(f"{c}({name_map.get(c,'')}){v:+.0%}" for c, v in worst.items()))

    print("\n" + "=" * 60)
    print("⚠️ 怎么读(重要, 别被一季带偏):")
    print("  · 1季=1样本=纯噪声。F-score价值已由16年回测+稳健性三关定性, 不靠这季重判。")
    print("  · '过滤效果'这季多半很小(与稳健性结论一致: 组合增量弱)——它的价值在长期降回撤, 非单季。")
    print("  · '去陷阱验证'看③vs②: 长期看剔除的该更差; 但单季可能反过来(噪声), 别据此下结论。")
    print("  · 用途: 看清过滤机制 + 感受value逆风期波动(练心态)。不是评判策略对错。")


if __name__ == '__main__':
    main()
