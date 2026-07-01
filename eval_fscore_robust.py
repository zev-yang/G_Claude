# -*- coding: utf-8 -*-
"""
eval_fscore_robust.py — F-score 稳健性三关: 验上一条"全样本改善但子期2未转正"是真增量还是侥幸。

背景: F-score 全样本 IR 0.12->0.39/0.35, 回撤 -10.6%->-5.5%, TE稳, 与value正交。机械判定因子期2未转正卡住,
      但分析: 子期2的负来自 value 自身(value单独子期2=-0.66), F-score 已把它从-0.66救到-0.16。
      故子期2为负不是 F-score 投机, 是 value 逆风。本脚本用更严的关验"F-score是否真的持续在帮value"。

三关(写死, 不调参):
  ① 阈值敏感性: 过滤保留50/60/70% 是否都改善value (不能只在60%灵)。
  ② 滚动增量: F-score对value的增量(enhanced-value)是否普遍为正(单期为正占比 + 滚动均值为正占比)。
            -> 这是对"子期2为负"的精细回答: 若增量普遍为正, 说明F-score持续帮value, 子期2负属value自身。
  ③ 成本敏感性: 0.3% vs 0.5% 下 F-score 是否仍改善value。
判定: 三关都过 -> F-score是稳健增量(尽管子期2绝对值仍负, 因那是value自身逆风), 采用。否则维持纯value。
复用 eval_fscore / eval_value_longhist。
"""
import numpy as np
import pandas as pd

import eval_value_longhist as V
import eval_fscore as FS

THRESHOLDS = (0.50, 0.60, 0.70)
COSTS = (0.003, 0.005)
ROLL = 8   # 滚动窗口(期); STEP=3 季度 -> 8期≈2年


def _excess(df, cost):
    net = df['port'] - df['turn'] * cost
    return (net - df['bench']).rename('exc')


def build():
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
    fsc = FS.load_fscore(rebal)
    fscore_neu = V._xs_rank(V.neutralize(fsc.rename('fs'), ind, lncap, rebal))
    rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')
             and d in fscore_neu.index.get_level_values('date')]
    ppy = 12.0 / FS.STEP
    return value_neu, fscore_neu, circ, wide, st, rebal, ppy


def main():
    print("构建 value(中性) + F-score(中性) ...")
    value_neu, fscore_neu, circ, wide, st, rebal, ppy = build()
    print(f"   季度 {len(rebal)} | ppy={ppy:.0f}\n")

    bt_v = FS.backtest(value_neu, circ, wide, st, rebal, 'value')
    ir_v0, dd_v0 = FS._metrics(bt_v, 0.003, ppy)[2], FS._metrics(bt_v, 0.003, ppy)[3]
    sir_v = FS._subperiod_ir(bt_v, ppy)
    print(f"基准 value单独: IR {ir_v0:.2f} | 回撤 {dd_v0:.1%} | 子期 {sir_v[0]:.2f}/{sir_v[1]:.2f}/{sir_v[2]:.2f}\n")

    # ① 阈值敏感性
    print("【关①】阈值敏感性: F-score过滤在 保留50/60/70% 下是否都改善value")
    print(f"{'保留比例':>10}{'IR':>8}{'回撤':>9}{'子期1':>8}{'子期2':>8}{'子期3':>8}{'改善?':>8}")
    thr_pass = True
    for thr in THRESHOLDS:
        FS.FSCORE_KEEP = thr
        bt = FS.backtest(None, circ, wide, st, rebal, 'filter', value_neu=value_neu, fscore_neu=fscore_neu)
        _, _, ir, dd = FS._metrics(bt, 0.003, ppy)
        sir = FS._subperiod_ir(bt, ppy)
        ok = (ir > ir_v0 + 0.03) or (dd > dd_v0 + 0.02)
        thr_pass = thr_pass and ok
        print(f"{thr:>9.0%}{ir:>8.2f}{dd:>9.1%}{sir[0]:>8.2f}{sir[1]:>8.2f}{sir[2]:>8.2f}{'✅' if ok else '❌':>8}")
    print(f"  -> 阈值关: {'✅ 三个阈值都改善' if thr_pass else '❌ 改善不稳(依赖特定阈值)'}\n")

    # ② 滚动增量 (@0.3%, 保留60%)
    FS.FSCORE_KEEP = 0.60
    bt_f = FS.backtest(None, circ, wide, st, rebal, 'filter', value_neu=value_neu, fscore_neu=fscore_neu)
    bt_c = FS.backtest(0.5 * value_neu + 0.5 * fscore_neu, circ, wide, st, rebal, 'composite')
    ev = _excess(bt_v, 0.003)
    print("【关②】滚动增量: F-score对value的增量(enhanced-value)是否普遍为正")
    print("  (回应'子期2为负': 若增量普遍为正, 说明F-score持续帮value, 子期2负属value自身逆风)")
    roll_pass = True
    for label, btx in (('过滤', bt_f), ('复合', bt_c)):
        ex = _excess(btx, 0.003)
        incr = (ex - ev).dropna()
        if len(incr) < ROLL + 2:
            print(f"  {label}: 样本不足"); roll_pass = False; continue
        t = incr.mean() / (incr.std() / np.sqrt(len(incr)) + 1e-12)
        frac = (incr > 0).mean()
        roll = incr.rolling(ROLL).mean().dropna()
        roll_frac = (roll > 0).mean()
        ok = (t > 1.5) and (frac > 0.55) and (roll_frac > 0.7)
        roll_pass = roll_pass and ok
        print(f"  {label}: 增量年化 {incr.mean()*ppy:+.1%} | t={t:.2f} | 单期为正 {frac:.0%} | "
              f"{ROLL}期滚动均值为正 {roll_frac:.0%} {'✅' if ok else '❌'}")
    print(f"  -> 滚动关: {'✅ 增量普遍为正(持续帮value)' if roll_pass else '❌ 增量不稳/集中某段'}\n")

    # ③ 成本敏感性
    print("【关③】成本敏感性: 0.3% vs 0.5% 下 F-score 是否仍改善value")
    print(f"{'成本':>8}{'value':>9}{'+过滤':>9}{'+复合':>9}{'过滤仍改善?':>14}")
    cost_pass = True
    for cost in COSTS:
        irv = FS._metrics(bt_v, cost, ppy)[2]
        irf = FS._metrics(bt_f, cost, ppy)[2]
        irc = FS._metrics(bt_c, cost, ppy)[2]
        ok = irf > irv + 0.03
        cost_pass = cost_pass and ok
        print(f"{cost:>8.1%}{irv:>9.2f}{irf:>9.2f}{irc:>9.2f}{'✅' if ok else '❌':>14}")
    print(f"  -> 成本关: {'✅ 0.5%下仍改善' if cost_pass else '❌ 高成本下改善消失'}\n")

    # 综合判定
    print("=" * 60)
    print(f"=== F-score 稳健性综合判定 ===")
    print(f"  关① 阈值: {'✅' if thr_pass else '❌'} | 关② 滚动增量: {'✅' if roll_pass else '❌'} | 关③ 成本: {'✅' if cost_pass else '❌'}")
    if thr_pass and roll_pass and cost_pass:
        print("  -> ✅✅ F-score 是稳健增量。子期2绝对值仍负, 但那是value自身逆风, F-score三关证明它持续在帮value。")
        print("     采用: 写进生产组合选股层(value便宜 ∩ F-score健康); 同时可作集中选股避雷短名单。")
    elif thr_pass and cost_pass and not roll_pass:
        print("  -> ⚠️ 阈值+成本过, 但滚动增量不够普遍。F-score可能只在某些时段帮value -> 谨慎, 倾向不进组合,")
        print("     但若回撤稳定变浅, 仍可作集中选股的避雷短名单(你的实际用途)。")
    else:
        print("  -> ❌ 未通过稳健性。F-score在A股无稳健组合增量, 维持纯value。")
        print("     (但其去陷阱效果若降回撤, 仍可单独作集中选股的健康度参考。)")
    print("  注: 全程预注册阈值/窗口/成本, 未调参。")


if __name__ == '__main__':
    main()
