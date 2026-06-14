# -*- coding: utf-8 -*-
"""
horizon_sweep.py — 扫描持仓周期 {3,5,8,13,20}, 每个 horizon 独立算 target + 独立留出期 OOS。

目的 (纪律): 回答"这套因子在哪个持仓周期上有【样本外】alpha", 看【区间稳健性】——
  · 若某区间(如 10-15天)留出期普遍为正且平滑过渡 -> 真信号, 8天只是没踩对周期;
  · 若所有 horizon 留出期都负 -> 不是周期问题, 是方法本身在 A 股日频价量上没有稳健 alpha。
铁律: 绝不挑"留出期最高"的单点当生产参数 (那是对 horizon 调参/snooping)。判据是
      【区间内方向一致 + 平滑】, 不是【单点最大值】。

实现: 完整复用现有引擎 (load -> AlphaLabV25_1.run -> DailyAuditor.run_simulation), 每个
  horizon 覆盖 CONFIG['horizon'] 后重跑特征工程(target 依赖 horizon)。不改任何核心文件。
  跑前强制 AB/Lab 关闭 -> 纯 42 因子基线。每个 horizon ≈ 一次完整回测, 5 个 ≈ 过夜。

用法:  python horizon_sweep.py
"""
import sys
import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from factors import AlphaLabV25_1
from backtest import DailyAuditor

HORIZONS = [3, 5, 8, 13, 20]
HOLDOUT_FRAC = 0.70          # 前70% 开发期, 后30% 留出期(OOS)


def _seg(strat, h):
    """单段(全样本/开发/OOS)的 CAGR / Sharpe / MaxDD, 与 analyze() 同公式。"""
    strat = strat.dropna()
    if len(strat) < 2:
        return float('nan'), float('nan'), float('nan')
    af = 252.0 / h
    ann = strat.mean() * af
    sh = ann / (strat.std() * np.sqrt(af) + 1e-9)
    eq = (1 + strat).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return ann, sh, dd


def main():
    # 强制纯基线: 关掉 A/B 与观察舱 (横扫的是干净的 42 因子方法本身)
    CONFIG['AB_NEW_FACTORS'] = False
    CONFIG['USE_FACTOR_LAB'] = False

    print("📋 [horizon sweep] Loading data lake once ...")
    panel0 = load_universe_audit(CONFIG['stock_data_path'])
    print(f"  > Loaded: {len(panel0):,} rows\n")

    rows = []
    for h in HORIZONS:
        CONFIG['horizon'] = h
        print(f"{'='*78}\n=== HORIZON = {h} 天 (重算 target + 特征 + 模拟) ===")
        eng = AlphaLabV25_1()
        panel, feats = eng.run(panel0.copy())          # copy: 防引擎就地改写污染下一个 horizon
        auditor = DailyAuditor(panel, feats)            # check_ic 可跳过: run_simulation 不依赖它
        res_df, _ = auditor.run_simulation()
        strat = res_df['Strat']
        n = len(strat)
        cut = int(n * HOLDOUT_FRAC)
        full = _seg(strat, h)
        dev = _seg(strat.iloc[:cut], h)
        oos = _seg(strat.iloc[cut:], h)
        rows.append(dict(h=h, n=n,
                         full_cagr=full[0], full_sh=full[1], full_dd=full[2],
                         dev_cagr=dev[0], dev_sh=dev[1],
                         oos_cagr=oos[0], oos_sh=oos[1], oos_dd=oos[2]))
        print(f"  H={h}: 窗口={n} | 全样本 CAGR {full[0]:+.1%} Sh {full[1]:.2f} "
              f"| 开发 {dev[0]:+.1%}/{dev[1]:.2f} | 留出OOS {oos[0]:+.1%}/{oos[1]:.2f}")
        del panel, eng, auditor, res_df

    # ── 汇总表: 重点看 OOS 列的区间稳健性 ──
    print(f"\n\n{'='*78}\n=== HORIZON SWEEP 汇总 (判据=OOS区间稳健性, 非单点最大) ===")
    print(f"{'H':>4}{'窗口':>6}{'全样本CAGR':>12}{'全样本Sh':>10}"
          f"{'留出CAGR':>11}{'留出Sh':>9}{'留出MaxDD':>11}")
    for r in rows:
        print(f"{r['h']:>4}{r['n']:>6}{r['full_cagr']:>11.1%}{r['full_sh']:>10.2f}"
              f"{r['oos_cagr']:>10.1%}{r['oos_sh']:>9.2f}{r['oos_dd']:>11.1%}")
    print("\n判读指南:")
    print("  ✅ 真信号: 某区间(连续几个 horizon)留出 CAGR/Sh 普遍为正且平滑过渡;")
    print("  ❌ 方法到顶: 所有 horizon 留出期都为负 -> 不是周期问题, 是日频价量在A股没稳健alpha;")
    print("  ⚠️  绝不挑留出最高的单点当生产参数 — 那是对 horizon 调参。看区间, 不看单点。")


if __name__ == '__main__':
    main()
