# -*- coding: utf-8 -*-
"""
run_lurking_fundamental.py —— 装配脚本: 把 基本面 composite 喂进 潜伏 回测引擎。

链路:  价格面板(hfq) + 三表 -> build_factors -> gate(潜伏调仓频率) -> composite -> 潜伏回测(3/6/12月 OOS)

★ 第一版是【纯基本面、未中性化、未与 hidden_alpha_neutral 混合】的 baseline ——
  先看基本面 composite 单独有没有 3/6/12 月 OOS 信号, 再决定中性化 / 混合。不混合, 避免混淆归因。
"""
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
import lurking_fundamental as LF
from lurking_backtest import backtest, _rebalance_dates

# ── 你拥有的 pre-registration 旋钮 (按经济先验定, 不按哪个好看定) ──
GATE_HORIZON_DAYS = 126        # gate 前向收益 horizon (~6mo 交易日). 设成潜伏持有期: 63≈3mo/126≈6mo/252≈12mo
NW_LAG_MONTHS     = 6          # Newey-West 滞后 (单位=调仓数), ≈ horizon 的月数
USE_FACTORS       = None       # 看完 gate 后填幸存因子 list; None = 先用全 5 个跑 baseline
TOP_N             = 30
# ────────────────────────────────────────────────────────────


def build_fwd(price, horizon):
    """price[(date,code)->close] -> 长表 [date, code, fwd_ret] (前向 horizon 交易日)。"""
    wide = price.unstack('code')
    fwd = (wide.shift(-horizon) / wide - 1.0).stack().rename('fwd_ret').reset_index()
    fwd.columns = ['date', 'code', 'fwd_ret']
    return fwd


def main():
    print("① 加载价格面板 (hfq) ...")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']

    print("② 构建 5 个基本面因子 (PIT: 公告日对齐/TTM/首次披露) ...")
    fac = LF.build_factors(panel)
    print("   因子非空率:\n" + (fac.notna().mean() * 100).round(1).to_string())

    print(f"\n③ gate @ horizon={GATE_HORIZON_DAYS}d, 月度调仓, NW_lag={NW_LAG_MONTHS} (读 IC_t_NW, 非 ICIR_full):")
    all_dates = price.index.get_level_values('date').unique().sort_values()
    rebal = _rebalance_dates(all_dates)
    fwd = build_fwd(price, GATE_HORIZON_DAYS)
    # existing: 传你的 hidden_alpha_neutral 或潜伏因子矩阵, 查这套基本面是否只是重述已有 value/quality。
    rep = LF.gate(fac, fwd, existing=None, rebalance_dates=rebal, nw_lag=NW_LAG_MONTHS)
    print(rep.to_string(index=False))
    print("   >>> 把过 IC_t_NW + 正交的因子填进 USE_FACTORS, 再跑一遍精炼 composite。")

    print(f"\n④ composite -> 潜伏回测 (top_n={TOP_N}, 3/6/12月各自 OOS):")
    score = LF.composite(fac, use=USE_FACTORS)
    print(f"   composite 非空 {int(score.notna().sum())}/{len(score)}")
    backtest(price, score, top_n=TOP_N)

    print("\n⚠️ 这是【未中性化 + 未混合 hidden_alpha_neutral】的纯基本面 baseline。")
    print("   若 OOS 见信号 -> 下一步: (a) 按 lurking_synthesis 同款 行业/size 中性化; (b) 再谈与 hidden_alpha 混合。")
    print("   若 OOS 不行 -> 基本面 composite 单独无效, 别硬塞进 潜伏 (同 V25/H13 纪律)。")


if __name__ == '__main__':
    main()
