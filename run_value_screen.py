# -*- coding: utf-8 -*-
"""
run_value_screen.py — 独立选股 装配脚本 (与 V25/潜伏 隔离; 仅共享只读数据层 + 复用 lurking_backtest).

链路: 价格(hfq)+行业表+财务+估值 -> [质量] -> 行业相对便宜/价格低位 软分 -> 综合 top-N -> 月度调仓回测

两种形态 (MODE):
  'soft' (B1, 默认/推荐): 质量也软排序, 全市场按 质量+便宜+低位 rank 相加取 top-N。
                          池恒定 N 只、无 count 塌缩; 质量阈值溶进 rank, magic number 最少。
  'hard' (B2): 质量做硬池(6 floor AND) 再池内软排序。已知 count 不稳(3~385), 顺周期塌缩。

v1 = 模块 1~4 (forecast 未接)。阈值在 value_screen 里已冻死, 结果照单全收, 别回头调。
"""
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from lurking_fundamentals import pit_fundamentals          # 共享数据层
from lurking_quality_value import load_daily_basic_ts      # 共享数据层
from lurking_backtest import backtest, _rebalance_dates    # 复用通用回测引擎
import value_screen as VS

MODE = 'soft'              # 'soft'=B1(推荐) / 'hard'=B2(已知池不稳)
FORECAST_PANEL = None      # 拉到 forecast 后传入 (date,code)->预告类型; 现 None = 跑 4/5 模块
TOP_N = 30


def main():
    print(f"① 价格面板(hfq) + 行业/名称表  [MODE={MODE}] ...")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    price = panel['close']
    ind_name = VS.load_industry_name()
    dates = price.index.get_level_values('date').unique().sort_values()
    codes = price.index.get_level_values('code').unique().tolist()
    rebal = _rebalance_dates(dates.tolist())

    print("② 财务 (PIT fina_indicator, 仅调仓日对齐) + ST 剔除 ...")
    fund = pit_fundamentals(pd.DatetimeIndex(rebal), codes)
    name = pd.Series(fund.index.get_level_values('code').map(ind_name['name']).to_numpy(),
                     index=fund.index).fillna('')
    not_st = ~name.str.contains(VS.ST_TAG)                  # ST/*ST (当前name; PIT局限见下)

    print("③ 模块三 行业相对便宜 + 模块四 价格低位 (软分) ...")
    db = load_daily_basic_ts()
    val = VS.value_score(db, ind_name).reindex(fund.index)
    pos = VS.position_score(price).reindex(fund.index)
    bonus = VS.forecast_bonus(FORECAST_PANEL)
    bonus = bonus.reindex(fund.index) if bonus is not None else None

    if MODE == 'hard':
        pool = VS.quality_pool(fund) & not_st
        alpha = VS.screen_alpha(pool, val, pos, bonus)
        sz = pool.groupby(level='date').sum()
        print(f"   质量硬池规模: 均值 {sz.mean():.0f} 只 (区间 {int(sz.min())}~{int(sz.max())})")
    else:
        qual = VS.quality_score(fund)
        elig = not_st & qual.notna()
        alpha = qual + val + pos
        if bonus is not None:
            alpha = alpha + bonus.fillna(0.0)
        alpha = alpha.where(elig).rename('screen_alpha').astype('float32')
        sz = elig.groupby(level='date').sum()
        print(f"   可选域规模(非ST+有财务): 均值 {sz.mean():.0f} 只 (区间 {int(sz.min())}~{int(sz.max())})")

    pickable = alpha.groupby(level='date').apply(lambda s: int(s.notna().sum()))
    print(f"   综合分非空: 均值 {pickable.mean():.0f} 只/调仓日 (min {int(pickable.min())})")

    print(f"\n④ 喂 lurking_backtest (top_n={TOP_N}, 3/6/12月 全样本+留出OOS) ...")
    backtest(price, alpha, top_n=TOP_N)

    print("\n⚠️ v1 注记:")
    print("  · 仅 4/5 模块 (forecast 未接); 接上把 FORECAST_PANEL 传入即可。")
    print("  · ST 用【当前】name、行业用【当前】快照 -> 轻微 PIT 局限。")
    print("  · 退市/幸存者依赖数据湖是否含退市股 (load_universe_audit)。")
    print("  · 判读: 留出 OOS 在 3/6/12 月方向稳健为正才算数; 阈值已冻, 别回头调。")


if __name__ == '__main__':
    main()
