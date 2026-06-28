# -*- coding: utf-8 -*-
"""
value_screen.py — 独立"好公司 + 便宜 + 低位 + 催化"选股 (与 V25 / 潜伏 严格隔离).

隔离原则: 不 import 任何 V25 / 潜伏 的【策略】逻辑; 只共享【只读数据层】(行业表/财务/估值/价格)
          + 复用通用回测引擎 lurking_backtest。本系统自己的打分逻辑独立。

形态 (软漏斗): 质量做【硬池】(几个相关质量条件 AND, count 稳) -> 池内按【行业相对便宜 + 价格低位】
              软排序(rank 相加, 无 15.0 vs 14.8 悬崖) -> top-N。forecast 当 soft bonus, 不当 gate。

★ 纪律: 所有阈值在下方【写死/预注册】, 绝不在看过的回测上回头调。判读照单全收。

接口 (全部 (date,code) 键, 直接喂 lurking_backtest):
  load_industry_name()                          -> code -> (name, industry)  当前快照
  quality_pool(fund)                            -> Series[bool]  模块二 硬质量池
  value_score(db, ind_name)                     -> Series 0~1   模块三 行业相对便宜(软)
  position_score(price)                         -> Series 0~1   模块四 价格低位(软)
  forecast_bonus(fcst_panel)                    -> Series       模块五 加分项(可选)
  screen_alpha(pool, value, position, bonus)    -> Series       综合分(池外 NaN) -> 回测 alpha
"""
import os
import glob

import numpy as np
import pandas as pd

ENGINE = 'fastparquet'
IND_FILE = 'tushare_cache/_partial/industry/stock_industry.parquet'

# ── 冻死的阈值 (pre-registered; 绝不在回测上调) ──────────────────────
Q_FLOORS = dict(
    roe_min=15.0,            # ROE >= 15%  (盈利能力核心)
    npm_min=10.0,            # 销售净利率 >= 10% (定价权)
    debt_max=60.0,           # 资产负债率 <= 60% (财务稳健)
    ocf_to_profit_min=0.8,   # 经营现金流/营业利润 >= 0.8 (利润含金量; 注意分母是营业利润)
    or_yoy_min=10.0,         # 营收同比 >= 10% (成长)
    np_yoy_min=10.0,         # 净利同比 >= 10% (成长)
)
POS_LOOKBACK = 250           # 价格低位回看 ~1 年
ST_TAG = 'ST'                # name 含 'ST' (覆盖 ST/*ST/SST) -> 剔除
POSITIVE_FORECAST = ('预增', '略增', '扭亏', '续盈')   # 模块五 正面预告类型
# ────────────────────────────────────────────────────────────────────


def _xs_rank(s):
    """横截面(同日)百分位 rank, 0~1。"""
    return s.groupby(level='date').rank(pct=True)


def load_industry_name(path=IND_FILE):
    """读行业表 -> DataFrame(index=code, 列=name/industry)。当前快照(非时点);
    ST 用 name, 行业相对估值用 industry。PIT 局限见 README。"""
    df = pd.read_parquet(path, engine=ENGINE)
    return df.drop_duplicates('code').set_index('code')[['name', 'industry']]


def quality_pool(fund, floors=Q_FLOORS):
    """模块二: 硬质量池 (相关质量条件 AND)。用最近一期 PIT fina_indicator;
    NaN(财报未公告) -> 不入池 (不可得即不选)。
    注: '连续2-3年 ROE' 需多期回看, v1 用单期最近值; 多期版见 README 扩展。"""
    f = fund
    m = pd.Series(True, index=f.index)
    if 'roe' in f:              m &= (f['roe'] >= floors['roe_min'])
    if 'netprofit_margin' in f: m &= (f['netprofit_margin'] >= floors['npm_min'])
    if 'debt_to_assets' in f:   m &= (f['debt_to_assets'] <= floors['debt_max'])
    if 'ocf_to_profit' in f:    m &= (f['ocf_to_profit'] >= floors['ocf_to_profit_min'])
    if 'or_yoy' in f:           m &= (f['or_yoy'] >= floors['or_yoy_min'])
    if 'netprofit_yoy' in f:    m &= (f['netprofit_yoy'] >= floors['np_yoy_min'])
    return m.fillna(False)


def quality_score(fund):
    """B1 软质量 (替代硬池): 6 个质量指标各自横截面 rank 后取均值 (0~1, 越高越优)。
    消除硬池 count 不稳 (3~385) 的顺周期塌缩; 且无质量阈值 -> magic number 更少。
    方向: roe/净利率/ocf含金量/营收增/净利增 越高越好; 负债越低越好(取负)。"""
    parts = [_xs_rank(fund[c]) for c in
             ('roe', 'netprofit_margin', 'ocf_to_profit', 'or_yoy', 'netprofit_yoy') if c in fund]
    if 'debt_to_assets' in fund:
        parts.append(_xs_rank(-fund['debt_to_assets']))
    if not parts:
        return pd.Series(np.nan, index=fund.index, name='quality_score')
    return pd.concat(parts, axis=1).mean(axis=1).rename('quality_score').astype('float32')


def value_score(db, ind_name):
    """模块三: 行业相对便宜 (软分, 越高越便宜)。
    rel = 个股 PE / 同日同行业中位 PE (剔 PE<=0); PB 同理; score = mean(rank(-rel_pe), rank(-rel_pb))。
    用行业中位而非绝对值 -> 解决'银行PE5 / 科技PE30 都正常'的跨行业不可比。"""
    dates = db.index.get_level_values('date')
    codes = db.index.get_level_values('code')
    industry = pd.Series(codes.map(ind_name['industry']).to_numpy(), index=db.index).fillna('UNKNOWN')
    parts = {}
    for col in ('pe_ttm', 'pb'):
        if col not in db.columns:
            continue
        s = db[col].where(db[col] > 0)                      # 剔非正估值
        tmp = pd.DataFrame({'v': s.to_numpy(), 'd': dates, 'i': industry.to_numpy()})
        med = tmp.groupby(['d', 'i'])['v'].transform('median').to_numpy()
        rel = s.to_numpy() / med                            # <1 = 比同业便宜
        parts[col] = _xs_rank(pd.Series(rel, index=s.index))
    if not parts:
        return pd.Series(np.nan, index=db.index, name='value_score')
    return pd.concat(parts, axis=1).mean(axis=1).rename('value_score').astype('float32')


def position_score(price, lookback=POS_LOOKBACK):
    """模块四: 价格低位 (软分)。
    off-high = 1 - close/250日高 (离高点越多越大); off-low = close/250日低 - 1 (离低点越多越大)。
    score = mean(rank(off-high), rank(off-low)) -> 偏好'从高点跌下来但已脱离最低点'的甜区,
            两者相抵以压住'一路新低'的 value trap (off-low 低 -> 拉低综合)。"""
    wide = price.unstack('code').sort_index()
    mp = max(lookback // 2, 20)
    roll_max = wide.rolling(lookback, min_periods=mp).max()
    roll_min = wide.rolling(lookback, min_periods=mp).min()
    off_high = (1.0 - wide / roll_max).stack().rename('oh')
    off_low = (wide / roll_min - 1.0).stack().rename('ol')
    sc = pd.concat([_xs_rank(off_high), _xs_rank(off_low)], axis=1).mean(axis=1)
    return sc.rename('position_score').astype('float32')


def forecast_bonus(fcst_panel, bonus=0.5):
    """模块五(加分项, 可选): 最近一期【已公告】预告为正面类型 -> +bonus; 无预告 -> 0(中性, 不剔)。
    fcst_panel: (date,code)->预告类型字符串 (已 PIT 对齐); None 则跳过本模块。"""
    if fcst_panel is None:
        return None
    return (fcst_panel.isin(POSITIVE_FORECAST).astype(float) * bonus).rename('forecast_bonus')


def screen_alpha(pool, value, position, bonus=None):
    """综合: 池内 alpha = value_rank + position_rank (+ forecast_bonus); 池外 -> NaN(不入选)。
    直接喂 lurking_backtest: 它 nlargest(top_n) = 在质量池内选'最便宜+最低位'的前 N 只等权。
    value/position 缺 -> 该股无法评估, alpha NaN 退出; forecast 缺 -> +0 (中性)。"""
    idx = pool.index
    score = value.reindex(idx) + position.reindex(idx)
    if bonus is not None:
        score = score + bonus.reindex(idx).fillna(0.0)
    return score.where(pool).rename('screen_alpha').astype('float32')
