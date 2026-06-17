# -*- coding: utf-8 -*-
"""
lurking_mispricing.py — 潜伏模式 第3块: 错杀程度 + 资金验证 (Layer3).

核心逻辑: 股价大跌 + 基本面稳健(已由Layer1质量池保证) + 资金暗中流入 = 真错杀(非价值陷阱)。
  纯"跌得多"会买到垃圾股; 叠加"聪明钱在跌势中逆向流入", 才是底部吸筹的信号。

慢化原则 (潜伏模式 vs V25 的根本区别):
  V25 是日度短窗(5-20日)。潜伏模式持仓 3-12 个月, 所有窗口【慢化到月频量级】:
  回撤看 1 年(250日), 资金看 60 日累计。短窗噪声对长持仓无意义。

复用现有数据 (不新增下载):
  · 日线湖(close)          -> drawdown_1y, price_pct_1y
  · margin_factors         -> 融资余额动量 (杠杆资金在跌势中是否逆向加仓)
  · moneyflow 原始分层      -> 主力(大+超大单)净流入 60 日累计 / 成交额 (主力是否吸筹)

条件触发 (方案要求, 防噪声): 资金因子【仅在下跌环境中】才计分 —— 仅当个股近 60 日
  跌幅 > 10% 时, 资金流入才被解读为"逆向吸筹"; 否则资金分置中位(0.5), 不参与区分。
  理由: 上涨中的资金流入是追涨(已被价格反映), 只有跌势中的流入才是"错杀验证"。

接口:
  mispricing_score(close_panel, margin_panel, mf_raw_panel) -> Series 0~1, index=(date,code)
  所有成分横截面 rank(0~1) 后加权; 资金数据缺失的股票该成分取中位 0.5 (优雅降级)。
"""
import numpy as np
import pandas as pd

# ── 权重 (写死, 来自方案; 错杀=价格跌+资金验证) ───────────────────────────────────
W_DRAWDOWN = 0.45     # 价格回撤深度 (跌得越多越可能错杀)
W_PRICEPCT = 0.20     # 价格在自身1年区间的位置 (越低越好)
W_MARGIN   = 0.20     # 融资盘逆向加仓 (跌势中融资余额上升)
W_MFLOW    = 0.15     # 主力资金逆向吸筹 (跌势中主力净流入为正)
DROP_GATE  = -0.10    # 资金因子仅在近60日跌幅 > 10% 时生效


def _xs_rank(s):
    """横截面(按日) rank -> 0~1; 全NaN日返回NaN。"""
    return s.groupby(level='date').rank(pct=True)


def mispricing_score(close_panel, margin_panel=None, mf_raw_panel=None):
    """close_panel: Series/DataFrame['close'] index=(date,code); margin/mf 可为 None(降级)。"""
    if isinstance(close_panel, pd.DataFrame):
        close = close_panel['close']
    else:
        close = close_panel
    close = close.sort_index()
    g = close.groupby(level='code', sort=False)

    # ── 价格维度 (慢化: 1 年窗口) ──
    # 1年滚动最高 -> 当前相对最高的回撤 (越深 -> 越可能错杀)
    roll_max_1y = g.transform(lambda x: x.rolling(250, min_periods=120).max())
    drawdown_1y = close / roll_max_1y - 1.0                 # ≤0, 越负回撤越深
    # 1年内价格分位 (越低越好)
    roll_min_1y = g.transform(lambda x: x.rolling(250, min_periods=120).min())
    price_pct_1y = (close - roll_min_1y) / (roll_max_1y - roll_min_1y + 1e-9)
    # 近60日跌幅 (资金因子的触发门)
    ret_60 = close / g.shift(60) - 1.0

    parts = {}
    parts['dd']  = _xs_rank(-drawdown_1y) * W_DRAWDOWN      # 回撤越深(−dd越大)分越高
    parts['ppct'] = _xs_rank(1.0 - price_pct_1y) * W_PRICEPCT

    idx = close.index
    drop_mask = (ret_60 < DROP_GATE).reindex(idx).fillna(False)   # 仅跌势中资金计分

    # ── 资金维度: 融资盘逆向加仓 (慢化, 用 marg_rzye_chg5 = 融资余额5日变化率) ──
    if margin_panel is not None and 'marg_rzye_chg5' in getattr(margin_panel, 'columns', []):
        marg = margin_panel['marg_rzye_chg5'].reindex(idx)
        marg_rank = _xs_rank(marg)                          # 融资余额上升排名高
        # 仅在跌势中计分; 否则中位 0.5
        marg_score = marg_rank.where(drop_mask, 0.5).fillna(0.5)
    else:
        marg_score = pd.Series(0.5, index=idx)
    parts['marg'] = marg_score * W_MARGIN

    # ── 资金维度: 主力(大+超大单)净流入 60 日累计 / 成交额 ──
    if mf_raw_panel is not None and {'buy_lg_amount', 'buy_elg_amount'}.issubset(
            getattr(mf_raw_panel, 'columns', [])):
        mf = mf_raw_panel
        main_net = ((mf['buy_lg_amount'] + mf['buy_elg_amount'])
                    - (mf['sell_lg_amount'] + mf['sell_elg_amount']))
        turn = (mf['buy_lg_amount'] + mf['buy_elg_amount']
                + mf['sell_lg_amount'] + mf['sell_elg_amount']
                + mf.get('buy_sm_amount', 0) + mf.get('buy_md_amount', 0))
        mn = main_net.sort_index().groupby(level='code', sort=False).apply(
            lambda x: x.droplevel('code').rolling(60, min_periods=30).sum()
            if x.index.nlevels > 1 else x.rolling(60, min_periods=30).sum())
        tn = turn.sort_index().groupby(level='code', sort=False).apply(
            lambda x: x.droplevel('code').rolling(60, min_periods=30).sum()
            if x.index.nlevels > 1 else x.rolling(60, min_periods=30).sum())
        mn.index = main_net.sort_index().index              # 复位 index (groupby-apply 可能加层)
        tn.index = turn.sort_index().index
        major_ratio = (mn / (tn.abs() + 1e-9)).reindex(idx)
        mf_rank = _xs_rank(major_ratio)
        mf_score = mf_rank.where(drop_mask, 0.5).fillna(0.5)
    else:
        mf_score = pd.Series(0.5, index=idx)
    parts['mflow'] = mf_score * W_MFLOW

    score = sum(parts.values())
    return score.rename('mispricing_score').astype('float32')
