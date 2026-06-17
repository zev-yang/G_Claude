# -*- coding: utf-8 -*-
"""
lurking_synthesis.py — 潜伏模式 第5块: hidden_alpha 合成 + 行业/市值中性化.

合成 (方案权重):
  hidden_alpha = 0.30*quality + 0.25*valuation + 0.25*mispricing + 0.20*technical(exhaustion)
  · quality_score: 由质量池布尔转成"达标程度"分? 不 —— Layer1 是硬过滤(进/不进池),
    这里 quality 作为【池内】的一个软分: 用 ROE 的横截面 rank 作质量强度 (池外股票不参与)。
  · 其余三项已是 0~1 分, 直接加权。某项 NaN -> 该项中性 0.5, 权重重分配。

★ 行业/市值中性化 (必须在选股前做, 方案强调):
  不中性化 -> 错杀+低估的票会扎堆在某个超跌行业(如地产/医药)或全是小盘股,
  组合变成"赌一个行业/赌小盘", 而非"选个股"。中性化 = 在每个截面上, 把 hidden_alpha
  对【行业哑变量 + ln(市值)】做横截面回归, 取残差 -> 剔除行业和市值的系统性影响,
  只留下"同行业、同市值档里相对更优"的个股 alpha。

接口:
  hidden_alpha(quality_mask, val_score, mis_score, exh_score, roe_panel) -> 合成分 (date,code)
  neutralize(alpha, industry_map, lncap_panel)                          -> 中性化残差 (date,code)
  其中 industry_map: code->行业 (来自 fetch_industry.load_industry);
       lncap_panel:  (date,code)->ln(circ_mv) (来自 daily_basic)。
"""
import numpy as np
import pandas as pd

W_QUALITY = 0.30
W_VALUE   = 0.25
W_MISPRICE = 0.25
W_TECH    = 0.20


def _xs_rank(s):
    return s.groupby(level='date').rank(pct=True)


def hidden_alpha(q_mask, val_score, mis_score, exh_score, roe_panel):
    """四层合成 -> 0~1 区间的 hidden_alpha; 仅在质量池内有值(池外 NaN)。"""
    idx = q_mask.index
    # 质量软分: 池内用 ROE 横截面 rank; 池外置 NaN (不入选)
    roe = roe_panel.reindex(idx)
    q_soft = _xs_rank(roe.where(q_mask))                       # 池内 ROE 越高分越高

    comps = {
        'quality':  (q_soft, W_QUALITY),
        'value':    (val_score.reindex(idx), W_VALUE),
        'misprice': (mis_score.reindex(idx), W_MISPRICE),
        'tech':     (exh_score.reindex(idx), W_TECH),
    }
    total_w = sum(w for _, w in comps.values())
    alpha = sum(s.fillna(0.5) * (w / total_w) for s, w in comps.values())
    # 只在质量池内保留 (池外 -> NaN, 不参与选股)
    return alpha.where(q_mask).rename('hidden_alpha').astype('float32')


def neutralize(alpha, industry_map, lncap_panel):
    """对 hidden_alpha 做【行业 + 市值】横截面中性化, 返回残差 (越大=同行业同市值里相对更优)。

    每个交易日独立做一次 OLS: alpha ~ 行业哑变量 + ln(市值), 取残差。
    行业或市值缺失的股票退化为"只减去当日全市场均值"(优雅降级, 不丢样本)。
    """
    df = pd.DataFrame({'alpha': alpha}).dropna(subset=['alpha'])
    if df.empty:
        return alpha.rename('hidden_alpha_neutral')
    codes = df.index.get_level_values('code')
    df['industry'] = pd.Series(codes.map(industry_map), index=df.index).fillna('UNKNOWN')
    df['lncap'] = lncap_panel.reindex(df.index)

    out = []
    for date, g in df.groupby(level='date'):
        y = g['alpha'].to_numpy(dtype=float)
        n = len(g)
        if n < 5:
            out.append(pd.Series(y - y.mean(), index=g.index))      # 样本太少: 仅去均值
            continue
        # 设计矩阵: 截距 + 行业哑变量(drop one) + 标准化ln市值
        ind_d = pd.get_dummies(g['industry'], drop_first=True).to_numpy(dtype=float)
        cap = g['lncap'].to_numpy(dtype=float)
        cap = np.where(np.isnan(cap), np.nanmean(cap), cap)
        if np.all(np.isnan(cap)):
            cap = np.zeros(n)
        else:
            sd = cap.std()
            cap = (cap - cap.mean()) / sd if sd > 1e-9 else np.zeros(n)
        X = np.column_stack([np.ones(n), ind_d, cap]) if ind_d.size else np.column_stack([np.ones(n), cap])
        try:
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ beta
        except Exception:
            resid = y - y.mean()
        out.append(pd.Series(resid, index=g.index))

    neut = pd.concat(out)
    return neut.reindex(alpha.index).rename('hidden_alpha_neutral').astype('float32')
