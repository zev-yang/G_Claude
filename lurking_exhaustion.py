# -*- coding: utf-8 -*-
"""
lurking_exhaustion.py — 潜伏模式 第4块: 下跌衰竭确认 (Layer 4).

定位 (和前三层的分工): Layer1 保证质量、Layer2 保证便宜、Layer3 保证"错杀+有人吸筹",
  但"便宜+被吸筹"的股票可能还在下跌途中。Layer4 回答最后一个问题:
  【这波下跌是不是已经接近衰竭(卖压枯竭)?】—— 避免"抄在半山腰"。

衰竭信号 (全部慢化, 适配长持仓; 越接近底部分越高):
  1) RSI14 历史分位低     -> 超卖 (动量衰竭)
  2) 缩量程度            -> turnover_20 / turnover_120 低分位 (地量, 卖盘枯竭)
  3) 波动率衰减          -> ATR(20) 分位 + 近 20 日斜率为负 (波动收敛, 恐慌退潮)
  4) 融资买入占比拐头     -> margin_buy_intensity 从低位连续回升 (多头开始试探)
  5) 主力连续流入天数     -> 近 20 日主力净流入天数多 (资金面底部确认)

复用现成数据: 日线湖(close/high/low/vol) + margin_factors + moneyflow helpers。
  缺失的子项 -> 该格中性 0.5, 权重重分配 (优雅降级)。

接口:
  exhaustion_score(close, high, low, vol, margin_panel=None, mf_main_net=None) -> Series 0~1
  其中 mf_main_net = 主力净流入(日频, 正/负)的 (date,code) 面板, 用于数"连续流入天数"。
"""
import numpy as np
import pandas as pd

# ── 权重 (写死) ─────────────────────────────────────────────────────────────────
W_RSI    = 0.25
W_VOLUME = 0.25
W_ATR    = 0.20
W_MARGIN = 0.15
W_INFLOW = 0.15

RSI_WIN_PCT = 500    # RSI 历史分位窗口
ATR_WIN_PCT = 500
INFLOW_LOOKBACK = 20


def _xs_rank(s):
    return s.groupby(level='date').rank(pct=True)


def _per_code(s, fn):
    """对每只股票的时间序列应用 fn(numpy)->numpy, 拼回 (date,code) Series。"""
    parts = []
    for code, g in s.groupby(level='code', sort=False):
        parts.append(pd.Series(fn(g.to_numpy()), index=g.index))
    return pd.concat(parts)


def _rsi(close, n=14):
    def fn(v):
        out = np.full(len(v), np.nan)
        if len(v) < n + 1:
            return out
        d = np.diff(v)
        gain = np.where(d > 0, d, 0.0)
        loss = np.where(d < 0, -d, 0.0)
        ag = np.convolve(gain, np.ones(n) / n, 'valid')
        al = np.convolve(loss, np.ones(n) / n, 'valid')
        rs = ag / np.where(al == 0, np.nan, al)
        rsi = 100 - 100 / (1 + rs)
        out[n:] = rsi[:len(out) - n]
        return out
    return _per_code(close, fn)


def _roll_low_pct(s, window, min_periods=120):
    """s 当前值在过去 window 的分位 (低=底部特征)。"""
    def fn(v):
        out = np.full(len(v), np.nan)
        for i in range(len(v)):
            lo = max(0, i - window + 1)
            w = v[lo:i + 1]; w = w[~np.isnan(w)]
            if len(w) >= min_periods and not np.isnan(v[i]):
                out[i] = (w <= v[i]).mean()
        return out
    return _per_code(s, fn)


def _atr(high, low, close, n=20):
    """简化 ATR: 近 n 日 (high-low) 均值 / close (尺度无关)。"""
    hl = (high - low)
    parts = []
    for code, g in hl.groupby(level='code', sort=False):
        atr = g.rolling(n, min_periods=5).mean()
        parts.append(atr)
    atr = pd.concat(parts)
    return (atr / close.replace(0, np.nan))


def _slope_20(s):
    """近 20 日线性斜率 (负=下行)。"""
    def fn(v):
        out = np.full(len(v), np.nan)
        x = np.arange(20)
        for i in range(19, len(v)):
            w = v[i - 19:i + 1]
            if np.isnan(w).any():
                continue
            out[i] = np.polyfit(x, w, 1)[0]
        return out
    return _per_code(s, fn)


def exhaustion_score(close, high, low, vol, margin_panel=None, mf_main_net=None):
    close = close.sort_index()
    idx = close.index
    comps = {}

    # 1) RSI 低分位 -> 超卖
    rsi = _rsi(close).reindex(idx)
    rsi_pct = _roll_low_pct(rsi, RSI_WIN_PCT)
    comps['rsi'] = (_xs_rank(1.0 - rsi_pct.reindex(idx)), W_RSI)

    # 2) 缩量: turnover_20/turnover_120 低分位 (用 vol 代理换手, 尺度在截面 rank 中消解)
    parts20, parts120 = [], []
    for code, g in vol.groupby(level='code', sort=False):
        parts20.append(g.rolling(20, min_periods=5).mean())
        parts120.append(g.rolling(120, min_periods=30).mean())
    vol20 = pd.concat(parts20); vol120 = pd.concat(parts120)
    vol_ratio = (vol20 / vol120.replace(0, np.nan)).reindex(idx)
    comps['volume'] = (_xs_rank(-vol_ratio), W_VOLUME)     # 缩量(低比值) -> 高分

    # 3) 波动率衰减: ATR 低分位 + 斜率为负
    atr = _atr(high, low, close).reindex(idx)
    atr_pct = _roll_low_pct(atr, ATR_WIN_PCT).reindex(idx)
    atr_slope = _slope_20(atr).reindex(idx)
    atr_score = 0.5 * _xs_rank(1.0 - atr_pct) + 0.5 * _xs_rank(-atr_slope)  # 低波动+收敛
    comps['atr'] = (atr_score, W_ATR)

    # 4) 融资买入占比拐头 (margin_buy_intensity 近5日上升)
    if margin_panel is not None and 'marg_buy_intensity' in getattr(margin_panel, 'columns', []):
        mbi = margin_panel['marg_buy_intensity'].reindex(idx)
        chg = _per_code(mbi, lambda v: np.concatenate([[np.nan] * 5,
                        v[5:] - v[:-5]]) if len(v) > 5 else np.full(len(v), np.nan))
        comps['margin'] = (_xs_rank(chg.reindex(idx)), W_MARGIN)  # 占比回升 -> 高分

    # 5) 主力连续流入天数 (近20日主力净流入为正的天数)
    if mf_main_net is not None:
        mn = mf_main_net.reindex(idx)
        if isinstance(mn, pd.DataFrame):
            mn = mn.iloc[:, 0]
        pos_days = _per_code((mn > 0).astype(float),
                             lambda v: pd.Series(v).rolling(INFLOW_LOOKBACK, min_periods=5)
                             .sum().to_numpy())
        comps['inflow'] = (_xs_rank(pos_days.reindex(idx)), W_INFLOW)

    total_w = sum(w for _, w in comps.values())
    score = sum(s.fillna(0.5) * (w / total_w) for s, w in comps.values())
    return score.rename('exhaustion_score').astype('float32')
