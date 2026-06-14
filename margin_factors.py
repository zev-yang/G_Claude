# -*- coding: utf-8 -*-
"""
margin_factors.py — 个股融资融券明细 -> 截面因子 (panel, observe-only 候选).

仅供 Factor Lab 观察使用; 不进选股池。每个因子都做了【尺度归一化】(除以滚动自身均值或
取变化率), 因为原始余额是【元】, 直接喂会变成市值代理 (大盘股融资余额天然大) —— 这正是
我们要避免的 size 陷阱。归一化后捕捉的是【杠杆资金的边际行为】, 而非绝对规模。

候选因子 (全部 per (date, code), 仅在两融标的上有值):
  · marg_rzye_chg5   : 融资余额 5 日变化率 = rzye / rzye.shift(5) - 1
                       杠杆多头的加仓/减仓速度 (正交于价量动量?待验证)
  · marg_buy_intensity: 融资买入额 / 融资余额 = rzmre / rzye
                       当日新增杠杆买入相对存量的强度 (换手视角)
  · marg_short_ratio : 融券余额 / (融资余额+融券余额) = rqye / rzrqye
                       空头占比 — 截面上空头押注更重的票 (反向信号?待验证)
  · marg_net_lever5  : (融资买入-融资偿还) 5日和 / 融资余额 = Σ(rzmre-rzche)/rzye
                       净杠杆流入强度

铁律: 这些方向(正/负)全部留给数据判, 不预设符号; 闸门未过即弃, 不回炉。
"""
import os
import glob

import numpy as np
import pandas as pd

_ENGINE = 'fastparquet'
_RAW = ['ts_code', 'trade_date', 'rzye', 'rqye', 'rzmre', 'rzche', 'rzrqye']
MARGIN_FACTORS = ['marg_rzye_chg5', 'marg_buy_intensity', 'marg_short_ratio', 'marg_net_lever5']


def load_margin(src='tushare_cache/_partial/margin_detail'):
    if os.path.isfile(src):
        df = pd.read_parquet(src, columns=_RAW, engine=_ENGINE)
    else:
        files = sorted(glob.glob(os.path.join(src, '*.parquet')))
        if not files:
            raise FileNotFoundError(f"no parquet files in {src}")
        df = pd.concat((pd.read_parquet(f, columns=_RAW, engine=_ENGINE) for f in files),
                       ignore_index=True)
    return df


def build_margin_factors(mg):
    """Raw margin rows -> per-(date, code) panel of normalized margin factors."""
    code = mg['ts_code'].astype(str).str[:6]
    date = pd.to_datetime(mg['trade_date'].astype(str), format='%Y%m%d')
    out = pd.DataFrame({'code': code, 'date': date})
    for c in ['rzye', 'rqye', 'rzmre', 'rzche', 'rzrqye']:
        out[c] = pd.to_numeric(mg[c], errors='coerce').astype('float64')
    out = out.sort_values(['code', 'date']).set_index(['date', 'code'])

    g = out.groupby(level='code')
    rzye = out['rzye'].replace(0, np.nan)
    # 1) 融资余额 5 日变化率
    out['marg_rzye_chg5'] = (out['rzye'] / g['rzye'].shift(5) - 1.0)
    # 2) 融资买入强度 = 当日融资买入额 / 融资余额
    out['marg_buy_intensity'] = (out['rzmre'] / rzye)
    # 3) 空头占比 = 融券余额 / 融资融券总余额
    out['marg_short_ratio'] = (out['rqye'] / out['rzrqye'].replace(0, np.nan))
    # 4) 净杠杆流入强度 = (融资买入-融资偿还) 的 5 日和 / 融资余额
    net_in = out['rzmre'] - out['rzche']
    out['marg_net_lever5'] = (net_in.groupby(level='code').rolling(5).sum().droplevel(0) / rzye)

    keep = MARGIN_FACTORS
    res = out[keep].replace([np.inf, -np.inf], np.nan).astype('float32')
    return res


def margin_panel(src='tushare_cache/_partial/margin_detail'):
    """Convenience: load shards -> factor panel indexed by (date, code). None-safe upstream."""
    return build_margin_factors(load_margin(src))
