# -*- coding: utf-8 -*-
"""
screen_repurchase.py — 回购候选【打分 watchlist】(2026), 按研报四步实现可从结构化数据做的部分。

定位: 当下候选清单 + 尽调起点, 不是验证过的 alpha。阈值是研报给定(预注册), 非回测拟合。
数据: repurchase_2026 (fetch_repurchase) + daily_basic 分片 + fina_indicator 分片 + 价格(load_universe_audit)。
所有 enrichment 按【公告日 as-of】对齐 (PIT)。

四步落地:
  step1 硬筛: 回购金额/市值 >= 1%  AND  proc ∈ {实施,完成} (剔预案/股东大会通过/停止/忽悠式)。
              [缺失] 回购用途(注销/激励) & 资金来源(自有/贷款): 接口无此字段, 需公告文本 -> 本工具不做, 标 N/A。
  step2 质地: ROE / PB(破净 PB<1) / 现金质量(ocf_to_profit, 作 FCF 近似) / 股息率(dv_ttm, 5年稳定性需 dividend 接口, 此处仅当前值)。
  step3 信心强度 = rank(回购股本比例 vol/总股本) + rank(回购溢价 high_limit/公告前收盘-1)。
  step4 择时: 现价 vs 回购成本区间(low_limit~high_limit) -> near_cost 标记; 事件前~2月跌幅。
输出: 控制台排序表 + repurchase_watchlist.csv。
"""
import os
import glob

import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit

ENGINE = 'fastparquet'
RP_FILE = './tushare_cache/_partial/repurchase/repurchase_2026.parquet'
DB_DIR = './tushare_cache/_partial/daily_basic'
FUND_DIR = './tushare_cache/_partial/fundamentals'
OUT_CSV = './repurchase_watchlist.csv'

# ── 预注册阈值 (研报给定, 非回测拟合) ──
AMT_MKT_MIN = 0.01            # 回购金额/市值 >= 1%
VALID_PROC = ('实施', '完成')  # 已实施/完成
DRAWDOWN_DAYS = 60           # 事件前 ~2 月 (日历日)
# ──────────────────────────────────


def _read_partials(d, cols):
    fs = sorted(glob.glob(os.path.join(d, '*.parquet')))
    if not fs:
        raise FileNotFoundError(f"无分片: {d}")
    return pd.concat([pd.read_parquet(f, engine=ENGINE) for f in fs], ignore_index=True)[cols]


def _asof_by_code(left, right, left_dt, right_dt, vals):
    """按 code, 取 right 在 left_dt 之前(含)最新的 vals。"""
    r = right.dropna(subset=[right_dt]).sort_values(right_dt)
    return pd.merge_asof(left.sort_values(left_dt), r[['code', right_dt] + vals],
                         left_on=left_dt, right_on=right_dt, by='code', direction='backward')


def main():
    rp = pd.read_parquet(RP_FILE, engine=ENGINE)
    rp['ann_date'] = pd.to_datetime(rp['ann_date'])
    rp['code'] = rp['ts_code'].astype(str).str[:6]
    rp = rp.dropna(subset=['ann_date', 'amount'])
    print(f"载入回购事件: {len(rp)} 行 / {rp['code'].nunique()} 家")

    # ── daily_basic as-of: total_mv(万元) / total_share(万股) / pb / dv_ttm ──
    db = _read_partials(DB_DIR, ['trade_date', 'ts_code', 'total_mv', 'total_share', 'pb', 'dv_ttm'])
    db['trade_date'] = pd.to_datetime(db['trade_date'])
    db['code'] = db['ts_code'].astype(str).str[:6]
    db = db[db['trade_date'] >= rp['ann_date'].min() - pd.Timedelta(days=420)]
    rp = _asof_by_code(rp, db, 'ann_date', 'trade_date', ['total_mv', 'total_share', 'pb', 'dv_ttm'])

    # ── fina_indicator as-of: roe / ocf_to_profit(FCF 近似) ──
    fund = _read_partials(FUND_DIR, ['ts_code', 'ann_date', 'roe', 'ocf_to_profit'])
    fund['f_ann'] = pd.to_datetime(fund['ann_date'])
    fund['code'] = fund['ts_code'].astype(str).str[:6]
    rp = _asof_by_code(rp.drop(columns=[c for c in ('trade_date',) if c in rp]),
                       fund, 'ann_date', 'f_ann', ['roe', 'ocf_to_profit'])

    # ── 价格: 公告前收盘 / 事件前2月收盘 / 最新收盘 ──
    price = load_universe_audit(CONFIG['stock_data_path'])['close']
    pl = price.reset_index(); pl.columns = ['date', 'code', 'close']
    pre = _asof_by_code(rp.assign(pre_dt=rp['ann_date'] - pd.Timedelta(days=1)),
                        pl.rename(columns={'date': 'd', 'close': 'pre_close'}), 'pre_dt', 'd', ['pre_close'])
    rp = pre.drop(columns=['d', 'pre_dt'])
    bk = _asof_by_code(rp.assign(bk_dt=rp['ann_date'] - pd.Timedelta(days=DRAWDOWN_DAYS)),
                       pl.rename(columns={'date': 'd2', 'close': 'close_2mo'}), 'bk_dt', 'd2', ['close_2mo'])
    rp = bk.drop(columns=['d2', 'bk_dt'])
    cur = pl.sort_values('date').groupby('code')['close'].last().rename('cur_close')
    rp = rp.merge(cur, on='code', how='left')

    # ── 指标 ──
    rp['回购金额比例'] = rp['amount'] / (rp['total_mv'] * 1e4)              # 元 / (万元*1e4)
    rp['回购股本比例'] = rp['vol'] / (rp['total_share'] * 1e4)             # 股 / (万股*1e4)
    rp['回购溢价'] = rp['high_limit'] / rp['pre_close'] - 1.0
    rp['破净'] = rp['pb'] < 1.0
    rp['事件前2月跌幅'] = rp['pre_close'] / rp['close_2mo'] - 1.0
    rp['现价低于回购上限'] = rp['cur_close'] <= rp['high_limit']            # step4 择时: 潜伏在回购成本附近/以下

    # ── step1 硬筛 ──
    keep = rp[(rp['回购金额比例'] >= AMT_MKT_MIN) & (rp['proc'].isin(VALID_PROC))].copy()
    # 同一公司多条 -> 取回购金额比例最大的一条 (最有诚意的那次)
    keep = keep.sort_values('回购金额比例', ascending=False).drop_duplicates('code', keep='first')
    print(f"step1 硬筛后 (金额/市值>=1% 且 已实施/完成): {len(keep)} 家")

    # ── step3 信心强度 = rank(股本比例) + rank(溢价) ──
    keep['信心强度'] = keep['回购股本比例'].rank(pct=True) + keep['回购溢价'].rank(pct=True)
    keep = keep.sort_values('信心强度', ascending=False)

    cols = ['ts_code', 'ann_date', 'proc', '回购金额比例', '回购股本比例', '回购溢价',
            '信心强度', 'roe', 'pb', '破净', 'dv_ttm', 'ocf_to_profit',
            '事件前2月跌幅', 'cur_close', 'high_limit', 'low_limit', '现价低于回购上限']
    out = keep[cols].reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    pd.set_option('display.width', 200, 'display.max_columns', 30)
    print(f"\n=== 回购候选 watchlist (信心强度 Top 20) ===")
    show = out.head(20).copy()
    for c in ('回购金额比例', '回购股本比例', '回购溢价', '事件前2月跌幅'):
        show[c] = (show[c] * 100).round(1).astype(str) + '%'
    show['dv_ttm'] = show['dv_ttm'].round(2).astype(str) + '%'   # dv_ttm 已是百分数, 不再 *100
    print(show.to_string(index=False))
    print(f"\n完整 {len(out)} 家已存 {OUT_CSV}")
    print("\n⚠️ 说明:")
    print("  · [缺失] 回购用途(注销/激励)、资金来源(自有/贷款): repurchase 接口无此字段, 需公告文本, 本工具未做。")
    print("  · 股息: dv_ttm 是当前股息率; '近5年分红稳定'需 dividend 接口, 未纳入。FCF 用 ocf_to_profit 近似。")
    print("  · '现价低于回购上限'=True 即 step4 的'潜伏在回购成本附近/以下'(安全边际更高)。")
    print("  · 这是候选清单/尽调起点, 非验证过的策略; 研报'25.4%超额'是卖方回测数字, 勿当真。")


if __name__ == '__main__':
    main()
