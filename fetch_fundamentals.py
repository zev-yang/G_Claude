# -*- coding: utf-8 -*-
"""
fetch_fundamentals.py — 季度财务指标 (Tushare fina_indicator) 增量拉取。【潜伏模式专用地基】

★ 这是潜伏模式的命根子, 唯一目的: 把【未来函数】堵死。
  财报报告期(end_date) 与实际公告日(ann_date) 之间有 1-4 个月时滞:
    2023 年报 end_date=20231231, 但 ann_date=20240403 (4月3日才公告)。
  若按 end_date 对齐 -> 用 4 月才知道的 ROE 做去年12月的决策 = 未来函数。
  潜伏模式持仓 3-12 个月, 这种偏差会放大成巨大的假收益。所以本 fetcher 存的是
  【(ann_date, ts_code) -> 财务指标】, 下游按【交易日 >= ann_date】对齐, 任何一天只能
  看到那天之前已公告的财报。这是潜伏模式回测可信的前提。

去重: 同一 (ts_code, end_date) 可能多次披露(预告/快报/正式/修正), 保留 ann_date 最早的
  那条(即"市场首次知道"的时点; 修正版会晚于首次, 用首次更保守、更接近实盘可得信息)。

共享原则: 与 V25 共用日线/复权/资金流/融资融券; 本脚本只新增 fina_indicator 这一类。
  输出: tushare_cache/_partial/fundamentals/{ann_date_YYYYMM}.parquet (按公告月分片)
"""
import os
import glob
import time
import datetime as dt

import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

OUT_DIR_DEFAULT = './tushare_cache/_partial/fundamentals'


class _RateLimiter:
    """主动限速 (令牌桶简化版): 保证每 60 秒内调用不超过 max_per_min 次。

    从源头避免撞 Tushare 的 500次/分钟 上限 —— 比"撞墙后重试退避"高效得多。
    留 10% 安全边际 (默认按 450/min 节流, 而非贴着 500)。
    """
    def __init__(self, max_per_min):
        self.interval = 60.0 / max_per_min      # 两次请求的最小间隔
        self._last = 0.0

    def wait(self):
        now = time.monotonic()
        gap = self.interval - (now - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


# 默认按 450/min 节流 (用户上限 500, 留安全边际)。如需更快/更慢, 改这里。
_LIMITER = _RateLimiter(max_per_min=450)
# 质量过滤 + 深度价值层要用的字段 (来自 fina_indicator)
FIELDS = ('ts_code,ann_date,end_date,roe,roa,debt_to_assets,'
          'ocfps,netprofit_yoy,or_yoy,grossprofit_margin,netprofit_margin,'
          'q_profit_yoy,profit_to_op,assets_turn')
KEEP = FIELDS.split(',')
_NUM = [c for c in KEEP if c not in ('ts_code', 'ann_date', 'end_date')]
START_FALLBACK = '20180101'        # 财务要长历史做分位/增速 (潜伏模式看 5 年估值分位)


def _out_dir():
    try:
        from config import CONFIG
        return CONFIG.get('fundamentals_q_path', OUT_DIR_DEFAULT)
    except Exception:
        return OUT_DIR_DEFAULT


def _frontier_ann(out_dir):
    """已落盘分片里最大的公告日 (YYYYMMDD)，无则 None。按公告日增量。"""
    mx = None
    for f in glob.glob(os.path.join(out_dir, '*.parquet')):
        try:
            d = pd.read_parquet(f, columns=['ann_date'], engine='fastparquet')
            m = str(d['ann_date'].max())
            if m and m != 'nan' and (mx is None or m > mx):
                mx = m
        except Exception:
            continue
    return mx


def _stock_list(pro):
    """全部 A 股代码 (含已退市, 避免幸存者偏差); 剔北交所与 V25 一致。"""
    frames = []
    for status in ('L', 'D', 'P'):       # 上市/退市/暂停
        try:
            sb = F.call(pro, 'stock_basic', exchange='', list_status=status, fields='ts_code')
            if not sb.empty:
                frames.append(sb)
        except Exception:
            continue
    if not frames:
        return []
    codes = pd.concat(frames, ignore_index=True)['ts_code'].astype(str)
    return sorted(codes[codes.str[-3:].isin(['.SZ', '.SH'])].unique())


def update_fundamentals(pro=None, out_dir=None):
    pro = pro or F.get_pro()
    out_dir = out_dir or _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    today = dt.datetime.now().strftime('%Y%m%d')
    front = _frontier_ann(out_dir)
    start = front if front else START_FALLBACK     # 用公告日做增量起点(含重叠, 后面去重)
    print(f"  [fundamentals] 按公告日拉取 fina_indicator: {start}..{today}"
          + ("  ← 首次全量回填" if not front else f"  (增量, frontier={front})"))

    codes = _stock_list(pro)
    if not codes:
        print("  [fundamentals] SKIP: 取不到股票列表 (检查 stock_basic 权限)")
        return
    print(f"  [fundamentals] {len(codes)} 只股票 (含退市, 防幸存者偏差)")

    frames, n_ok = [], 0
    _t0 = time.monotonic()
    for i, code in enumerate(codes, 1):
        _LIMITER.wait()                          # 主动限速: 从源头不撞 500/min 上限
        try:
            df = F.call(pro, 'fina_indicator', ts_code=code,
                        start_date=start, end_date=today, fields=FIELDS)
        except Exception:
            continue
        if df is None or df.empty:
            continue
        frames.append(df)
        n_ok += 1
        if i % 500 == 0:                         # 进度 (5530 只约需 ~12-13 分钟 @450/min)
            _el = (time.monotonic() - _t0) / 60
            print(f"  [fundamentals] 进度 {i}/{len(codes)} | 有数据 {n_ok} | 已用 {_el:.1f} 分钟")
    if not frames:
        print("  [fundamentals] 本次无新数据")
        return

    allf = pd.concat(frames, ignore_index=True)
    allf = allf.dropna(subset=['ann_date', 'end_date'])
    for c in _NUM:
        if c in allf.columns:
            allf[c] = pd.to_numeric(allf[c], errors='coerce')
    # 去重: 同 (ts_code,end_date) 多次披露 -> 保留 ann_date 最早 (市场首次知道的时点, 最保守)
    allf = (allf.sort_values(['ts_code', 'end_date', 'ann_date'])
                 .drop_duplicates(subset=['ts_code', 'end_date'], keep='first'))

    # 按公告月分片落盘 (与已落盘的合并去重, 幂等)
    allf['_ym'] = allf['ann_date'].astype(str).str[:6]
    written = 0
    for ym, grp in allf.groupby('_ym'):
        path = os.path.join(out_dir, f"{ym}.parquet")
        grp = grp.drop(columns='_ym')
        if os.path.exists(path):
            old = pd.read_parquet(path, engine='fastparquet')
            grp = (pd.concat([old, grp], ignore_index=True)
                     .sort_values(['ts_code', 'end_date', 'ann_date'])
                     .drop_duplicates(subset=['ts_code', 'end_date'], keep='first'))
        grp[KEEP].to_parquet(path, engine='fastparquet', index=False)
        written += 1
    print(f"  [fundamentals] ✅ {n_ok} 只有数据, 写入 {written} 个公告月分片 "
          f"(累计 {len(glob.glob(os.path.join(out_dir,'*.parquet')))})")


if __name__ == '__main__':
    update_fundamentals()
