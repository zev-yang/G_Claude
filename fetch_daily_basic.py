# -*- coding: utf-8 -*-
"""
fetch_daily_basic.py — 每日指标 (Tushare daily_basic, doc_id=32) 回填 + 增量。

背景: 你已有 2022-01-04 至今的 daily_basic 分片 (1074 个), 但潜伏模式的估值分位想要
  更长历史 (5 年滚动 PE/PB)。本脚本【向前回填】到接口起点(~2009), 并【向后增量】到今天,
  与已有分片无缝衔接 (按日分片, 已存在的日期默认跳过)。

为何按日循环 (trade_date=) 而非逐股: daily_basic 单次 trade_date 返回【全市场~5000行】,
  按交易日循环 ~4000 天 ≈ 4000 次调用; 逐股要 5500×多年 = 几十万次。按日快几十倍。

落盘: tushare_cache/_partial/daily_basic/{YYYYMMDD}.parquet — 与现有分片同目录同命名。
  字段: ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,
        pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,
        total_mv,circ_mv  (潜伏模式估值层主要用 pe_ttm/pb/circ_mv; 多存无害)。

纪律: 主动限速 450/min (你的上限 500); 剔北交所(.BJ); 幂等(已存在的日期跳过, 中断可续);
  接口对早期日期返回空 -> 视为"该日无数据", 跳过, 不报错。
"""
import os
import glob
import time
import datetime as dt

import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

OUT_DIR_DEFAULT = './tushare_cache/_partial/daily_basic'
FIELDS = ('ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,'
          'pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,'
          'free_share,total_mv,circ_mv')
KEEP = FIELDS.split(',')
EARLIEST = '20090101'          # daily_basic 接口实际起点约在此; 更早返回空会自动跳过


class _RateLimiter:
    def __init__(self, max_per_min):
        self.interval = 60.0 / max_per_min
        self._last = 0.0
    def wait(self):
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            time.sleep(gap)
        self._last = time.monotonic()


_LIMITER = _RateLimiter(max_per_min=450)


def _out_dir():
    try:
        from config import CONFIG
        p = CONFIG.get('fundamentals_path', OUT_DIR_DEFAULT)   # V25 的估值分片路径
        return p
    except Exception:
        return OUT_DIR_DEFAULT


def _existing_days(out_dir):
    return {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(out_dir, '*.parquet'))
            if os.path.basename(f)[:-8].isdigit()}


def update_daily_basic(pro=None, out_dir=None, start=EARLIEST, end=None):
    pro = pro or F.get_pro()
    out_dir = out_dir or _out_dir()
    os.makedirs(out_dir, exist_ok=True)
    end = end or dt.datetime.now().strftime('%Y%m%d')

    # 全部交易日 (start..end)
    cal = F.call(pro, 'trade_cal', exchange='', start_date=start, end_date=end, is_open='1')
    if cal.empty:
        print("  [daily_basic] trade_cal 返回空, 无法继续")
        return
    all_days = sorted(cal['cal_date'].astype(str).tolist())

    have = _existing_days(out_dir)
    todo = [d for d in all_days if d not in have]          # 幂等: 跳过已落盘日期
    print(f"  [daily_basic] 区间 {start}..{end}: 交易日 {len(all_days)}, 已有 {len(have)}, "
          f"待补 {len(todo)} 天")
    if not todo:
        print("  [daily_basic] 已完整, 无需补")
        return

    _t0 = time.monotonic()
    n_written, n_empty = 0, 0
    for i, d in enumerate(todo, 1):
        _LIMITER.wait()
        try:
            df = F.call(pro, 'daily_basic', trade_date=d, fields=FIELDS)
        except Exception:
            continue
        if df is None or df.empty:                          # 早期无数据的日期 -> 跳过
            n_empty += 1
            continue
        df = df[df['ts_code'].str[-3:].isin(['.SZ', '.SH'])]   # 剔北交所
        for c in KEEP:
            if c not in df.columns:
                df[c] = pd.NA
        df[KEEP].to_parquet(os.path.join(out_dir, f"{d}.parquet"),
                            engine='fastparquet', index=False)
        n_written += 1
        if i % 500 == 0:
            _el = (time.monotonic() - _t0) / 60
            print(f"  [daily_basic] 进度 {i}/{len(todo)} | 写入 {n_written} | "
                  f"空日 {n_empty} | 已用 {_el:.1f} 分钟")
    print(f"  [daily_basic] ✅ 写入 {n_written} 天, 跳过空日 {n_empty} "
          f"(累计 {len(_existing_days(out_dir))} 个分片)")


if __name__ == '__main__':
    update_daily_basic()
