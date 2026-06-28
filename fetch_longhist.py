# -*- coding: utf-8 -*-
"""
fetch_longhist.py — value 长样本/生产组合的数据底座: close + adj_factor + pe/pb/circ_mv 回到 ~2010。

独立目录 tushare_cache/_longhist/{daily,adj_factor,daily_basic}, 不碰你现有湖与 _partial。
复用 fetch_moneyflow_extra 的 get_pro / call / pull_by_date (断点续传, 跳过已拉)。

两种用法:
  · 单独跑:        python fetch_longhist.py
  · 并入日更:      run_data_update.py 里  import fetch_longhist; fetch_longhist.update_longhist(pro)
首次跑回填 2010->今 (~25-40 分钟); 之后每次只补新交易日。
"""
import os
import datetime as dt

import fetch_moneyflow_extra as F   # 复用 get_pro / call / pull_by_date

START_DATE = '20100101'
LH_DIR = 'tushare_cache/_longhist'
RECENT_GUARD = 3                    # 最近 N 个交易日即使空也不锁定, 留待下次重试 (与 run_data_update 一致)

ENDPOINTS = [
    ('daily',       'ts_code,trade_date,close'),
    ('adj_factor',  'ts_code,trade_date,adj_factor'),
    ('daily_basic', 'ts_code,trade_date,pe_ttm,pb,circ_mv'),
]


def _clear_recent_empty(out_dir, recent):
    n = 0
    for d in recent:
        p = os.path.join(out_dir, f"{d}.empty")
        if os.path.exists(p):
            os.remove(p); n += 1
    return n


def update_longhist(pro):
    """长历史(2010起)增量, 拉进 _longhist/。断点续传; 可被 run_data_update 直接调用 (复用其 pro)。"""
    today = dt.date.today().strftime('%Y%m%d')
    cal = F.call(pro, 'trade_cal', start_date=START_DATE, end_date=today, is_open='1')
    days = sorted(str(d) for d in cal['cal_date'].tolist())
    if not days:
        print("  [longhist] 无交易日 (检查 token/日期)"); return
    recent = days[-RECENT_GUARD:]
    print(f"  [longhist] {len(days)} 交易日 {days[0]}->{days[-1]} | 3 endpoint, 断点续传")
    for api, fields in ENDPOINTS:
        out_dir = os.path.join(LH_DIR, api)
        F.pull_by_date(pro, api, out_dir, days, extra={'fields': fields}, label='longhist/' + api)
        c = _clear_recent_empty(out_dir, recent)
        if c:
            print(f"  [longhist/{api}] cleared {c} recent .empty -> retry next run")


def main():
    update_longhist(F.get_pro())
    print("✅ 长历史更新完成。下一步可跑 eval_value_* / build_value_portfolio.py")


if __name__ == '__main__':
    main()
