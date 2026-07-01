# -*- coding: utf-8 -*-
"""
fetch_dividend.py — Tushare dividend(分红送股) 增量拉取, 按 ann_date 循环 (dividend 用ann_date而非trade_date)。

用途: 筛选器的"连续分红年数"(财务质量/避雷: 能连续派真金白银现金分红 -> 利润大概率真实, 非纸面/造假)。
存 _partial/dividend/{ann_date}.parquet; 断点续传(跳过已拉); 复用 fetch_moneyflow_extra 的 get_pro/call/trading_days。
可被 run_data_update 直接调用 (复用其 pro)。首次拉 2022->今(~1000个公告日, 几分钟); 之后只补新日。
dividend 接口 ann_date 单参可返回全市场当日公告 (doc: ts_code 和 ann_date 至少输入一个)。
"""
import os
import time
import glob

import pandas as pd

import fetch_moneyflow_extra as F

START_DATE = '20220101'
RECENT_GUARD = 5   # 最近N个公告日即使空也不打.empty, 留待重试
OUT = os.path.join(F.CACHE_DIR, '_partial', 'dividend')


def update_dividend(pro=None):
    pro = pro or F.get_pro()
    days = [d for d in F.trading_days(pro) if d >= START_DATE]
    if not days:
        print("  [dividend] 无交易日")
        return
    os.makedirs(OUT, exist_ok=True)
    done = set(os.path.splitext(os.path.basename(f))[0]
               for f in glob.glob(os.path.join(OUT, '*.parquet')) + glob.glob(os.path.join(OUT, '*.empty')))
    todo = [d for d in days if d not in done]
    recent = set(days[-RECENT_GUARD:])
    print(f"  [dividend] 按ann_date增量: {len(todo)} 个公告日待拉 ({len(done)} 已缓存)")
    wrote = 0
    for i, d in enumerate(todo, 1):
        try:
            df = F.call(pro, 'dividend', ann_date=d)
        except Exception as e:
            print(f"    {d} FAILED: {e!r} (下次重试)")
            continue
        if df is None or df.empty:
            if d not in recent:                                  # 近期空不标记, 留待重试
                open(os.path.join(OUT, f"{d}.empty"), 'w').close()
        else:
            df.to_parquet(os.path.join(OUT, f"{d}.parquet"), engine='fastparquet')
            wrote += 1
        if i % 50 == 0 or i == len(todo):
            print(f"    dividend: {i}/{len(todo)} (last {d}, 写入 {wrote})")
        time.sleep(F.THROTTLE_S)
    print(f"  [dividend] ✅ 写入 {wrote} 个公告日分片 (累计 {len(glob.glob(os.path.join(OUT,'*.parquet')))})")


def main():
    update_dividend()


if __name__ == '__main__':
    main()
