# -*- coding: utf-8 -*-
"""
fetch_repurchase.py — 拉取 2026 年至今的上市公司回购数据 (Tushare repurchase, 600 积分)。

复用 fetch_moneyflow_extra 的 get_pro / call (分页 + 重试), 与现有基建一致。
输出: tushare_cache/_partial/repurchase/repurchase_2026.parquet

字段 (官方): ts_code, ann_date, end_date, proc(进度), exp_date, vol(回购数量,股), amount(回购金额,元),
            high_limit(回购最高价), low_limit(回购最低价)。
注: proc 取值如 预案 / 股东大会通过 / 实施 / 完成 / 停止。同一公司同一事件可能多条(进度更新)。
   无【回购用途/资金来源】字段 —— 那两条研报标准需公告文本, 本结构化数据做不了 (screen 里会标注)。
"""
import os
import datetime as dt

import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

OUT_DIR = './tushare_cache/_partial/repurchase'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pro = F.get_pro()
    today = dt.date.today().strftime('%Y%m%d')
    print(f"拉取回购数据 20260101 ~ {today} ...")
    df = F.call(pro, 'repurchase', start_date='20260101', end_date=today)
    if df.empty:
        print("⚠️ 无 2026 回购数据返回 (检查积分/日期)。")
        return
    for c in ('ann_date', 'end_date', 'exp_date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    df['code'] = df['ts_code'].astype(str).str[:6]
    df = df.sort_values('ann_date').reset_index(drop=True)
    path = os.path.join(OUT_DIR, 'repurchase_2026.parquet')
    df.to_parquet(path, engine='fastparquet', index=False)
    print(f"✅ 已存 {path}")
    print(f"   {len(df)} 行 | {df['ts_code'].nunique()} 家公司 | 进度分布:")
    print(df['proc'].value_counts().to_string())


if __name__ == '__main__':
    main()
