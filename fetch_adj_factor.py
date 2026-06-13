"""
fetch_adj_factor.py — 复权因子增量拉取 (Tushare `adj_factor`, doc_id=28)。

落盘: tushare_cache/_partial/adj_factor/{YYYYMMDD}.parquet (与 moneyflow 同模式),
字段 ts_code / trade_date / adj_factor。复权因子的历史值【永不改写】(自上市起累乘
定义), 所以按日分片天然 append-only, 增量友好 — 这也是选 hfq(后复权)而非 qfq 的原因。

范围: ADJ_START(写死 2022-01-01, 覆盖 max_history_days=1000 个交易日 + 指标预热)起
到今天; 已存在的分片跳过; 遇到第一个未发布日期即停(不留中间空洞)。
只保留 .SZ/.SH, 与湖 universe 一致。
"""
import glob
import os
import datetime as dt

import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

OUT_DIR   = './tushare_cache/_partial/adj_factor'
ADJ_START = '20220101'   # 覆盖回测所需 1000 交易日 + 预热; 不要调小


def update_adj_factor(pro=None):
    pro = pro or F.get_pro()
    os.makedirs(OUT_DIR, exist_ok=True)
    have = {os.path.basename(p)[:-8] for p in glob.glob(os.path.join(OUT_DIR, '*.parquet'))}
    today = dt.datetime.now().strftime('%Y%m%d')

    cal = F.call(pro, 'trade_cal', exchange='', start_date=ADJ_START,
                 end_date=today, is_open='1')
    days = sorted(set(cal['cal_date'].astype(str)) - have)
    if not days:
        print(f"  [adj_factor] 已是最新 ({len(have)} 个分片)")
        return
    print(f"  [adj_factor] 待补 {len(days)} 个交易日 ({days[0]}..{days[-1]})"
          + ("  ← 首次全量回填, 约 3-5 分钟" if len(days) > 200 else ""))

    pulled = 0
    for d in days:                                   # 升序, 第一个空日即停
        df = F.call(pro, 'adj_factor', trade_date=d)
        if df.empty:
            print(f"  [adj_factor] {d} 尚未入库 — 到此为止, 下次运行续传")
            break
        df = df[df['ts_code'].str[-3:].isin(['.SZ', '.SH'])]
        df[['ts_code', 'trade_date', 'adj_factor']].to_parquet(
            os.path.join(OUT_DIR, f"{d}.parquet"), engine='fastparquet', index=False)
        pulled += 1
    print(f"  [adj_factor] ✅ 新增 {pulled} 个分片 (累计 {len(have)+pulled})")


if __name__ == '__main__':
    update_adj_factor()
