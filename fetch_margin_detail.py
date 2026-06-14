# -*- coding: utf-8 -*-
"""
fetch_margin_detail.py — 个股融资融券明细 (Tushare margin_detail, doc_id=59) 增量拉取。

为什么是这个接口而不是 margin(doc_id=58):
  · margin    = 按交易所的市场级汇总 -> 只有市场级择时信号 (我们已证伪四次, 不要)
  · margin_detail = 逐股明细 -> 有截面区分度, 才能做选股因子。

输出: 独立 parquet 分片 tushare_cache/_partial/margin_detail/{YYYYMMDD}.parquet
  与 moneyflow/adj_factor 同一套增量纪律: 每个交易日一个文件, 停在未发布日, 幂等, 剔北交所。
  绝不碰 TDX 价格湖 (stock_data_all)。

关键边界:
  · 融资融券只覆盖【两融标的股】(约 1500-3500 只, 非全市场), 其余股票该列为 NaN —
    这是数据本身的覆盖, 不是 bug。观察舱的 check_ic 会按日 dropna, 只用覆盖到的票算 IC。
  · 字段单位: rzye/rqye/rzmre/rzche/rzrqye 是【元】, rqyl/rqchl/rqmcl 是【股】。原样存,
    因子构造时再归一化 (除以流通市值/成交额), 避免在抓取层做任何会引入前视的尺度处理。
  · 发布时滞: 交易所一般 T+0 晚间披露, 遇到第一个空交易日就停, 下次续传。
"""
import os
import glob
import datetime as dt

import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

OUT_DIR_DEFAULT = './tushare_cache/_partial/margin_detail'
# 存这些列即可 (name 会变, 不存; 用代码对齐价格湖)
KEEP = ['ts_code', 'trade_date', 'rzye', 'rqye', 'rzmre', 'rqyl',
        'rzche', 'rqchl', 'rqmcl', 'rzrqye']
_NUM = ['rzye', 'rqye', 'rzmre', 'rqyl', 'rzche', 'rqchl', 'rqmcl', 'rzrqye']
START_FALLBACK = '20220101'        # 空仓时的回填起点 (与价格湖回测区间对齐, 留足训练前摇)


def _out_dir():
    try:
        from config import CONFIG
        return CONFIG.get('margin_detail_path', OUT_DIR_DEFAULT)
    except Exception:
        return OUT_DIR_DEFAULT


def _frontier(out_dir):
    """已落盘分片的最新交易日 (YYYYMMDD), 无则 None。"""
    files = glob.glob(os.path.join(out_dir, '*.parquet'))
    days = [os.path.basename(f)[:-8] for f in files if os.path.basename(f)[:-8].isdigit()]
    return max(days) if days else None


def update_margin_detail(pro=None, out_dir=None):
    pro = pro or F.get_pro()
    out_dir = out_dir or _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    today = dt.datetime.now().strftime('%Y%m%d')
    front = _frontier(out_dir)
    start = (dt.datetime.strptime(front, '%Y%m%d') + dt.timedelta(days=1)).strftime('%Y%m%d') \
            if front else START_FALLBACK
    if front and start > today:
        print(f"  [margin_detail] 已是最新 (frontier {front})")
        return

    cal = F.call(pro, 'trade_cal', exchange='', start_date=start, end_date=today, is_open='1')
    days = sorted(cal['cal_date'].astype(str).tolist()) if not cal.empty else []
    if not days:
        print(f"  [margin_detail] 已是最新 (frontier {front}, 其后无交易日)")
        return
    print(f"  [margin_detail] 待补 {len(days)} 个交易日 ({days[0]}..{days[-1]})"
          + ("  ← 首次全量回填" if not front else ""))

    n_new = 0
    for d in days:                                  # 升序; 第一个空日即停, 保证无中间空洞
        df = F.call(pro, 'margin_detail', trade_date=d)
        if df.empty:
            print(f"  [margin_detail] {d} 尚未入库 — 到此为止, 下次运行续传")
            break
        df = df[df['ts_code'].str[-3:].isin(['.SZ', '.SH'])]      # 剔北交所, 与 universe 一致
        for c in KEEP:
            if c not in df.columns:
                df[c] = pd.NA
        for c in _NUM:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df[KEEP].to_parquet(os.path.join(out_dir, f"{d}.parquet"),
                            engine='fastparquet', index=False)
        n_new += 1
    print(f"  [margin_detail] ✅ 新增 {n_new} 个分片 (累计 {len(glob.glob(os.path.join(out_dir,'*.parquet')))})")


if __name__ == '__main__':
    update_margin_detail()
