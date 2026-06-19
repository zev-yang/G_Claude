# -*- coding: utf-8 -*-
"""
fetch_financials_tushare.py — 拉取三张财报表 (资产负债表/利润表/现金流量表) 进
tushare_cache，供 潜伏 的基本面因子使用。

与 fetch_tushare.py 唯一的结构差异：财报非日频，按【报告期 period=end_date】循环，用
**VIP endpoint** (balancesheet_vip/income_vip/cashflow_vip) 一次取该报告期全市场所有股票
(需 ~5000 积分；普通 endpoint 强制要 ts_code 无法按报告期取)。约 (年数×4×3) 次调用 ——
比你的日频拉取轻得多 (~1 分钟)。只保留 report_type==1 (合并报表/累计口径)，避免 TTM 去累计出错。
token / 分页 / 重试 / 限频 / 幂等分片 / .BJ 排除 全部复用 fetch_tushare 的同一套。

★ ALL fields 全量拉取——不依赖我猜的列名。拉完把 _combine 打印的 cols 回给 Claude，
  锁定 FIELD_MAP 后再把因子模块适配进 潜伏 的 (date, code) 透明 composite。

RUN (PowerShell):
    $env:TUSHARE_TOKEN="<token>"
    python fetch_financials_tushare.py
或在 run_data_update.py 里：  import fetch_financials_tushare as FF;  FF.update_financials(pro)
"""
import os
import glob
import time
from datetime import date

import pandas as pd

# 复用你已验证的 helpers，确保和现有管线同一套限频/重试/幂等纪律
from fetch_tushare import get_pro, call, _done, CACHE_DIR, THROTTLE_S

# ───────────────────────── config ─────────────────────────
# 财报需要回测窗口【之前】~2 年历史：TTM = 上年年报 + 本期累计 − 上年同期累计；YoY 要上年同期。
# START_YEAR 设成 (你 lake 起始年 − 2) 左右。END_YEAR 自动到当年。
START_YEAR = 2018
END_YEAR   = date.today().year                        # 自动含最新年, 无需手改
# cache 名 -> Tushare VIP endpoint。VIP 支持 period= 一次取全市场 (需 ~5000 积分;
# 普通 balancesheet/income/cashflow 强制要 ts_code, 无法按报告期取 —— 这就是之前报错的原因)。
STATEMENTS = {
    'balancesheet': 'balancesheet_vip',
    'income':       'income_vip',
    'cashflow':     'cashflow_vip',
}
# 一个报告期要 ~1 个月才陆续披露完。每次重拉最近 N 期, 纳入晚披露/重述公司
# (镜像 run_data_update.py 日频的 _clear_recent_empty)。旧报告期已稳定, 永久缓存、不重拉。
REFETCH_RECENT = 2
# ───────────────────────────────────────────────────────────


def report_periods(y0, y1):
    """季度末 end_date 列表 (如 20180331, ...)，并剔除尚未结束的报告期。"""
    today = date.today().strftime('%Y%m%d')
    ps = [f"{y}{mmdd}" for y in range(y0, y1 + 1) for mmdd in ('0331', '0630', '0930', '1231')]
    return [p for p in ps if p <= today]              # 季度未结束 -> 还没有任何数据, 不拉


def _clear_recent(out_dir, periods, n):
    """删掉最近 n 个报告期的缓存分片, 强制重拉以纳入晚披露/重述公司。"""
    for p in periods[-n:]:
        for ext in ('parquet', 'empty'):
            f = os.path.join(out_dir, f"{p}.{ext}")
            if os.path.exists(f):
                os.remove(f)


def _combine_stmt(name, out_dir):
    """concat 全部分片 -> tushare_cache/<name>.parquet。
    廉价 dedup: 同 (ts_code,end_date,f_ann_date) 留最新; 真·vintage(不同 f_ann_date) 留给
    因子模块 _first_disclosure。不做全列 drop_duplicates (152 列 × 数十万行又慢又无用)。"""
    files = sorted(glob.glob(os.path.join(out_dir, '*.parquet')))
    print(f"      [{name}] 读取 {len(files)} 个分片 ...")
    parts = [pd.read_parquet(f) for f in files]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print(f"   {name}: 无数据。")
        return
    full = pd.concat(parts, ignore_index=True)
    keys = [k for k in ('ts_code', 'end_date', 'f_ann_date') if k in full.columns]
    if keys:
        full = full.drop_duplicates(keys, keep='last')     # 廉价 3 列去重, 非全列 152 列
    if 'ts_code' in full.columns:                          # 6 位无后缀 code, 对齐 lake 的 (date, code)
        full['code'] = full['ts_code'].astype(str).str[:6]
    dest = os.path.join(CACHE_DIR, f"{name}.parquet")
    print(f"      [{name}] 写出 {len(full):,} 行 × {full.shape[1]} 列 -> {dest} ...")
    full.to_parquet(dest)
    nper = full['end_date'].nunique() if 'end_date' in full.columns else len(parts)
    print(f"   -> {dest}  ({len(full):,} 行, {nper} 个报告期)")
    print(f"      cols: {list(full.columns)}")         # ← 把这行回给 Claude 锁 FIELD_MAP


def fetch_statement(pro, name, api_name, periods):
    out_dir = os.path.join(CACHE_DIR, '_partial', name)
    os.makedirs(out_dir, exist_ok=True)
    if REFETCH_RECENT:                                 # 重拉最近 N 期, 纳入晚披露公司
        _clear_recent(out_dir, periods, REFETCH_RECENT)
    todo = [p for p in periods if p not in _done(out_dir)]
    print(f"\n[{name}] {len(todo)} 个报告期待拉 ({len(periods) - len(todo)} 已缓存)")

    consec_fail = 0
    for i, p in enumerate(todo, 1):
        try:
            df = call(pro, api_name, period=p)         # VIP: 全市场 × 该报告期, ALL fields
            consec_fail = 0
        except Exception as e:
            consec_fail += 1
            print(f"   {name} {p}: 失败 ({e})")
            if consec_fail >= 3:
                print(f"!! {name}: 连续 3 次失败 — 中止。VIP 需 ~5000 积分; 若积分够仍失败把报错回给 Claude。")
                return
            continue
        if not df.empty:
            # 只留 report_type==1 (合并报表/累计口径)。单季(2)/调整(4)等混入会让 TTM 去累计出错。
            if 'report_type' in df.columns:
                df = df[df['report_type'].astype(str) == '1']
            # 排除 .BJ (北交所)，对齐你 lake 的沪深口径
            if 'ts_code' in df.columns:
                df = df[~df['ts_code'].astype(str).str.endswith('.BJ')]
        if df.empty:
            open(os.path.join(out_dir, f"{p}.empty"), 'w').close()   # 标记，避免重拉
        else:
            df.to_parquet(os.path.join(out_dir, f"{p}.parquet"))
        if i % 8 == 0 or i == len(todo):
            print(f"   {name}: {i}/{len(todo)}  last={p}  rows={len(df)}")
        time.sleep(THROTTLE_S)
    # 注意: 这里【不】合并。合并放到 update_financials 的阶段2, 避免某张表合并慢挡住其它表下载。


def update_financials(pro=None):
    """供 run_data_update.py 调用 (镜像 fetch_adj_factor.update_adj_factor(pro) 的形态)。
    先把三张表的分片全下完, 再统一合并 —— 某张表合并慢/失败也不会挡住其它表的【下载】。"""
    pro = pro or get_pro()
    periods = report_periods(START_YEAR, END_YEAR)
    print(f"{len(periods)} 个报告期  {START_YEAR}Q1 -> {END_YEAR}Q4 "
          f"| 三表(VIP)约 {len(periods) * len(STATEMENTS)} 次调用")
    # 阶段 1: 三张表分片全部下载 (真正的下载, 不被任何合并阻塞)
    for name, api_name in STATEMENTS.items():
        try:
            fetch_statement(pro, name, api_name, periods)
        except Exception as e:
            print(f"!! {name} 拉取中止: {e!r} (继续其余)")
    # 阶段 2: 统一合并 (一张慢/失败不影响其它; 分片已在盘上, 重跑会续上)
    print("\n--- 合并三表 ---")
    for name in STATEMENTS:
        try:
            _combine_stmt(name, os.path.join(CACHE_DIR, '_partial', name))
        except Exception as e:
            print(f"!! {name} 合并失败: {e!r} (分片已在 _partial/{name}/)")
    # 最终汇总: 明确每张表的 combined 文件是否生成 (一眼看出三表都齐没齐)
    print("\n=== 三表汇总 ===")
    for name in STATEMENTS:
        dest = os.path.join(CACHE_DIR, f"{name}.parquet")
        if os.path.exists(dest):
            try:
                n = len(pd.read_parquet(dest, columns=['ts_code']))
                print(f"  ✅ {name:13s} {dest} ({n:,} 行)")
            except Exception:
                print(f"  ✅ {name:13s} {dest} (已生成)")
        else:
            print(f"  ❌ {name:13s} 未生成 — 看上面该表的报错")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    update_financials(get_pro())
    print("\nDone. 把每张表 _combine 打印的 cols 回给 Claude —— 锁定 FIELD_MAP 后再适配因子模块到 潜伏。")


if __name__ == '__main__':
    main()
