"""
run_data_update.py — ONE-COMMAND updater for the whole 资金流 data family.

Pulls the increment for every moneyflow endpoint in a single run (so you don't run the
fetch scripts one by one), into the same tushare_cache/_partial/<endpoint>/ layout that
moneyflow_factors.py reads. Reuses the proven helpers in fetch_moneyflow_extra.py.

Works for BOTH the initial full pull AND daily increments:
  • first run on an empty cache  -> find_data_start binary-searches each endpoint's true
    history start (raw moneyflow ~2022, moneyflow_dc ~2023-09) and pulls everything;
  • later runs                   -> _done() skips cached days, so only NEW days are fetched.

Safe to run ANY time of day / schedule daily — no "is the market closed?" logic needed:
if a recent day's data isn't published yet the API returns empty, and we deliberately do
NOT write a .empty marker for the last few days, so it is retried next run until it lands.

Endpoints updated each run:
  moneyflow              raw 个股分层资金流  (moneyflow_factors.py 直接读这个)
  moneyflow_dc           东财个股净流入      (Layer-1, 2023-09 起)
  moneyflow_ind_dc[行业] 板块资金流          (Layer-2)
  moneyflow_ind_dc[概念] 板块资金流          (Layer-3)
  daily_basic            circ_mv 等          (将来市值中性化)
(cyq_perf / margin_detail / top_inst 等非资金流接口仍由 fetch_tushare.py 负责。)

RUN (PowerShell, 在 G49_Claude 目录):
    $env:TUSHARE_TOKEN="<your token>"
    python run_data_update.py
Windows 定时: 任务计划程序每天 19:00 跑一次 (A 股盘后数据通常傍晚可得)。
"""
import os
import fetch_moneyflow_extra as F   # reuse get_pro / call / trading_days / pull_by_date / CACHE_DIR

RECENT_GUARD = 3   # 最近 N 个交易日即使返回空也不打 .empty 标记, 留待下次重试

# (api_name, subfolder under _partial, extra params, label)
JOBS = [
    ('moneyflow',        'moneyflow',                 {},                       'moneyflow(raw)'),
    ('moneyflow_dc',     'moneyflow_dc',              {},                       'moneyflow_dc'),
    ('moneyflow_ind_dc', 'moneyflow_ind_dc/industry', {'content_type': '行业'}, 'moneyflow_ind_dc[行业]'),
    ('moneyflow_ind_dc', 'moneyflow_ind_dc/concept',  {'content_type': '概念'}, 'moneyflow_ind_dc[概念]'),
    ('daily_basic',      'daily_basic',               {},                       'daily_basic'),
]


def _clear_recent_empty(out_dir, recent_dates):
    """Remove .empty markers for the most recent trading days so an incomplete 'today'
    (data not yet published) is retried next run instead of being skipped forever."""
    n = 0
    for d in recent_dates:
        p = os.path.join(out_dir, f"{d}.empty")
        if os.path.exists(p):
            os.remove(p); n += 1
    return n


def daily_update():
    pro = F.get_pro()
    days = F.trading_days(pro)                 # uses fetch_moneyflow_extra START_DATE..END_DATE
    if not days:
        print("no trading days in range — check START_DATE / END_DATE in fetch_moneyflow_extra.py")
        return
    recent = days[-RECENT_GUARD:]
    print(f"🔄 run_data_update | {len(days)} trading days {days[0]}->{days[-1]} | {len(JOBS)} endpoints")
    for api_name, sub, extra, label in JOBS:
        out_dir = os.path.join(F.CACHE_DIR, '_partial', sub)
        try:
            F.pull_by_date(pro, api_name, out_dir, days, extra=extra, label=label)
        except Exception as e:
            print(f"  [{label}] update FAILED: {e!r} (other endpoints continue; rerun later)")
            continue
        cleared = _clear_recent_empty(out_dir, recent)
        if cleared:
            print(f"  [{label}] cleared {cleared} recent .empty marker(s) -> retry next run")

    # ── A股日线增量 (Tushare daily, doc_id=27) -> 直接续写 TDX 数据湖 stock_data_all/*.csv ──
    # 失败不影响上面的资金流更新 (独立 try); 空湖/未到发布时间会自动 SKIP/续传。
    try:
        import fetch_daily_tushare
        fetch_daily_tushare.update_daily_lake(pro)
    except Exception as e:
        print(f"  [daily->lake] update FAILED: {e!r} (资金流已更新; 修复后重跑即可)")

    # ── 复权因子增量 (hfq 迁移的数据底座) ──
    try:
        import fetch_adj_factor
        fetch_adj_factor.update_adj_factor(pro)
    except Exception as e:
        print(f"  [adj_factor] update FAILED: {e!r} (其余更新不受影响)")

    # ── 个股融资融券明细 (Tushare margin_detail, doc_id=59) -> 独立 parquet 分片 ──
    # 真·新信息 (杠杆资金仓位); 失败不影响上面的更新 (独立 try)。observe-only 候选数据源。
    try:
        import fetch_margin_detail
        fetch_margin_detail.update_margin_detail(pro)
    except Exception as e:
        print(f"  [margin_detail] update FAILED: {e!r} (其余已更新; 修复后重跑即可)")

    print("✅ daily_update done. Increments pulled into tushare_cache/_partial/.")


if __name__ == '__main__':
    daily_update()
