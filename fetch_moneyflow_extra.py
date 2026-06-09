"""
fetch_moneyflow_extra.py — companion to fetch_tushare.py. Pulls the EXTRA endpoints the
4-layer 资金流 design needs, into the SAME tushare_cache/_partial/<endpoint>/ parquet-partials
layout (so it resumes on Ctrl-C and the consumer code finds it the same way as `moneyflow`).

Deliberately NOT a new DB-backed system: your existing partials-resume already gives
incremental safety, so this reuses the proven idioms (env token, paginated call(),
binary-search the empty-history prefix, .empty markers for genuinely-empty days).

Endpoints (ALL fields pulled — never guess column names; confirm them in the output):
  moneyflow_dc       东财个股净流入 (net_amount 等). 历史 ~2023-09 起 -> 早期自动跳过.
  moneyflow_ind_dc   东财板块资金流, content_type='行业' / '概念'  (Layer-2/3, 现 DEFERRED).
  daily_basic        circ_mv 等 (将来市值中性化用).

⚠ moneyflow_dc / moneyflow_ind_dc 需足够 Tushare 积分; 若 call 持续报权限错说明账号未开通。

NOTE: this module also exposes get_pro / call / trading_days / pull_by_date / find_data_start
so run_data_update.py can reuse them without duplicating logic.

RUN (PowerShell, 在 G49_Claude 目录):
    $env:TUSHARE_TOKEN="<your token>"
    python fetch_moneyflow_extra.py
"""
import os
import time
import glob
from datetime import datetime
import pandas as pd

try:
    import tushare as ts
except ImportError:
    raise SystemExit("Tushare not installed:  pip install tushare --upgrade")

# ───────────────────────── config (match fetch_tushare.py) ─────────────────────────
START_DATE = '20220101'      # moneyflow_dc 实际 ~2023-09 起; find_data_start 自动跳过空前缀
END_DATE   = datetime.now().strftime('%Y%m%d')   # ★ 动态取到今天 — 绝不写死, 否则拉不到最新交易日
CACHE_DIR  = 'tushare_cache'
EXCHANGE   = 'SSE'
PAGE_LIMIT = 6000
THROTTLE_S = 0.40
MAX_RETRIES = 5

PLAIN_ENDPOINTS = {
    'moneyflow_dc': {'enabled': True, 'params': {}},
    'daily_basic':  {'enabled': True, 'params': {}},
}
IND_CONTENT_TYPES = ['行业', '概念']     # add '地域' if wanted


def get_pro():
    tok = os.environ.get('TUSHARE_TOKEN')
    if not tok:
        raise SystemExit("Set the TUSHARE_TOKEN env var first (never hard-code the token).")
    return ts.pro_api(tok)


def call(pro, api_name, **params):
    """One logical API call: paginates (offset/limit) and retries with backoff."""
    frames, offset = [], 0
    while True:
        df = None
        for attempt in range(MAX_RETRIES):
            try:
                df = pro.query(api_name, offset=offset, limit=PAGE_LIMIT, **params)
                break
            except Exception as e:
                wait = THROTTLE_S * (2 ** attempt) + 1.0
                print(f"      retry {attempt + 1}/{MAX_RETRIES} {api_name} {params}: {e} -> {wait:.1f}s")
                time.sleep(wait)
        if df is None:
            raise RuntimeError(f"{api_name} {params} failed after {MAX_RETRIES} retries")
        if df.empty:
            break
        frames.append(df)
        if len(df) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(THROTTLE_S)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def trading_days(pro):
    cal = call(pro, 'trade_cal', exchange=EXCHANGE,
               start_date=START_DATE, end_date=END_DATE, is_open='1')
    return sorted(cal['cal_date'].astype(str).tolist())


def _done(out_dir):
    stems = set()
    for ext in ('*.parquet', '*.empty'):
        for f in glob.glob(os.path.join(out_dir, ext)):
            stems.add(os.path.splitext(os.path.basename(f))[0])
    return stems


def find_data_start(pro, api_name, days, extra=None, probe_back=3):
    """Binary-search the FIRST date carrying data (skips an empty-history prefix, e.g.
    moneyflow_dc before ~2023-09). Probes a SETTLED recent day (days[-1-probe_back], not
    today — whose data may not be published yet) to decide if the endpoint has data at all.
    Returns an index into `days`, or None if the endpoint is empty / no-access in range."""
    extra = extra or {}
    if not days:
        return None
    hi = len(days) - 1
    probe = max(0, hi - probe_back)                 # recent but already-settled day
    if call(pro, api_name, trade_date=days[probe], **extra).empty:
        # scan back up to ~20 days for ANY data; if none, treat as no data / no access
        found = None
        for d in days[max(0, probe - 20):probe + 1][::-1]:
            if not call(pro, api_name, trade_date=d, **extra).empty:
                found = days.index(d); break
        if found is None:
            return None
        probe = found
    # binary-search [0, probe] for the first non-empty day
    lo, hi2, first = 0, probe, probe
    while lo <= hi2:
        mid = (lo + hi2) // 2
        if call(pro, api_name, trade_date=days[mid], **extra).empty:
            lo = mid + 1
        else:
            first, hi2 = mid, mid - 1
        time.sleep(THROTTLE_S)
    return first


def pull_by_date(pro, api_name, out_dir, days, extra=None, label=None):
    """Loop trade_date, one parquet partial per day; resume-safe; .empty marks blank days."""
    extra = extra or {}
    label = label or api_name
    os.makedirs(out_dir, exist_ok=True)
    done = _done(out_dir)
    start_i = find_data_start(pro, api_name, days, extra)
    if start_i is None:
        print(f"  [{label}] no data anywhere in range (check account access). skipped.")
        return
    todo = [d for d in days[start_i:] if d not in done]
    print(f"  [{label}] data starts {days[start_i]} | {len(todo)} day(s) to fetch "
          f"({len(done)} already cached)")
    for i, d in enumerate(todo, 1):
        try:
            df = call(pro, api_name, trade_date=d, **extra)
        except Exception as e:
            print(f"    {d} FAILED: {e!r} (will retry on next run)")
            continue
        if df.empty:
            open(os.path.join(out_dir, f"{d}.empty"), 'w').close()   # mark blank day, don't refetch
        else:
            df.to_parquet(os.path.join(out_dir, f"{d}.parquet"), engine='fastparquet')
        if i % 50 == 0 or i == len(todo):
            print(f"    {label}: {i}/{len(todo)}  (last {d})")
        time.sleep(THROTTLE_S)


def main():
    pro = get_pro()
    days = trading_days(pro)
    print(f"📥 fetch_moneyflow_extra | {len(days)} trading days {START_DATE}->{END_DATE}")
    for api_name, cfg in PLAIN_ENDPOINTS.items():
        if not cfg.get('enabled'):
            continue
        pull_by_date(pro, api_name, os.path.join(CACHE_DIR, '_partial', api_name),
                     days, extra=cfg.get('params', {}))
    for ct in IND_CONTENT_TYPES:
        sub = {'行业': 'industry', '概念': 'concept', '地域': 'region'}.get(ct, ct)
        pull_by_date(pro, 'moneyflow_ind_dc',
                     os.path.join(CACHE_DIR, '_partial', 'moneyflow_ind_dc', sub),
                     days, extra={'content_type': ct}, label=f"moneyflow_ind_dc[{ct}]")
    print("✅ done. Inspect columns before wiring the consumer (pulled in full, not guessed).")


if __name__ == '__main__':
    main()
