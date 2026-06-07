"""
fetch_tushare.py — pull 主力 / 筹码 / 融资 data from Tushare Pro for the factor pipeline.

Curated for orthogonal alpha (not raw point-spend). Pulls, by default:
  moneyflow_dc  — 大单/超大单净流入 (东财 order flow)     [primary, dense, daily]
  cyq_perf      — 每日筹码分布 stats / 集中度             [primary, dense, daily]
  margin_detail — 融资余额/买入/偿还 + 融券               [secondary, margin-eligible only]
  top_inst      — 龙虎榜机构席位净买额                    [optional, event-sparse]

Disabled by default (flip enabled=True to override — see why in the README notes):
  hk_hold       — 北向个股持股: DEAD since ~Aug 2024. Historical only; no live signal.
  cyq_chips     — full raw chip distribution; heavy. cyq_perf's summary is usually enough.
  top_list      — 龙虎榜个股 (retail-heavy): sparser + noisier than top_inst.

RUN (PowerShell):
    $env:TUSHARE_TOKEN="<your token>"          # session only; or: setx TUSHARE_TOKEN "<token>"
    python fetch_tushare.py

Outputs one combined parquet per endpoint at ./tushare_cache/<endpoint>.parquet.
Per-date partials are cached under ./tushare_cache/_partial/, so an interrupted run
RESUMES where it stopped — safe to Ctrl-C and rerun. This fetch is network-bound (not
CPU/RAM), so the engineering-speed issue from before doesn't apply; expect ~30–60 min
for a multi-year pull, less on reruns.

NOTE: the token is read ONLY from the env var and never stored. Endpoint names are from
your verified list; all FIELDS are pulled in full so we don't depend on guessed column
names — confirm the exact 大单/超大单 net-inflow column names in the output before wiring.
"""
import os
import time
import glob
import pandas as pd

try:
    import tushare as ts
except ImportError:
    raise SystemExit("Tushare not installed:  pip install tushare --upgrade")

# ───────────────────────── config ─────────────────────────
START_DATE = '20220101'     # <- set to (backtest_start − ~300 trading days) for train history
END_DATE   = '20251231'     # <- set to your data lake's last date
CACHE_DIR  = 'tushare_cache'
EXCHANGE   = 'SSE'          # trade-calendar source; SSE == all A-share trading days
PAGE_LIMIT = 6000          # per-call row cap; one A-share day (~5.4k) fits in one page
THROTTLE_S = 0.40          # base sleep between calls (~150/min). Raise if you see many retries.
MAX_RETRIES = 5

# api_name -> options. loop_by='trade_date' = one call per day returns ALL stocks that day.
ENDPOINTS = {
    'moneyflow':     {'enabled': True},   # 大单/超大单 gross buy+sell, 2010+ (full coverage)
    'cyq_perf':      {'enabled': True},
    'margin_detail': {'enabled': True},
    'top_inst':      {'enabled': True},
    # ── disabled by default ──
    'moneyflow_dc':  {'enabled': False},  # 东财 net flows + rate, but history only starts ~2023
    'hk_hold':       {'enabled': False},  # 北向个股: dead since ~Aug 2024
    'cyq_chips':     {'enabled': False},  # raw chip distribution; heavy
    'top_list':      {'enabled': False},  # 龙虎榜个股: retail-heavy, sparse
}
# ───────────────────────────────────────────────────────────


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
        if len(df) < PAGE_LIMIT:          # last page
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


def find_data_start(pro, api_name, days):
    """Binary-search the first date carrying data, so we skip an empty-history prefix
    instead of grinding years of blanks. Returns an index into `days`, or None if the
    endpoint has no data anywhere in the range."""
    lo, hi = 0, len(days) - 1
    if not call(pro, api_name, trade_date=days[lo]).empty:
        return 0
    if call(pro, api_name, trade_date=days[hi]).empty:
        return None
    while lo + 1 < hi:                      # invariant: days[lo] empty, days[hi] has data
        mid = (lo + hi) // 2
        if call(pro, api_name, trade_date=days[mid]).empty:
            lo = mid
        else:
            hi = mid
        time.sleep(THROTTLE_S)
    return hi


def fetch_endpoint(pro, api_name, days):
    out_dir = os.path.join(CACHE_DIR, '_partial', api_name)
    os.makedirs(out_dir, exist_ok=True)

    # skip an empty-history prefix unless we already have real rows cached
    if not glob.glob(os.path.join(out_dir, '*.parquet')):
        s = find_data_start(pro, api_name, days)
        if s is None:
            print(f"\n[{api_name}] no data anywhere in {days[0]}..{days[-1]} — skipping.")
            return
        if s > 0:
            print(f"\n[{api_name}] data starts ~{days[s]} — skipping {s} empty earlier dates.")
        days = days[s:]

    todo = [d for d in days if d not in _done(out_dir)]
    print(f"\n[{api_name}] {len(todo)} dates to fetch ({len(days) - len(todo)} cached)")

    consec_fail = 0
    for i, d in enumerate(todo, 1):
        try:
            df = call(pro, api_name, trade_date=d)
            consec_fail = 0
        except Exception as e:
            consec_fail += 1
            print(f"   {api_name} {d}: failed ({e})")
            if consec_fail >= 3:
                print(f"!! {api_name}: 3 consecutive failures — aborting this endpoint "
                      f"(check the name / your points against the doc). Other endpoints continue.")
                return
            continue
        if df.empty:
            open(os.path.join(out_dir, f"{d}.empty"), 'w').close()   # marker so we don't refetch
        else:
            df.to_parquet(os.path.join(out_dir, f"{d}.parquet"))
        if i % 25 == 0 or i == len(todo):
            print(f"   {api_name}: {i}/{len(todo)}  last={d}  rows={len(df)}")
        time.sleep(THROTTLE_S)

    _combine(api_name, out_dir)


def _combine(api_name, out_dir):
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(out_dir, '*.parquet')))]
    parts = [p for p in parts if not p.empty]
    if not parts:
        print(f"   {api_name}: no rows for this range (endpoint empty here, or skipped).")
        return
    full = pd.concat(parts, ignore_index=True).drop_duplicates()
    # add a bare 6-digit 'code' (strip .SZ/.SH/.BJ) to match the local lake's zero-padded
    # string key; ts_code is kept as the canonical Tushare id. Merge later on (date, code).
    if 'ts_code' in full.columns:
        full['code'] = full['ts_code'].astype(str).str[:6]
    dest = os.path.join(CACHE_DIR, f"{api_name}.parquet")
    full.to_parquet(dest)
    ndates = full['trade_date'].nunique() if 'trade_date' in full.columns else len(parts)
    print(f"   -> {dest}  ({len(full):,} rows, {ndates} dates)")
    print(f"      cols: {list(full.columns)}")


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    pro = get_pro()
    days = trading_days(pro)
    print(f"{len(days)} trading days  {START_DATE} -> {END_DATE}")
    for api_name, opt in ENDPOINTS.items():
        if not opt.get('enabled'):
            continue
        try:
            fetch_endpoint(pro, api_name, days)
        except Exception as e:
            print(f"!! {api_name} aborted: {e}  (continuing with the rest)")
    produced = sorted(glob.glob(os.path.join(CACHE_DIR, '*.parquet')))
    print("\nDone. Upload these:")
    for f in produced:
        print(f"   {f}")


if __name__ == '__main__':
    main()
