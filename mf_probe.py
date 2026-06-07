"""
mf_probe.py — pinpoint exactly where the moneyflow pipeline dies. Self-contained
(does not import moneyflow_factors), flushes after every step so the LAST printed line
is the failure point even on a native crash with no Python traceback.

Run from the project folder:
    python mf_probe.py tushare_cache\\_partial\\moneyflow
(or pass the absolute path to your moneyflow cache folder.)

Read the result by which stage prints last:
  • dies in STAGE 2 at a filename  -> ONE corrupt parquet file (re-fetch just that date; not a format problem)
  • passes 2, dies in STAGE 3      -> the pyarrow.dataset MULTI-FILE reader (native crash) -> read per-file instead, still parquet, no re-download
  • passes 3, dies in STAGE 4      -> the rolling/groupby BUILD (memory or pandas) -> not parquet at all
  • ALL STAGES PASS                -> load+build is fine alone; the crash is the JOIN/combined-memory inside run_icir
"""
import os, glob, sys, time, gc, traceback
import numpy as np, pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else r'tushare_cache\_partial\moneyflow'
COLS = ['ts_code', 'trade_date', 'buy_sm_amount', 'buy_md_amount',
        'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']

def p(*a): print(*a, flush=True)

p("STAGE 0: env")
import pyarrow
p(f"  pyarrow {pyarrow.__version__} | pandas {pd.__version__}")

files = sorted(glob.glob(os.path.join(SRC, '*.parquet')))
p(f"STAGE 1: {len(files)} parquet files in {SRC}")
if not files:
    p("  >>> no parquet files at that path — fix the path arg"); sys.exit(1)

p("STAGE 2: read EACH file with pandas (pinpoints a single bad/corrupt file)")
ok = 0
for i, f in enumerate(files):
    try:
        _ = pd.read_parquet(f)
        ok += 1
    except Exception:
        p(f"  >>> FAILED reading {os.path.basename(f)}")
        traceback.print_exc()
        sys.exit(2)
    if i % 100 == 0 or i == len(files) - 1:
        p(f"  [{i+1}/{len(files)}] last ok: {os.path.basename(f)}")
p(f"  per-file pandas read: {ok}/{len(files)} clean")

p("STAGE 3: the CURRENT load path — pyarrow.dataset multi-file to_table")
try:
    import pyarrow.dataset as ds
    t = time.time()
    tbl = ds.dataset(files, format='parquet').to_table(columns=COLS)
    p(f"  to_table OK: {tbl.num_rows:,} rows in {time.time()-t:.1f}s")
    df = tbl.to_pandas(); del tbl; gc.collect()
    p(f"  to_pandas OK: {df.shape}")
except Exception:
    p("  >>> FAILED in pyarrow.dataset path")
    traceback.print_exc()
    sys.exit(3)

p("STAGE 4: build factors (rolling/groupby) from the loaded frame")
try:
    for c in COLS[2:]:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    code = df['ts_code'].astype(str).str[:6]
    date = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
    main = (df['buy_lg_amount'] + df['buy_elg_amount']) - (df['sell_lg_amount'] + df['sell_elg_amount'])
    elg  =  df['buy_elg_amount'] - df['sell_elg_amount']
    tot  = (df['buy_sm_amount'] + df['buy_md_amount'] +
            df['buy_lg_amount'] + df['buy_elg_amount']).replace(0, np.nan)
    g = pd.DataFrame({'code': code, 'date': date,
                      'r': (main / tot).astype('float32'),
                      'e': (elg  / tot).astype('float32')})
    del df, code, date, main, elg, tot; gc.collect()
    g = g.drop_duplicates(['code', 'date']).sort_values(['code', 'date'])
    gm = g.groupby('code', sort=False)
    g['mf_cum20']  = gm['r'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    g['elg_cum20'] = gm['e'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    fac = g.set_index(['date', 'code'])[['mf_cum20', 'elg_cum20']].astype('float32')
    p(f"  build OK: {fac.shape} | dup(date,code)={int(fac.index.duplicated().sum())} "
      f"| non-NaN mf_cum20={int(fac['mf_cum20'].notna().sum()):,}")
except Exception:
    p("  >>> FAILED in build")
    traceback.print_exc()
    sys.exit(4)

p("ALL STAGES PASSED — moneyflow load+build is fine in isolation;")
p("  => the crash is the integration in run_icir (df.join / combined memory while the price panel is resident).")
