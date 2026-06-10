"""
fundamental_factors.py — 基本面/估值 factors from Tushare `daily_basic` (建议③).

The 48-factor pool is pure price/volume — the model has never seen SIZE or VALUATION.
This module turns the daily_basic partials (already pulled by run_data_update.py, ALL
fields) into 5 same-day factors and hands them to the standard pipeline: cross-sectional
pct-rank -> ICIR structured selection -> LGBM. ★ NO hand weights, NO thresholds — the
per-window model decides usage; the Factor Audit shows whether they earn their seat.

Factors (all same-day published values -> trivially point-in-time; signal t, trade t+1 open):
    size_lnmv    ln(流通市值)            size — the strongest documented A-share cross-sectional effect
    val_pe_ttm   市盈率 TTM (亏损 -> NaN)  valuation
    val_pb       市净率                  valuation
    val_dv_ttm   股息率 TTM              valuation / quality-income
    to_rate_f    换手率(自由流通股)        true turnover level (existing factors only proxy its CV)

NaN policy: these columns are EXCLUDED from the global dropna in factors.py — a loss-making
company (pe_ttm NaN) must NOT fall out of the universe. NaN survives rank(pct=True) and is
handled natively by LGBM.

Reads per-date parquet partials via fastparquet, column-tolerant (older partials missing a
field simply yield NaN for that factor).
"""
import glob
import os

import numpy as np
import pandas as pd

_ENGINE = 'fastparquet'

FUND_FACTORS = ['size_lnmv', 'val_pe_ttm', 'val_pb', 'val_dv_ttm', 'to_rate_f']

# source columns (tolerated if absent in older files)
_WANT = ['ts_code', 'trade_date', 'circ_mv', 'pe_ttm', 'pb', 'dv_ttm',
         'turnover_rate_f', 'turnover_rate']


def _read_one(f):
    """Read one partial with whatever subset of _WANT it has (column-tolerant)."""
    df = pd.read_parquet(f, engine=_ENGINE)
    keep = [c for c in _WANT if c in df.columns]
    return df[keep]


def load_daily_basic(src='tushare_cache/_partial/daily_basic'):
    if os.path.isfile(src):
        return _read_one(src)
    files = sorted(glob.glob(os.path.join(src, '*.parquet')))   # ignores .empty markers
    if not files:
        raise FileNotFoundError(f"no parquet files in {src} — run run_data_update.py first")
    return pd.concat((_read_one(f) for f in files), ignore_index=True)


def fundamentals_panel(src='tushare_cache/_partial/daily_basic'):
    """daily_basic partials -> (date, code)-indexed float32 panel of FUND_FACTORS."""
    db = load_daily_basic(src)
    if 'circ_mv' not in db.columns:
        raise ValueError("daily_basic partials lack circ_mv — refetch with run_data_update.py")

    code = db['ts_code'].astype(str).str[:6]
    date = pd.to_datetime(db['trade_date'].astype(str), format='%Y%m%d')

    def _num(col):
        return (pd.to_numeric(db[col], errors='coerce')
                if col in db.columns else pd.Series(np.nan, index=db.index))

    circ_mv = _num('circ_mv')
    # true free-float turnover preferred; plain turnover_rate as row-level fallback
    to_f = _num('turnover_rate_f').fillna(_num('turnover_rate'))

    out = pd.DataFrame({
        'date': date, 'code': code,
        'size_lnmv':  np.log(circ_mv.where(circ_mv > 0)).astype('float32'),
        'val_pe_ttm': _num('pe_ttm').astype('float32'),     # 亏损 -> NaN (kept NaN by design)
        'val_pb':     _num('pb').astype('float32'),
        'val_dv_ttm': _num('dv_ttm').astype('float32'),
        'to_rate_f':  to_f.astype('float32'),
    })
    out = out.drop_duplicates(['date', 'code'], keep='last')
    return out.set_index(['date', 'code'])[FUND_FACTORS].sort_index()


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'tushare_cache/_partial/daily_basic'
    p = fundamentals_panel(src)
    print(f"fundamentals panel: {len(p):,} rows, "
          f"{p.index.get_level_values('date').nunique()} days "
          f"({p.index.get_level_values('date').min().date()}..{p.index.get_level_values('date').max().date()})")
    print("\nnon-null coverage per factor:")
    print((p.notna().mean() * 100).round(1).astype(str) + '%')
    print("\nsample (last day, first 5 codes):")
    print(p.xs(p.index.get_level_values('date').max(), level='date').head().to_string())
