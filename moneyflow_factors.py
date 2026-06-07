"""
moneyflow_factors.py — 主力 (smart-money) factors from Tushare basic `moneyflow`.

Reads the per-date partial files directly (same idea as data_loader reading your
5000+ stock files) — no manual combine step. One fast pyarrow pass over the folder.

All factors are unit/adjustment-free ratios (净额 / 当日总流入), so 未复权 is a non-issue.
主力净额 is defined transparently as (buy_lg+buy_elg) - (sell_lg+sell_elg) rather than
trusting net_mf_amount (which is NOT Sum(buy)-Sum(sell); those equal turnover).

Used by factors.py: moneyflow_panel(dir) -> per-(date, code) factor columns to .join().
Standalone check:  python moneyflow_factors.py tushare_cache/_partial/moneyflow
"""
import os
import glob
import numpy as np
import pandas as pd

MONEYFLOW_FACTORS = ['mf_cum20', 'mf_trend', 'elg_cum20']
_COLS = ['ts_code', 'trade_date', 'buy_sm_amount', 'buy_md_amount',
         'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']


def load_moneyflow(src='tushare_cache/_partial/moneyflow'):
    """Read every per-date parquet in `src` (folder) in one pyarrow pass, or a single
    combined parquet if `src` is a file. Only the columns the factors need."""
    if os.path.isfile(src):
        return pd.read_parquet(src, columns=_COLS)
    files = sorted(glob.glob(os.path.join(src, '*.parquet')))   # ignores any .empty markers
    if not files:
        raise FileNotFoundError(f"no parquet files in {src}")
    import pyarrow.dataset as ds
    return ds.dataset(files, format='parquet').to_table(columns=_COLS).to_pandas()


def build_moneyflow_factors(mf):
    """Raw moneyflow rows -> per-(date, code) factor panel (mf_cum20, mf_trend, elg_cum20)."""
    mf = mf.copy()
    mf['code'] = mf['ts_code'].astype(str).str[:6]
    mf['date'] = pd.to_datetime(mf['trade_date'].astype(str), format='%Y%m%d')

    # same-day flow ratios (unit/adjustment-free): net large-order inflow / the day's buy flow
    main_net = (mf['buy_lg_amount'] + mf['buy_elg_amount']) - (mf['sell_lg_amount'] + mf['sell_elg_amount'])
    elg_net  =  mf['buy_elg_amount'] - mf['sell_elg_amount']
    total    = (mf['buy_sm_amount'] + mf['buy_md_amount'] +
                mf['buy_lg_amount'] + mf['buy_elg_amount']).replace(0, np.nan)
    mf['mf_net_rate']  = main_net / total
    mf['elg_net_rate'] = elg_net  / total

    # accumulation over time (per stock, chronological)
    mf = mf.sort_values(['code', 'date'])
    gm = mf.groupby('code', sort=False)
    mf['mf_cum20']  = gm['mf_net_rate'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    mf_cum5         = gm['mf_net_rate'].transform(lambda s: s.rolling(5,  min_periods=3).sum())
    mf['mf_trend']  = mf_cum5 / 5.0 - mf['mf_cum20'] / 20.0          # is accumulation accelerating?
    mf['elg_cum20'] = gm['elg_net_rate'].transform(lambda s: s.rolling(20, min_periods=10).sum())

    return mf.set_index(['date', 'code'])[MONEYFLOW_FACTORS].astype('float32').sort_index()


def moneyflow_panel(src='tushare_cache/_partial/moneyflow'):
    """Load partials (or a combined file) -> factor panel indexed (date, code)."""
    return build_moneyflow_factors(load_moneyflow(src))


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'tushare_cache/_partial/moneyflow'
    fac = moneyflow_panel(src)
    print("moneyflow factor panel:", fac.shape,
          "| dates", fac.index.get_level_values('date').nunique(),
          "| stocks", fac.index.get_level_values('code').nunique())
    print("\nnon-NaN per factor (rolling needs ~20 trading days):")
    print(fac.notna().sum().to_string())
    print("\nsample (rows where all three are populated):")
    print(fac.dropna().head(8).to_string())
