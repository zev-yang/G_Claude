"""
moneyflow_factors.py — 主力 (smart-money) factors from Tushare basic `moneyflow`.

Reads the per-date partial files directly (one pyarrow pass over the folder) — no manual
combine step. All factors are unit/adjustment-free ratios (净额 / 当日总流入), so 未复权 is a
non-issue. 主力净额 := (buy_lg+buy_elg) - (sell_lg+sell_elg), NOT net_mf_amount (which is not
Sum(buy)-Sum(sell); those equal turnover).

MEMORY-FRUGAL (factor math unchanged — identical outputs to the prior version):
  • amounts downcast to float32 on load (halves the float memory),
  • frame reduced to the 4 needed columns BEFORE the rolling pass (frees the raw amount +
    string id columns immediately, instead of carrying ~5M object strings through 3 transforms),
  • duplicate (date, code) rows dropped so the downstream `df.join` in factors.py can't
    row-explode on repeated keys.

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
_AMT = ['buy_sm_amount', 'buy_md_amount', 'buy_lg_amount',
        'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']


def load_moneyflow(src='tushare_cache/_partial/moneyflow'):
    """Read every per-date parquet in `src` (folder) in one pyarrow pass, or a single
    combined parquet if `src` is a file. Only the needed columns; amounts as float32."""
    if os.path.isfile(src):
        df = pd.read_parquet(src, columns=_COLS)
    else:
        files = sorted(glob.glob(os.path.join(src, '*.parquet')))   # ignores any .empty markers
        if not files:
            raise FileNotFoundError(f"no parquet files in {src}")
        import pyarrow.dataset as ds
        df = ds.dataset(files, format='parquet').to_table(columns=_COLS).to_pandas()
    for c in _AMT:                                  # halve float memory before the heavy steps
        if c in df.columns and df[c].dtype != 'float32':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def build_moneyflow_factors(mf):
    """Raw moneyflow rows -> per-(date, code) factor panel (mf_cum20, mf_trend, elg_cum20)."""
    code = mf['ts_code'].astype(str).str[:6]
    date = pd.to_datetime(mf['trade_date'].astype(str), format='%Y%m%d')

    # same-day flow ratios (unit/adjustment-free): net large-order inflow / the day's buy flow
    main_net = (mf['buy_lg_amount'] + mf['buy_elg_amount']) - (mf['sell_lg_amount'] + mf['sell_elg_amount'])
    elg_net  =  mf['buy_elg_amount'] - mf['sell_elg_amount']
    total    = (mf['buy_sm_amount'] + mf['buy_md_amount'] +
                mf['buy_lg_amount'] + mf['buy_elg_amount']).replace(0, np.nan)

    # keep ONLY what the rolling needs — frees the raw amount + id columns right away
    mf = pd.DataFrame({
        'code': code, 'date': date,
        'mf_net_rate':  (main_net / total).astype('float32'),
        'elg_net_rate': (elg_net  / total).astype('float32'),
    })
    del code, date, main_net, elg_net, total

    # drop any accidental duplicate (date, code) so the downstream join can't multiply rows
    mf = mf.drop_duplicates(['code', 'date'], keep='last')

    # accumulation over time (per stock, chronological)
    mf = mf.sort_values(['code', 'date'])
    gm = mf.groupby('code', sort=False)
    mf['mf_cum20']  = gm['mf_net_rate'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    mf_cum5         = gm['mf_net_rate'].transform(lambda s: s.rolling(5,  min_periods=3).sum())
    mf['mf_trend']  = (mf_cum5 / 5.0 - mf['mf_cum20'] / 20.0).astype('float32')   # accumulation accelerating?
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
          "| stocks", fac.index.get_level_values('code').nunique(),
          "| dup (date,code):", int(fac.index.duplicated().sum()))
    print("\nnon-NaN per factor (rolling needs ~20 trading days):")
    print(fac.notna().sum().to_string())
    print("\nsample (rows where all three are populated):")
    print(fac.dropna().head(8).to_string())
