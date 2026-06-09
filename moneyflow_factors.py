"""
moneyflow_factors.py — 主力 (smart-money) factors from Tushare basic `moneyflow`.

Reads the per-date partial files directly — no manual combine step. All factors are
unit/adjustment-free ratios (净额 / 当日总买入额), so 未复权 is a non-issue. 主力净额 :=
(buy_lg+buy_elg) - (sell_lg+sell_elg), NOT net_mf_amount (which equals turnover, not net).

READER: uses fastparquet, NOT pyarrow (some Windows/conda pyarrow builds segfault on
read with no traceback). pip install fastparquet. Set _ENGINE=None to use pandas default.

OUTPUT = MONEYFLOW_FACTORS + MONEYFLOW_HELPERS:
  • MONEYFLOW_FACTORS (mf_cum20, mf_trend, elg_cum20) — ranked into the panel by factors.py.
  • MONEYFLOW_HELPERS — RAW signed rates/ratios, ride along on the same .join() but are NOT
    ranked. Consumers (point-in-time by construction, no fitted thresholds):
      - elg_net_rate ........ factors.py -> 阴线吸筹 factor mf_dipbuy (down-day accumulation)
      - main_net_3d, sm_net_3d  portfolio.py -> 拉高出货 veto (raw 3d SIGN)
      - mf_strength_8, mf_accel, sm_outflow_rate
                              factors.py -> Layer-4 资金流 overlay (mf_score), linear, NOT a
                              model feature. strength = Σ主力净额 / Σ当日总买入额 (ratio-of-sums).

Used by factors.py: moneyflow_panel(dir) -> per-(date, code) columns to .join().
Standalone check:  python moneyflow_factors.py tushare_cache/_partial/moneyflow
"""
import os
import glob
import numpy as np
import pandas as pd

MONEYFLOW_FACTORS = ['mf_cum20', 'mf_trend', 'elg_cum20']
# RAW helpers — joined alongside the factors but never ranked (consumed downstream).
MONEYFLOW_HELPERS = ['elg_net_rate', 'main_net_3d', 'sm_net_3d',
                     'mf_strength_8', 'mf_accel', 'sm_outflow_rate']
# sell_sm_amount needed for the 小单 net rate (veto) and 小单 outflow (retail_contrary).
_COLS = ['ts_code', 'trade_date', 'buy_sm_amount', 'sell_sm_amount', 'buy_md_amount',
         'buy_lg_amount', 'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']
_AMT = ['buy_sm_amount', 'sell_sm_amount', 'buy_md_amount', 'buy_lg_amount',
        'sell_lg_amount', 'buy_elg_amount', 'sell_elg_amount']
_ENGINE = 'fastparquet'      # avoid the pyarrow read segfault; set to None to use pandas default


def load_moneyflow(src='tushare_cache/_partial/moneyflow'):
    """Read every per-date parquet in `src` (folder), or a single combined parquet if `src`
    is a file. Per-file reads via fastparquet (no Arrow); only the needed columns; float32."""
    if os.path.isfile(src):
        df = pd.read_parquet(src, columns=_COLS, engine=_ENGINE)
    else:
        files = sorted(glob.glob(os.path.join(src, '*.parquet')))   # ignores any .empty markers
        if not files:
            raise FileNotFoundError(f"no parquet files in {src}")
        df = pd.concat(
            (pd.read_parquet(f, columns=_COLS, engine=_ENGINE) for f in files),
            ignore_index=True)
    for c in _AMT:                                  # halve float memory before the heavy steps
        if c in df.columns and df[c].dtype != 'float32':
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
    return df


def build_moneyflow_factors(mf):
    """Raw moneyflow rows -> per-(date, code) panel = MONEYFLOW_FACTORS + MONEYFLOW_HELPERS."""
    code = mf['ts_code'].astype(str).str[:6]
    date = pd.to_datetime(mf['trade_date'].astype(str), format='%Y%m%d')

    # same-day flows (单位:万元). 主力 := lg+elg.  total := 当日总买入额 (the denominator, per spec).
    main_net = (mf['buy_lg_amount'] + mf['buy_elg_amount']) - (mf['sell_lg_amount'] + mf['sell_elg_amount'])
    elg_net  =  mf['buy_elg_amount'] - mf['sell_elg_amount']
    sm_net   =  mf['buy_sm_amount']  - mf['sell_sm_amount']
    total    = (mf['buy_sm_amount'] + mf['buy_md_amount'] +
                mf['buy_lg_amount'] + mf['buy_elg_amount']).replace(0, np.nan)

    # keep the daily rates + the raw main_net/total (the latter feed the ratio-of-sums strength)
    mf = pd.DataFrame({
        'code': code, 'date': date,
        'mf_net_rate':  (main_net / total).astype('float32'),   # 大单+特大单 净流入率 (主力)
        'elg_net_rate': (elg_net  / total).astype('float32'),   # 特大单 净流入率
        'sm_net_rate':  (sm_net   / total).astype('float32'),   # 小单 净流入率
        '_main_net':    main_net.astype('float32'),             # raw, for Σ主力净额/Σ成交额
        '_total':       total.astype('float32'),                # raw 当日总买入额
    })
    del code, date, main_net, elg_net, sm_net, total

    mf = mf.drop_duplicates(['code', 'date'], keep='last')
    mf = mf.sort_values(['code', 'date'])
    gm = mf.groupby('code', sort=False)

    # ── existing 3 factors (unchanged) ────────────────────────────────────────────────
    mf['mf_cum20']  = gm['mf_net_rate'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    mf_cum5         = gm['mf_net_rate'].transform(lambda s: s.rolling(5,  min_periods=3).sum())
    mf['mf_trend']  = (mf_cum5 / 5.0 - mf['mf_cum20'] / 20.0).astype('float32')
    mf['elg_cum20'] = gm['elg_net_rate'].transform(lambda s: s.rolling(20, min_periods=10).sum())

    # ── 拉高出货 veto helpers: 3-day cumulative SIGN of 主力 vs 小单 (raw) ──────────────
    mf['main_net_3d'] = gm['mf_net_rate'].transform(lambda s: s.rolling(3, min_periods=2).sum()).astype('float32')
    mf['sm_net_3d']   = gm['sm_net_rate'].transform(lambda s: s.rolling(3, min_periods=2).sum()).astype('float32')

    # ── Layer-4 资金流 overlay helpers ─────────────────────────────────────────────────
    # mf_strength_8 / _20 = Σ主力净额 / Σ当日总买入额 (RATIO-OF-SUMS, less noisy than Σrate).
    # mf_accel = strength_8 - strength_20  (设计里叫 mf_accel_5, 其实是 8 日 vs 20 日之差).
    s8_main  = gm['_main_net'].transform(lambda s: s.rolling(8,  min_periods=4).sum())
    s8_tot   = gm['_total'].transform(lambda s: s.rolling(8,  min_periods=4).sum())
    s20_main = gm['_main_net'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    s20_tot  = gm['_total'].transform(lambda s: s.rolling(20, min_periods=10).sum())
    mf['mf_strength_8'] = (s8_main  / s8_tot ).astype('float32')
    _strength_20        = (s20_main / s20_tot)
    mf['mf_accel']      = (mf['mf_strength_8'] - _strength_20).astype('float32')
    # 小单净流出率 (daily) = (sell_sm - buy_sm)/total = -小单净流入率. retail_contrary 取其当日横截面分位。
    mf['sm_outflow_rate'] = (-mf['sm_net_rate']).astype('float32')

    return (mf.set_index(['date', 'code'])[MONEYFLOW_FACTORS + MONEYFLOW_HELPERS]
              .astype('float32').sort_index())


def moneyflow_panel(src='tushare_cache/_partial/moneyflow'):
    """Load partials (or a combined file) -> panel indexed (date, code)."""
    return build_moneyflow_factors(load_moneyflow(src))


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'tushare_cache/_partial/moneyflow'
    fac = moneyflow_panel(src)
    print("moneyflow panel:", fac.shape, "| cols", list(fac.columns),
          "| dates", fac.index.get_level_values('date').nunique(),
          "| stocks", fac.index.get_level_values('code').nunique(),
          "| dup:", int(fac.index.duplicated().sum()))
    print("\nnon-NaN per column:")
    print(fac.notna().sum().to_string())
    print("\nsample (3 cum factors populated):")
    print(fac.dropna(subset=MONEYFLOW_FACTORS).head(6).to_string())
