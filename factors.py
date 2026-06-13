"""
factors.py — feature engineering (AlphaLabV25_1):
~45 cross-sectional factors + forward-return label.
"""
import gc
import numpy as np
import pandas as pd
import talib

from config import CONFIG, Timer

# 主力 moneyflow factors (Tushare basic `moneyflow`). Module-level so a quick
# `from factors import moneyflow_panel` confirms the hook is present; try/except → None
# means a missing/stale moneyflow_factors.py degrades gracefully instead of crashing.
try:
    from moneyflow_factors import moneyflow_panel, MONEYFLOW_FACTORS
except Exception:
    moneyflow_panel, MONEYFLOW_FACTORS = None, []


class AlphaLabV25_1:
    def __init__(self):
        self.factors = []

    def run(self, panel):
        print("\n📋 [2] Feature Engineering V3 (Regime-Conditional Beta)...")

        panel = panel[~panel['name'].str.contains(r'ST|\*ST|退', na=False)]
        panel = panel[panel['close_raw'] > 4.0]   # 复权迁移: 绝对价过滤用原始价

        with Timer("Calc Factors"):
            df = panel.copy()
            del panel
            gc.collect()

            for col in df.columns:
                if df[col].dtype == 'float64':
                    df[col] = df[col].astype('float32')
                elif df[col].dtype == 'int64':
                    df[col] = df[col].astype('int32')
            gc.collect()

            df = df.sort_values(['code', 'date'])
            g  = df.groupby('code')

            # ── Label ──────────────────────────────────────────────
            h              = CONFIG['horizon']
            open_entry     = g['open'].shift(-1)
            open_exit      = g['open'].shift(-(1 + h))
            df['open_entry'] = open_entry
            df['open_exit']  = open_exit
            df['open_entry_raw'] = g['open_raw'].shift(-1)   # 报告/下单显示用真实价格
            df['ret_pnl']    = (open_exit / open_entry) - 1.0

            log_ret            = np.log(open_exit) - np.log(open_entry)
            df['temp_log_ret'] = log_ret
            if CONFIG.get('residualize_label', True):
                # Neutralize the forward return against same-day size-decile peers
                # (a proxy for size/sector beta), then rank. The target then rewards
                # stock SELECTION (alpha), not size/beta drift. Toggle off to compare.
                df['_lab_sq'] = df.groupby('date')['amount'].transform(
                    lambda x: pd.qcut(x.rank(method='first'), 10, labels=False, duplicates='drop')
                ).fillna(4).astype('int32')
                _grp_mean = df.groupby(['date', '_lab_sq'])['temp_log_ret'].transform('mean')
                df['_resid'] = df['temp_log_ret'] - _grp_mean
                df['target'] = (df.groupby('date')['_resid']
                                  .rank(pct=True).astype('float32'))
                del df['_lab_sq'], df['_resid']
            else:
                df['target'] = (df.groupby('date')['temp_log_ret']
                                  .rank(pct=True).astype('float32'))
            del df['temp_log_ret']

            # ── Existing factors (unchanged) ───────────────────────
            df['pct_chg']   = g['close'].pct_change(1)
            df['ret_5']     = g['close'].pct_change(5)
            df['ret_10']    = g['close'].pct_change(10)
            df['ret_20']    = g['close'].pct_change(20)
            df['mom_acc']   = df['ret_5']  - df['ret_20']
            df['mom_acc10'] = df['ret_10'] - df['ret_20']
            df['reversal']  = -1 * df['pct_chg']

            v_ma20 = g['volume'].transform(lambda x: x.rolling(20).mean())
            df['vol_ratio']   = df['volume'] / (v_ma20 + 1.0)
            v_ma10 = g['volume'].transform(lambda x: x.rolling(10).mean())
            df['vol_ratio10'] = df['volume'] / (v_ma10 + 1.0)
            v_ma5  = g['volume'].transform(lambda x: x.rolling(5).mean())
            df['vol_ratio5']  = df['volume'] / (v_ma5  + 1.0)

            df['adv']   = g['amount'].transform(lambda x: x.rolling(20).mean())
            df['adv10'] = g['amount'].transform(lambda x: x.rolling(10).mean())
            df['adv5']  = g['amount'].transform(lambda x: x.rolling(5).mean())

            df['vol_20'] = g['pct_chg'].transform(lambda x: x.rolling(20).std())
            df['vol_10'] = g['pct_chg'].transform(lambda x: x.rolling(10).std())
            df['vol_5']  = g['pct_chg'].transform(lambda x: x.rolling(5).std())

            delta     = g['close'].diff()
            up, down  = delta.clip(lower=0), -1 * delta.clip(upper=0)
            roll_up   = up.groupby('code').transform(
                lambda x: x.ewm(span=14, adjust=False).mean())
            roll_down = down.groupby('code').transform(
                lambda x: x.ewm(span=14, adjust=False).mean())
            df['rsi'] = (100 - (100 / (1 + roll_up / (roll_down + 1e-9)))).astype('float32')

            print("   ...Calculating PV Correlation")
            df['pv_corr'] = np.nan
            for code, group in g:
                df.loc[group.index, 'pv_corr'] = (
                    group['close'].rolling(20).corr(group['volume'])
                    .astype('float32'))
            gc.collect()

            vwap               = df['amount'] / (df['volume'] + 1e-9)
            df['smart_proxy']  = df['close_raw'] / vwap   # 复权迁移: 原始价/原始VWAP, 语义不变

            v_std20  = g['volume'].transform(lambda x: x.rolling(20).std())
            v_mean20 = g['volume'].transform(lambda x: x.rolling(20).mean())
            df['turnover_cv']   = v_std20  / (v_mean20 + 1e-9)
            v_std10  = g['volume'].transform(lambda x: x.rolling(10).std())
            v_mean10 = g['volume'].transform(lambda x: x.rolling(10).mean())
            df['turnover_cv10'] = v_std10  / (v_mean10 + 1e-9)
            v_std5   = g['volume'].transform(lambda x: x.rolling(5).std())
            v_mean5  = g['volume'].transform(lambda x: x.rolling(5).mean())
            df['turnover_cv5']  = v_std5   / (v_mean5  + 1e-9)

            df['skew_20'] = g['pct_chg'].transform(lambda x: x.rolling(20).skew())
            df['skew_10'] = g['pct_chg'].transform(lambda x: x.rolling(10).skew())
            df['skew_5']  = g['pct_chg'].transform(lambda x: x.rolling(5).skew())

            prev_close     = g['close'].shift(1)
            amp_series     = (df['high'] - df['low']) / (prev_close + 1e-9)
            # PRUNED: amp_ma_10 / amp_ma_20 — 0/59 selected in every Factor Audit (dead weight)
            df['amp_ma_5']  = amp_series.groupby('code').transform(lambda x: x.rolling(5).mean())

            clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (
                  df['high'] - df['low'] + 1e-9)
            df['clv_ma5']    = clv.groupby(df.index.get_level_values('code')).transform(
                lambda x: x.rolling(5).mean()).astype('float32')
            df['illiq']      = df['pct_chg'].abs() / (df['amount'] / 1e8 + 1e-9)
            df['illiq_ma10'] = g['illiq'].transform(
                lambda x: np.log1p(x).rolling(10).mean()).astype('float32')
            ma20             = g['close'].transform(lambda x: x.rolling(20).mean())
            df['bias_20']    = (df['close'] / (ma20 + 1e-9) - 1).astype('float32')
            v_mean5z         = g['volume'].transform(lambda x: x.rolling(5).mean())
            v_std5z          = g['volume'].transform(lambda x: x.rolling(5).std())
            df['vol_z5']     = ((df['volume'] - v_mean5z) / (v_std5z + 1e-9)).astype('float32')

            # PRUNED: near_high — rank-duplicate of dist_high (close/max vs close/max−1, 横截面 rank 全等)
            # PRUNED: mom_10 — literal duplicate of ret_10 (同一公式 close.pct_change(10))
            rolling_max      = g['close'].transform(lambda x: x.rolling(60).max())
            df['dist_high']  = ((df['close'] / (rolling_max + 1e-9)) - 1).astype('float32')
            df['vol_z']      = g['volume'].transform(
                lambda x: (x - x.rolling(10).mean()) / (x.rolling(10).std() + 1e-9)).astype('float32')

            is_up             = (df['close'] > g['close'].shift(1)).astype('float32')
            df['up_vol']      = (is_up * df['volume']).astype('float32')
            vol_sum_up        = g['up_vol'].transform(lambda x: x.rolling(5).sum())
            vol_sum_total     = g['volume'].transform(lambda x: x.rolling(5).sum())
            df['buy_force']   = (vol_sum_up / (vol_sum_total + 1e-9)).astype('float32')
            if 'up_vol' in df.columns: del df['up_vol']

            df['pv_divergence'] = df['ret_5'].rank(pct=True) - df['vol_ratio'].rank(pct=True)
            # PRUNED: vol_stability — turnover_cv5 的倒数 (mean/std vs std/mean, 同序列同窗口),
            # rank 互为镜像 (IC 表里 ±0.020292 / ±0.327692 精确对称即铁证); 树模型不分方向, 留 turnover_cv5。

            # ── NEW: Intraday Strength (smoothed only — raw had near-zero ICIR) ─
            df['open_strength']     = (
                (df['close'] - df['open']) / (df['open'] + 1e-9)
            ).astype('float32')
            df['open_strength_ma5'] = g['open_strength'].transform(
                lambda x: x.rolling(5).mean()).astype('float32')

            # ── NEW: TA-Lib ──────────────────────────────────────────
            print("   ...Calculating TA-Lib Factors (MACD / ATR / BB)")
            df['macd_hist']   = np.nan
            df['atr_ratio']   = np.nan
            df['bb_position'] = np.nan
            for code, group in g:
                c  = group['close'].values.astype('float64')
                hi = group['high'].values.astype('float64')
                lo = group['low'].values.astype('float64')
                ix = group.index
                _, _, hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
                df.loc[ix, 'macd_hist']   = hist.astype('float32')
                atr = talib.ATR(hi, lo, c, timeperiod=14)
                df.loc[ix, 'atr_ratio']   = (atr / (c + 1e-9)).astype('float32')
                upper, _, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2)
                bb_pos = (c - lower) / (upper - lower + 1e-9)
                df.loc[ix, 'bb_position'] = np.clip(bb_pos, -0.5, 1.5).astype('float32')
            df['macd_hist']   = df['macd_hist'].astype('float32')
            df['atr_ratio']   = df['atr_ratio'].astype('float32')
            df['bb_position'] = df['bb_position'].astype('float32')
            gc.collect()

            # PRUNED: alpha_mom10 — ret_10 减去当日中位数 (每日常数), 横截面 rank 与 ret_10 完全相同
            # (IC 表 mom_10/ret_10/alpha_mom10 三者 IC/ICIR 六位小数全等即铁证); 且其独立 AlphaMomFam
            # 让族内去重拦不住 — 同一信号占两个名额。

            # ── NEW: Sector Relative Strength ────────────────────────
            df['size_quintile'] = df.groupby('date')['adv'].transform(
                lambda x: pd.qcut(
                    x.rank(method='first'), 5,
                    labels=False, duplicates='drop')
            ).fillna(2).astype('int32')
            df['sector_rs_5']  = (
                df['ret_5'] -
                df.groupby(['date', 'size_quintile'])['ret_5'].transform('median')
            ).astype('float32')
            df['sector_rs_10'] = (
                df['ret_10'] -
                df.groupby(['date', 'size_quintile'])['ret_10'].transform('median')
            ).astype('float32')
            if 'size_quintile' in df.columns: del df['size_quintile']
            # ══════════════════════════════════════════════════════════════
            # SECTION A — New factor computation
            # Paste this block inside AlphaLabV25_1.run(), AFTER the existing
            # sector_rs_5/10 calculation and BEFORE the market-level indicators.
            # ══════════════════════════════════════════════════════════════
            #
            # ── A1. Opening Gap (开盘% equivalent) ──────────────────────
            # Captures overnight information: institutional pre-market positioning,
            # news reactions, foreign market moves.
            # Different from open_strength (which is close vs open intraday).
            df['gap']     = (df['open'] / g['close'].shift(1) - 1.0).astype('float32')
            df['gap_ma3'] = g['gap'].transform(lambda x: x.rolling(3).mean()).astype('float32')
            # ── A2. Sector Heat (热度 equivalent) ───────────────────────
            # Measures how "active/hot" the stock's peer group is.
            # Calculated as: avg peer return × avg peer volume ratio
            # A high value means: the whole sector is moving with volume = genuine trend.
            #
            # Step 1: compute size quintile (reuse from sector_rs calculation)
            df['_sq'] = df.groupby('date')['adv'].transform(
                lambda x: pd.qcut(x.rank(method='first'), 5, labels=False, duplicates='drop')
            ).fillna(2).astype('int32')
            #
            # Step 2: peer group average return (5-day)
            df['peer_ret5'] = df.groupby(['date', '_sq'])['ret_5'].transform('mean').astype('float32')
            #
            # Step 3: peer group average volume ratio
            df['peer_vol']  = df.groupby(['date', '_sq'])['vol_ratio'].transform('mean').astype('float32')
            #
            # Step 4: sector heat = peer trend × peer activity
            df['sector_heat'] = (df['peer_ret5'] * df['peer_vol']).astype('float32')
            #
            # Step 5: sector breadth (板块支撑) = % of peer group with positive 5d return
            df['sector_support'] = df.groupby(['date', '_sq'])['ret_5'].transform(
                lambda x: (x > 0).mean()
            ).astype('float32')
            #
            # Step 6: cleanup temp columns
            for tmp in ['_sq', 'peer_ret5', 'peer_vol']:
                if tmp in df.columns: del df[tmp]
            gc.collect()
            
            # ── NEW: VPT (directional volume-price trend) ────────────
            print("   ...Calculating VPT")
            df['vpt'] = np.nan
            for code, group in g:
                ret_    = group['close'].pct_change().fillna(0)
                vpt_raw = (ret_ * group['volume']).cumsum()
                vpt_z   = ((vpt_raw - vpt_raw.rolling(20).mean())
                           / (vpt_raw.rolling(20).std() + 1e-9))
                df.loc[group.index, 'vpt'] = vpt_z.clip(-3, 3).astype('float32')
            df['vpt'] = df['vpt'].astype('float32')
            gc.collect()

            # ═══════════════════════════════════════════════════════
            # NEW: Rolling Beta  +  FIX 1 (regime-conditional flip)
            # ═══════════════════════════════════════════════════════
            print("   ...Calculating Rolling Beta (regime-conditional)")
            mkt_ret_daily = df.groupby('date')['pct_chg'].transform('median')

            # Step A: compute raw beta
            df['beta_60d_raw'] = np.nan
            for code, group in g:
                s_ret = group['pct_chg']
                m_ret = mkt_ret_daily.loc[group.index]
                cov   = s_ret.rolling(60).cov(m_ret)
                mvar  = m_ret.rolling(60).var()
                beta  = (cov / (mvar + 1e-9)).clip(-3.0, 3.0)
                df.loc[group.index, 'beta_60d_raw'] = beta.astype('float32')
            df['beta_60d_raw'] = df['beta_60d_raw'].astype('float32')

            # Step B: compute daily market breadth (needed for regime flag)
            ma5_tmp        = g['close'].transform(lambda x: x.rolling(5).mean())
            is_above_ma5_tmp = (df['close'] > ma5_tmp).astype(float)
            daily_breadth  = is_above_ma5_tmp.groupby('date').transform('mean')

            # UPGRADE Edit 8: feed RAW beta. The old code flipped beta's sign on bear days
            # inside the same column, so one feature meant opposite things on different days
            # (a hidden state the model can't see). Let ICIR/LGBM learn the direction instead.
            is_bear        = daily_breadth < 0.40   # kept only for the diagnostic print below
            df['beta_60d'] = df['beta_60d_raw'].astype('float32')
            del df['beta_60d_raw']
            gc.collect()

            print(f"   ...Beta regime: bear days={is_bear.groupby('date').first().sum()}, "
                  f"bull days={(~is_bear).groupby('date').first().sum()}")

            # ── Market-level indicators (unchanged) ──────────────────
            df['mkt_breadth'] = daily_breadth.astype('float32')

            df['pct_chg_raw'] = g['close'].pct_change()
            cs_vol            = df.groupby('date')['pct_chg_raw'].transform('std')
            df['cs_vol_ma5']  = cs_vol.groupby(
                df.index.get_level_values('code')
            ).transform(lambda x: x.rolling(5).mean()).astype('float32')
            del df['pct_chg_raw']

            df['is_limit_down']    = (df['close'] / g['close'].shift(1) - 1.0 < -0.098).astype(float)
            df['limit_down_count'] = df.groupby('date')['is_limit_down'].transform('sum').astype('float32')
            df['mkt_low_level']    = df.groupby('date')['low'].transform('median').astype('float32')

            df['is_big_cap'] = (g['amount']
                                .transform(lambda x: x.rolling(20).mean())
                                .groupby('date').rank(ascending=False) <= 300)
            df['brd_300']  = df[df['is_big_cap']].groupby('date')['mkt_breadth'].transform('mean')
            df['brd_1000'] = df[~df['is_big_cap']].groupby('date')['mkt_breadth'].transform('mean')
            df['brd_300']  = df['brd_300'].ffill().fillna(df['mkt_breadth'])
            df['brd_1000'] = df['brd_1000'].ffill().fillna(df['mkt_breadth'])

            # ── Accumulation / Distribution factors (主力 footprints) ──────────
            # Detect institutional accumulation early; avoid buying into distribution.
            # All vectorised (groupby-transform) and all backward-looking (no look-ahead) —
            # the only forward-looking column in this file remains the label.
            #   downside_rs : mean excess return on DOWN-market days over 40d — a supported
            #                 stock holds up when the market falls (抗跌 / accumulation).
            #   accum_trend : 10-day change in the 20-day up-volume share — rising buy
            #                 pressure while price is quiet is the accumulation signature.
            #   coil        : 40d vs 10d average range — >1 means range is compressing
            #                 (蓄势 / coiling before a markup).
            _mkt_ret = df.groupby('date')['pct_chg'].transform('median')
            df['_dr_exc']   = (df['pct_chg'] - _mkt_ret).astype('float32')
            df['_dr_dnexc'] = df['_dr_exc'].where(_mkt_ret < 0)          # down-market days only
            df['downside_rs'] = g['_dr_dnexc'].transform(
                lambda x: x.rolling(40, min_periods=5).mean()).astype('float32')

            _is_up = (df['close'] > g['close'].shift(1)).astype('float32')
            df['_acc_upvol'] = (_is_up * df['volume']).astype('float32')
            _upsum20  = g['_acc_upvol'].transform(lambda x: x.rolling(20).sum())
            _volsum20 = g['volume'].transform(lambda x: x.rolling(20).sum())
            df['_acc_bf20'] = (_upsum20 / (_volsum20 + 1e-9)).astype('float32')
            df['accum_trend'] = (df['_acc_bf20'] - g['_acc_bf20'].shift(10)).astype('float32')

            df['_coil_tr'] = ((df['high'] - df['low']) / (g['close'].shift(1) + 1e-9)).astype('float32')
            _coil_r10 = g['_coil_tr'].transform(lambda x: x.rolling(10).mean())
            _coil_r40 = g['_coil_tr'].transform(lambda x: x.rolling(40).mean())
            df['coil'] = (_coil_r40 / (_coil_r10 + 1e-9)).astype('float32')

            for _t in ['_dr_exc', '_dr_dnexc', '_acc_upvol', '_acc_bf20', '_coil_tr']:
                if _t in df.columns: del df[_t]
            gc.collect()

            # ── Regime features (UPGRADE: market-state context for the model) ──
            # Market-LEVEL series (same value for every stock on a date). NOT added to
            # self.factors, so they bypass cross-sectional rank + ICIR selection (which
            # would be meaningless for a constant-per-date series). They are appended
            # directly to the model feature set in backtest.py / run.py, letting the GBDT
            # condition stock-factor splits on the regime (e.g. favour momentum in a bull,
            # reversal in a bear). Kept as raw levels (not rolling-z) so a persistent bear
            # still reads as genuinely low. Members: mkt_breadth, cs_vol_ma5 (above),
            # market_vol_ratio (built below), and mkt_trend_20 (here). LGBM handles the
            # warmup NaNs natively; all backtest dates are post-warmup anyway.
            df['mkt_trend_20'] = df.groupby('date')['ret_20'].transform('mean').astype('float32')

            # ── Register factors ─────────────────────────────────────
            self.factors = [
                # ── Momentum ────────────────────────────────────────────
                'mom_acc', 'ret_10', 'ret_20', 'dist_high', 'bias_20',
                'macd_hist',          # talib MACD histogram (ICIR 0.154)
                'bb_position',        # talib Bollinger Band position
                'sector_rs_5',        # outperformance vs size peers 5d (ICIR 0.131)
                'sector_rs_10',       # outperformance vs size peers 10d
                'gap',           # [NEW] opening gap → 开盘%
                'gap_ma3',       # [NEW] smoothed gap
                # ── Reversion ───────────────────────────────────────────
                'reversal', 'rsi',
                'beta_60d',           # raw beta, no inversion — ICIR handles direction
                'clv_ma5',            # close location value 5d MA
                # ── Volatility ──────────────────────────────────────────
                'vol_10', 'vol_20', 'vol_5',
                'skew_10', 'skew_20','skew_5',
                'atr_ratio',          # talib ATR / price
                # ── Volume / Liquidity ──────────────────────────────────
                'vol_ratio', 'vol_ratio5', 'vol_ratio10',
                'turnover_cv', 'turnover_cv10', 'turnover_cv5',
                'open_strength_ma5',  # smoothed intraday strength (raw ICIR=0.010 → excluded)
                'vpt',                # volume-price trend z-score
                'buy_force',
                'sector_heat',    # sector-level volume × trend activity
                'sector_support', # BUG7 FIX: moved from Reversion — % peers rising is MOMENTUM
                # ── Structure ───────────────────────────────────────────
                'pv_corr', 'smart_proxy', 'illiq_ma10',
                # ── Amplitude ───────────────────────────────────────────
                'amp_ma_5',
                # ── Market / Other ──────────────────────────────────────
                'vol_z', 'pv_divergence',
                # ── Accumulation / Distribution (主力 footprints) ────────
                'downside_rs',   # holds up on down-market days (support / 抗跌)
                'accum_trend',   # 20d up-volume share, rising = accumulation
                'coil',          # recent vs longer-term range compression (蓄势)
                # Removed: 'dist_risk' (ICIR 0.028 < floor, redundant +0.77 vol_ratio), 'streak' (ICIR 0.074), 'open_strength' raw (ICIR 0.010)
            ]
            # NOTE: 'streak' removed (ICIR=0.074 — below noise floor)
            # NOTE: 'open_strength' (raw) removed (ICIR=0.010 — pure noise)

            # ── NEW: 主力 moneyflow factors (Tushare basic `moneyflow`) ─────────
            # Defined in moneyflow_factors.py and joined here on the (date, code)
            # MultiIndex. They are deliberately LEFT AS NaN where moneyflow is missing
            # (per-stock rolling warmup, or names absent from the cache) and EXCLUDED
            # from the global dropna below, so that (a) check_ic's pairwise Spearman
            # reads a CLEAN IC over the covered cross-section (a neutral 0.5 fill would
            # add tied ranks and bias IC toward zero) and (b) LGBM uses its native NaN
            # handling in the backtest. If the cache is absent the run degrades
            # gracefully — the factors are simply skipped and the pipeline is unchanged.
            # ROLE: CONFIG['moneyflow_role'] decides how moneyflow is used —
            #   'screen' (default): ranked into the panel as a NEGATIVE SCREEN (drop the most-
            #            accumulated names before final selection); NOT a model feature.
            #   'factor': added to self.factors as model features (the older behaviour).
            #   'off'   : not joined at all (== the 48-factor baseline).
            # Columns are cross-sectionally pct-ranked per date; NaN where moneyflow is missing
            # (warmup / uncovered) — and NaN names are never screened or dropped.
            mf_added = []                                  # only non-empty in 'factor' mode
            _mf_role = CONFIG.get('moneyflow_role', 'screen')
            if moneyflow_panel is not None and _mf_role in ('screen', 'factor'):
                try:
                    _mf = moneyflow_panel(
                        CONFIG.get('moneyflow_path', './tushare_cache/_partial/moneyflow'))
                    # guard: ns vs us datetime-resolution mismatch silently yields an all-NaN join
                    _pan_dt = df.index.get_level_values('date').dtype
                    if _mf.index.get_level_values('date').dtype != _pan_dt:
                        _mf.index = _mf.index.set_levels(
                            _mf.index.levels[0].astype(_pan_dt), level='date')
                    df = df.join(_mf)                       # left join — keeps all price rows
                    _cols = [c for c in MONEYFLOW_FACTORS if c in df.columns]
                    _cov = df[_cols].notna().any(axis=1).mean() if _cols else 0.0
                    if _mf_role == 'factor':
                        self.factors += _cols              # ranked later by the standard loop
                        mf_added = _cols
                        print(f"   ...主力 moneyflow joined as MODEL FACTORS: {_cols} (coverage {_cov:.0%})")
                    else:  # 'screen' — not in self.factors, so rank here for the screen to use
                        for c in _cols:
                            df[c] = df.groupby(level='date')[c].rank(pct=True).astype('float32')
                        print(f"   ...主力 moneyflow joined as SCREEN columns (NOT model features): "
                              f"{_cols} (coverage {_cov:.0%})")
                    del _mf
                    gc.collect()
                except Exception as _e:
                    print(f"   ...⚠️  主力 moneyflow SKIPPED at join "
                          f"({type(_e).__name__}: {_e}) — pipeline continues without it.")
            else:
                print(f"   ...moneyflow_role='{_mf_role}'"
                      + (" / moneyflow_panel unavailable" if moneyflow_panel is None else "")
                      + " — moneyflow not joined (48-factor baseline).")

            # ── NEW (建议③): 基本面/估值 factors (Tushare daily_basic) ──────────────
            # 48 因子池纯价量, 模型从没见过市值/估值。这里把 daily_basic 的 5 个同日因子
            # 走【标准通道】接入: join -> rank(pct) -> ICIR 结构化选择 -> LGBM。
            # ★ 无手工权重/阈值 — 用不用、何时用, 由每窗口的模型自己决定 (Factor Audit 可见)。
            # ★ 必须排除在全局 dropna 之外 (见下) — 亏损股 pe_ttm 为 NaN, 不能因此掉出 universe。
            fund_added = []
            if CONFIG.get('USE_FUNDAMENTALS', False):
                try:
                    from fundamental_factors import fundamentals_panel, FUND_FACTORS
                    _fund = fundamentals_panel(
                        CONFIG.get('fundamentals_path', './tushare_cache/_partial/daily_basic'))
                    _pan_dt = df.index.get_level_values('date').dtype
                    if _fund.index.get_level_values('date').dtype != _pan_dt:
                        _fund.index = _fund.index.set_levels(
                            _fund.index.levels[0].astype(_pan_dt), level='date')
                    df = df.join(_fund)                     # left join — keeps all price rows
                    _fcols = [c for c in FUND_FACTORS if c in df.columns]
                    _fcov = df['size_lnmv'].notna().mean() if 'size_lnmv' in df.columns else 0.0
                    if _fcov >= 0.30:
                        self.factors += _fcols             # ranked later by the standard loop
                        fund_added = _fcols
                        print(f"   ...daily_basic joined as MODEL FACTORS: {_fcols} "
                              f"(coverage {_fcov:.0%})")
                    else:                                  # data missing/stale — don't half-join
                        df = df.drop(columns=_fcols)
                        print(f"   ...⚠️  daily_basic coverage only {_fcov:.0%} (<30%) — factors "
                              f"NOT added; run run_data_update.py to fetch daily_basic.")
                    del _fund
                    gc.collect()
                except Exception as _e:
                    print(f"   ...⚠️  daily_basic SKIPPED at join "
                          f"({type(_e).__name__}: {_e}) — pipeline continues without it.")

            # ── NEW: Layer-4 资金流 overlay 打分列 mf_score (线性叠加, 不喂 LGBM) ────────
            # mf_score = w_s·z(mf_strength_8) + w_a·z(mf_accel) + w_r·retail_contrary。
            # 三分量全 point-in-time; 阈值/权重全写死。mf_score 作为【独立打分列】保留 (不入
            # self.factors), 在 backtest/run 里以 final = zscore(base) + MF_WEIGHT·mf_score 叠加。
            # 无 moneyflow 覆盖 -> mf_strength_8 为 NaN -> mf_score NaN -> 叠加时按 0 (无 tilt)。
            if CONFIG.get('USE_MONEYFLOW', False) and 'mf_strength_8' in df.columns:
                # retail_contrary (需价): A 当日上涨, B 放量(>VOL_RATIO×20日均量),
                # C 乖离 |close/ma20-1|<BIAS_LIMIT; 在满足 A&B&C 的票里取【小单净流出】当日横截面前 10%。
                if (CONFIG.get('RETAIL_CONTRARY_ENABLE', True)
                        and {'close', 'volume', 'pct_chg'}.issubset(df.columns)):
                    _vr  = CONFIG.get('RETAIL_CONTRARY_VOL_RATIO', 1.2)
                    _bl  = CONFIG.get('RETAIL_CONTRARY_BIAS_LIMIT', 0.15)
                    _pcc = CONFIG.get('RETAIL_CONTRARY_PERCENTILE', 0.90)
                    _ma20c = (df['close'].sort_index(level=['code', 'date']).groupby(level='code')
                              .transform(lambda s: s.rolling(20, min_periods=10).mean()).reindex(df.index))
                    _ma20v = (df['volume'].sort_index(level=['code', 'date']).groupby(level='code')
                              .transform(lambda s: s.rolling(20, min_periods=10).mean()).reindex(df.index))
                    _A = df['pct_chg'] > 0
                    _B = df['volume'] > (_vr * _ma20v)
                    _C = (df['close'] / _ma20c - 1.0).abs() < _bl
                    _cond = _A & _B & _C
                    _tmp = df['sm_outflow_rate'].where(_cond)         # 仅在 A&B&C 票内排名
                    _rk  = _tmp.groupby(level='date').rank(pct=True)
                    df['retail_contrary'] = ((_rk > _pcc) & _cond).astype('float32')
                else:
                    df['retail_contrary'] = np.float32(0.0)
                # 当日横截面 z-score (clip ±3); NaN(未覆盖) 保留
                def _csz(_col):
                    _g = df.groupby('date')[_col]
                    return ((df[_col] - _g.transform('mean')) / (_g.transform('std') + 1e-9)).clip(-3, 3)
                _ws = CONFIG.get('MF_W_STRENGTH', 0.6)
                _wa = CONFIG.get('MF_W_ACCEL', 0.2)
                _wr = CONFIG.get('MF_W_RETAIL', 0.2)
                df['mf_score'] = (_ws * _csz('mf_strength_8')
                                  + _wa * _csz('mf_accel')
                                  + _wr * df['retail_contrary']).astype('float32')
                print(f"   ...mf_score built (overlay w={_ws}/{_wa}/{_wr}); "
                      f"coverage {int(df['mf_score'].notna().sum()):,} rows")

            # ── NEW (Stage-1 观察): IRCF 情境因子 ircf_score ───────────────────────────
            # 上涨日看机构大单净买入强度的横截面分位; 下跌日看小单卖出占比的横截面分位; 否则 0。
            # ★ 仅构建打分列 + 在 check_ic 里看它【自身 IC】(暂不接入选股, 先验证有没有 edge)。
            # 全 point-in-time, 阈值写死。无 moneyflow 覆盖 -> NaN (不污染 IC)。
            if CONFIG.get('USE_IRCF', False) and {'big_buy_strength', 'small_sell_ratio', 'pct_chg'}.issubset(df.columns):
                _up_hi = CONFIG.get('IRCF_MAX_RETURN_UP', 0.07)     # 上涨日剔涨停
                _dn_lo = CONFIG.get('IRCF_MIN_RETURN_DOWN', -0.05)  # 下跌日剔跌停
                _up   = (df['pct_chg'] > 0) & (df['pct_chg'] < _up_hi)
                _down = (df['pct_chg'] < 0) & (df['pct_chg'] > _dn_lo)
                _inst  = df['big_buy_strength'].where(_up).groupby(level='date').rank(pct=True)
                _panic = df['small_sell_ratio'].where(_down).groupby(level='date').rank(pct=True)
                _ircf = pd.Series(np.nan, index=df.index, dtype='float64')
                _ircf[_up]   = _inst[_up]              # 上涨日机构吸筹分位 (NaN where uncovered)
                _ircf[_down] = _panic[_down]           # 下跌日散户恐慌分位
                _covered = df['big_buy_strength'].notna()
                _ircf[_covered & ~_up & ~_down] = 0.0  # 有覆盖但既非涨日也非跌日 -> 0
                df['ircf_score'] = _ircf.astype('float32')
                print(f"   ...ircf_score built (Stage-1 观察; up<{_up_hi}, down>{_dn_lo}); "
                      f"coverage {int(df['ircf_score'].notna().sum()):,} rows")

            # ── 因子实验室 (observe-only): 新候选先量自身 IC, 不进选拔池/不进 LGBM ──────
            # ★ 上轮已证明: 动候选池会级联换队 (±5pp)。所以新因子必须先过这道零接触闸:
            #   列名以 lab_ 开头 -> 不在 self.factors -> 不参与 rank/选拔/dropna, 仅 check_ic 观察。
            #   窗口参数按提案原文写死 (20日相关/10日和), 不调。
            if CONFIG.get('USE_FACTOR_LAB', False):
                # lab_smart_intraday: 隔夜收益 vs 日内收益 的 20 日滚动相关 (机构/散户行为分解)
                _ov = (df['open'] / g['close'].shift(1) - 1.0)
                _in = (df['close'] / df['open'].replace(0, np.nan) - 1.0)
                _gc = lambda s, w, fn: s.groupby(s.index.get_level_values('code')).transform(
                    lambda x: getattr(x.rolling(w), fn)())
                _mxy = _gc(_ov * _in, 20, 'mean')
                _mx, _my = _gc(_ov, 20, 'mean'), _gc(_in, 20, 'mean')
                _sx, _sy = _gc(_ov, 20, 'std'),  _gc(_in, 20, 'std')
                df['lab_smart_intraday'] = ((_mxy - _mx * _my) /
                                            (_sx * _sy + 1e-12)).clip(-1, 1).astype('float32')
                # lab_upper_shadow: 上影线长度 × 量比, 10 日累计 (炸板/冲高回落微结构)
                _shadow = ((df['high'] - np.maximum(df['open'], df['close']))
                           / df['close'].replace(0, np.nan))
                df['lab_upper_shadow'] = _gc(_shadow * df['vol_ratio'], 10, 'sum').astype('float32')
                del _ov, _in, _mxy, _mx, _my, _sx, _sy, _shadow

                # ── 基本盘 (复权口径下复检): daily_basic 5 因子, 仅观察不进池 ──────────────
                # 旧结论 (34.84% vs 40.90%) 是未复权数据上得出的; 数据口径已变, 用观察舱复检。
                try:
                    from fundamental_factors import fundamentals_panel, FUND_FACTORS
                    _fp = fundamentals_panel(
                        CONFIG.get('fundamentals_path', './tushare_cache/_partial/daily_basic'))
                    _pdt = df.index.get_level_values('date').dtype
                    if _fp.index.get_level_values('date').dtype != _pdt:
                        _fp.index = _fp.index.set_levels(_fp.index.levels[0].astype(_pdt), level='date')
                    for _f in FUND_FACTORS:
                        if _f in _fp.columns:
                            df['lab_' + _f] = _fp[_f].reindex(df.index).astype('float32')
                    del _fp
                except Exception as _e:
                    print(f"   ...⚠️ 基本盘观察舱 SKIPPED ({type(_e).__name__}: {_e})")

                # ── 外部提案幸存者 (复检): mf_accel 来自已算好的资金流扩展列 ──────────────
                # 注: tail_squeeze/vp_exhaustion/price_volume_density/accumulation 等 4 个是
                #     既有因子的数学恒等重复 (clv_ma5/pv_divergence/smart_proxy/buy_force),
                #     hidden_accum/crowding 是已证伪信号/构造即snooping — 不再重测, 详见上轮审查。
                try:
                    from moneyflow_factors import moneyflow_panel as _mfp_fn
                    _mfp = _mfp_fn(CONFIG.get('moneyflow_path',
                                   './tushare_cache/_partial/moneyflow'))
                    if _mfp is not None and 'mf_accel' in _mfp.columns:
                        _pdt = df.index.get_level_values('date').dtype
                        if _mfp.index.get_level_values('date').dtype != _pdt:
                            _mfp.index = _mfp.index.set_levels(
                                _mfp.index.levels[0].astype(_pdt), level='date')
                        df['lab_mf_accel'] = _mfp['mf_accel'].reindex(df.index).astype('float32')
                        del _mfp
                except Exception as _e:
                    print(f"   ...⚠️ mf_accel 观察舱 SKIPPED ({type(_e).__name__}: {_e})")

                _labs = [c for c in df.columns if c.startswith('lab_')]
                print(f"   ...Factor Lab built (observe-only, 零阵容影响): {_labs}")

            # ── Cleanup ──────────────────────────────────────────────
            for tmp_col in ['amplitude', 'pct_chg', 'illiq',
                            'is_limit_down', 'is_big_cap',
                            'mf_strength_8', 'mf_accel', 'sm_outflow_rate', 'retail_contrary',
                            'big_buy_strength', 'small_sell_ratio']:
                if tmp_col in df.columns: del df[tmp_col]
            gc.collect()

            # ── Cross-sectional rank normalization ───────────────────

            MARKET_LEVEL_FACTORS = {'mkt_breadth'}

            for f in self.factors:
                if f not in df.columns:
                    continue
                if f in MARKET_LEVEL_FACTORS:
        #           # Time-series z-score across dates (preserves regime signal)
                    daily_vals = df.groupby('date')[f].first()
                    ts_mean = daily_vals.rolling(60, min_periods=10).mean().shift(1)
                    ts_std  = daily_vals.rolling(60, min_periods=10).std().shift(1)
                    ts_z    = ((daily_vals - ts_mean) / (ts_std + 1e-9)).clip(-3, 3)
                    df[f]   = df.index.get_level_values('date').map(ts_z).astype('float32')
                else:
        #           # Normal cross-sectional rank for all other factors
                    df[f] = df.groupby('date')[f].rank(pct=True).astype('float32')
            # ── Market volume ratio ──────────────────────────────────
            daily_amount       = df.groupby('date')['amount'].sum().sort_index()
            amount_ma20        = daily_amount.rolling(20).mean()
            market_vol_ratio_z = daily_amount / amount_ma20
            mean_mvr           = market_vol_ratio_z.rolling(60).mean().shift(1)
            std_mvr            = market_vol_ratio_z.rolling(60).std().shift(1)
            market_vol_ratio_s = ((market_vol_ratio_z - mean_mvr)
                                  / (std_mvr + 1e-9)).clip(-3, 3).fillna(0)
            market_vol_ratio_s = market_vol_ratio_s[
                ~market_vol_ratio_s.index.duplicated(keep='first')]
            market_vol_ratio_s.name = 'market_vol_ratio'
            df['market_vol_ratio'] = (df.index.get_level_values('date')
                                      .map(market_vol_ratio_s)
                                      .astype('float32'))
            del daily_amount, amount_ma20, market_vol_ratio_z, market_vol_ratio_s
            gc.collect()

            # moneyflow factors are intentionally excluded from this dropna (see the join
            # block above): keep price rows that lack moneyflow coverage rather than dropping
            # the whole row. Their mf_* values stay NaN (clean IC + native LGBM handling).
            # fund_added likewise — 亏损股 pe_ttm=NaN 绝不能因 dropna 掉出 universe。
            _core_feats = [f for f in self.factors if f not in mf_added and f not in fund_added]
            df = df.dropna(subset=['adv'] + _core_feats)

        print(f"  > Total Samples (Inc. Prediction): {len(df):,}")
        df = df[df['adv']   > 30e6]
        df = df[df['close_raw'] > 4.0]   # 复权迁移: 绝对价过滤用原始价
        return df, self.factors
