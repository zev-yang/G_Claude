"""
factors.py — feature engineering (AlphaLabV25_1):
~45 cross-sectional factors + forward-return label.
"""
import gc
import numpy as np
import pandas as pd
import talib

from config import CONFIG, Timer


class AlphaLabV25_1:
    def __init__(self):
        self.factors = []

    def run(self, panel):
        print("\n📋 [2] Feature Engineering V3 (Regime-Conditional Beta)...")

        panel = panel[~panel['name'].str.contains(r'ST|\*ST|退', na=False)]
        panel = panel[panel['close'] > 4.0]

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
            df['smart_proxy']  = df['close'] / vwap

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
            df['amp_ma_10'] = amp_series.groupby('code').transform(lambda x: x.rolling(10).mean())
            df['amp_ma_20'] = amp_series.groupby('code').transform(lambda x: x.rolling(20).mean())
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

            df['near_high']  = (df['close'] / (
                g['close'].transform(lambda x: x.rolling(20).max()) + 1e-9)).astype('float32')
            df['mom_10']     = g['close'].pct_change(10).astype('float32')
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
            vol_stb  = g['volume'].transform(lambda x: x.rolling(5).std())
            vol_mnb  = g['volume'].transform(lambda x: x.rolling(5).mean())
            df['vol_stability'] = vol_mnb / (vol_stb + 1e-9)

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

            # ── NEW: Alpha Momentum (beta-stripped) ──────────────────
            mkt_ret10         = df.groupby('date')['ret_10'].transform('median')
            df['alpha_mom10'] = (df['ret_10'] - mkt_ret10).astype('float32')

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
                'alpha_mom10',        # market-beta-stripped 10d return
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
                'amp_ma_5', 'amp_ma_10', 'amp_ma_20',
                # ── Market / Other ──────────────────────────────────────
                'vol_z', 'mom_10', 'near_high', 'pv_divergence', 'vol_stability',
                # ── Accumulation / Distribution (主力 footprints) ────────
                'downside_rs',   # holds up on down-market days (support / 抗跌)
                'accum_trend',   # 20d up-volume share, rising = accumulation
                'coil',          # recent vs longer-term range compression (蓄势)
                # Removed: 'dist_risk' (ICIR 0.028 < floor, redundant +0.77 vol_ratio), 'streak' (ICIR 0.074), 'open_strength' raw (ICIR 0.010)
            ]
            # NOTE: 'streak' removed (ICIR=0.074 — below noise floor)
            # NOTE: 'open_strength' (raw) removed (ICIR=0.010 — pure noise)

            # ── Cleanup ──────────────────────────────────────────────
            for tmp_col in ['amplitude', 'pct_chg', 'illiq',
                            'is_limit_down', 'is_big_cap']:
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

            df = df.dropna(subset=['adv'] + self.factors)

        print(f"  > Total Samples (Inc. Prediction): {len(df):,}")
        df = df[df['adv']   > 30e6]
        df = df[df['close'] > 4.0]
        return df, self.factors
