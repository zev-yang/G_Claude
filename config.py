"""
config.py — shared configuration, constants, and small helpers.
Imported by every other module. Edit thresholds/toggles here in one place.
"""
import os
import time
import warnings
import numpy as np
from datetime import datetime, timedelta

# Preserve original runtime setup: raise Jupyter's print-rate ceiling (the run is
# very print-heavy) and silence pandas/numpy warnings. Set on import of config.
os.environ['JUPYTER_IOPUB_DATA_RATE_LIMIT'] = '10000000'
warnings.filterwarnings('ignore')

# ===================== Dynamic config (Auto) =====================
auto_start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"

CONFIG = {
    "stock_data_path": "./stock_data_all/*.csv", 
    "results_dir": "./results_v25_1_production",
    
    "audit_start": auto_start,  
    "train_window": 300,
    "horizon": 8,
    "refit_freq": 1,
    
    "top_k": 30,                
    "target_vol": 0.15,
    "risk_aversion": 2.5,
    "max_weight": 0.08,
    "cost_bps": 0.0015,
    "adv_limit": 0.06,
    "icir_threshold": 0.15,  # === MODIFIED: 行业内较为稳健的初选标准 ===
    "icir_window": 60,          # === MODIFIED: 专门用于筛选特征的中短期窗口 ===
    # ===================== UPGRADE KEYS =====================
    "max_history_days": 1000,    # was effectively 400 (train_window+100); raise for a real backtest
    "enable_hedge": True,        # False -> no safety filter (measure raw alpha)
    "enable_logic_fusion": True, # False -> pure LGBM ranking (no LogicMatrix tilt)
    "logic_tilt": 0.3,           # weight of the logic signal in the z-blend
    "embargo": 0,                # extra purge days beyond horizon (0 = match production exactly)
}


def _zscore(x):
    """Cross-sectional z-score; puts LGBM and logic signals on the same scale before blending."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


class Timer:
    def __init__(self, name): self.name = name
    def __enter__(self): self.start = time.time(); print(f"⏳ [{self.name}] Processing...")
    def __exit__(self, *args): print(f"   ...Done ({time.time()-self.start:.2f}s)")


# ===================== Shared selection constants =====================
# FAMILY_MAP and GROUPS_V3 were previously defined inside __main__ / run_simulation.
# Centralised here so the backtest and production path share ONE definition
# (the two GROUPS_V3 copies were verified identical as sets).
FAMILY_MAP = {
    # Stability
    'turnover_cv':   'StabFam', 'turnover_cv10': 'StabFam',
    'turnover_cv5':  'StabFam', 'vol_stability': 'StabFam',
    # Raw momentum
    'ret_10': 'MomFam', 'ret_20': 'MomFam', 'mom_acc': 'MomFam',
    'mom_10': 'MomFam', 'near_high': 'MomFam', 'dist_high': 'MomFam',
    # Alpha momentum (separate family — different signal source)
    'alpha_mom10': 'AlphaMomFam',
    # Sector relative strength
    'sector_rs_5': 'SecRSFam', 'sector_rs_10': 'SecRSFam',
    # Gap [NEW]
    'gap': 'GapFam', 'gap_ma3': 'GapFam',
    # Volume ratio
    'vol_ratio': 'VolFam', 'vol_ratio5': 'VolFam', 'vol_ratio10': 'VolFam',
    # Volatility std
    'vol_5': 'StdFam', 'vol_10': 'StdFam', 'vol_20': 'StdFam',
    # Amplitude
    'amp_ma_5': 'AmpFam', 'amp_ma_10': 'AmpFam', 'amp_ma_20': 'AmpFam',
    # Skew
    'skew_5': 'SkewFam', 'skew_10': 'SkewFam', 'skew_20': 'SkewFam',
    # Intraday (smoothed only)
    'open_strength_ma5': 'OpenFam',
    # TA-Lib (each is its own family)
    'macd_hist': 'MACDFam', 'atr_ratio': 'ATRFam', 'bb_position': 'BBFam',
    # Beta (regime-adjusted — single factor, own family)
    'beta_60d': 'BetaFam',
    # Volume-price flow
    'vpt': 'VPTFam', 'buy_force': 'BuyForceFam',
    # Sector heat [NEW] (separate family from sector_rs)
    'sector_heat':    'SectorHeatFam',
    'sector_support': 'SectorSupportFam',
}

GROUPS_V3 = {
    'Momentum': [
        'near_high', 'mom_10', 'mom_acc', 'ret_10', 'ret_20',
        'bias_20', 'dist_high', 'macd_hist', 'bb_position',
        'alpha_mom10', 'sector_rs_5', 'sector_rs_10',
        'gap',          # opening gap belongs with momentum
        'gap_ma3',
    ],
    'Volume': [
        'vol_z', 'buy_force', 'vol_ratio', 'vol_ratio5', 'vol_ratio10',
        'turnover_cv', 'turnover_cv10', 'turnover_cv5',
        'illiq_ma10', 'vol_stability',
        'open_strength_ma5',  # smoothed intraday pressure
        'vpt',                # directional volume-price trend
        'sector_heat',    # sector-level volume × trend activity
        'sector_support', # BUG7 FIX: moved from Reversion — % peers rising is MOMENTUM
    ],
    'Reversion': [
        'reversal', 'rsi', 'smart_proxy', 'pv_divergence',
        'beta_60d',           # regime-adjusted: low beta = good in bear
        'clv_ma5',     # close location → intraday smart money direction
    ],
    'Stability': [
        'pv_corr', 'vol_10', 'vol_20', 'vol_5',
        'skew_10', 'skew_20', 'amp_ma_5', 'amp_ma_10', 'amp_ma_20',
        'atr_ratio','skew_5',
    ],
}
