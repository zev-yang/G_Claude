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
    "moneyflow_path": "./tushare_cache/_partial/moneyflow",  # Tushare basic `moneyflow` per-date parquet partials (folder), or a single combined .parquet
    "moneyflow_role": "screen",              # 'screen' = negative screen (NOT a model feature) | 'factor' = model feature | 'off' = 48-factor baseline
    "moneyflow_screen_cols": ["elg_cum20"],  # ranked moneyflow col(s) the screen uses: ["elg_cum20"] (sharpest reversal) | ["mf_cum20"] | ["elg_cum20","mf_cum20"]
    "moneyflow_screen_pool": 50,             # take top-N by model score, drop over-accumulated, then diversify to top_k
    "moneyflow_screen_pct": 0.90,            # drop names above this cross-sectional 主力-accumulation rank (top decile)
    # ════════ Layer-4 资金流 overlay (线性叠加, 不喂 LGBM) ════════════════════════════
    # 全部写死、point-in-time、A/B 只作 go/no-go 观察 (绝不在全样本上调 MF_WEIGHT 等参数)。
    "USE_MONEYFLOW": True,        # 总开关; False=不叠加 (与现有 screen 独立)
    "MF_WEIGHT": 0.15,           # 写死: final = zscore(base) + MF_WEIGHT·mf_score
    "MF_W_STRENGTH": 0.6,        # 写死: mf_score 内 strength 权重
    "MF_W_ACCEL": 0.2,           # 写死: accel 权重 (mf_accel = strength_8 - strength_20)
    "MF_W_RETAIL": 0.2,          # 写死: retail_contrary 权重
    "RETAIL_CONTRARY_ENABLE": True,
    "RETAIL_CONTRARY_PERCENTILE": 0.90,  # 写死: 小单净流出【当日横截面】前 10%
    "RETAIL_CONTRARY_VOL_RATIO": 1.2,    # 写死: 放量倍数 (当日量 / 20日均量)
    "RETAIL_CONTRARY_BIAS_LIMIT": 0.15,  # 写死: |close/ma20 - 1| 上限 (乖离约束)
    # ── DEFERRED (数据/择时未就绪, 留 False 以备后用) ───────────────────────────────
    "USE_MARKET_POSITION": False,  # Layer-1 大盘仓位 (需 moneyflow_dc; 归到"晚点升 V9")
    "MF_INDUSTRY_ENABLE": False,   # Layer-2 行业加分 (需 moneyflow_ind_dc + 个股→行业映射)
    "MF_CONCEPT_ENABLE": False,    # Layer-3 概念加分 (默认关)
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
    "enable_hedge": False,       # OFF permanently — hedge is value-destroying (CAGR ~5%, worse DD).
                                 # diagnostics force-flips per combo; run.py uses this default.
    "enable_logic_fusion": True, # False -> pure LGBM ranking (no LogicMatrix tilt)
    "logic_tilt": 0.3,           # weight of the logic signal in the z-blend
    "embargo": 0,                # extra purge days beyond horizon (0 = match production exactly)
    # ============ ACCURACY EXPERIMENT TOGGLES (compare each on/off) ============
    "residualize_label": True,   # target = rank of size-decile-residual fwd return (alpha), not raw
    "use_regime_features": True, # append market-state features so the model can adapt by regime
    "use_ranker": True,          # LGBMRanker (lambdarank, date=group) instead of LGBMRegressor
}

# Market-state context features appended to the model's input (NOT ranked factors).
# Built in factors.py; constant within a day, so the GBDT uses them to condition
# stock-factor splits on the regime rather than to rank within a day.
REGIME_FEATURES = ['mkt_breadth', 'cs_vol_ma5', 'mkt_trend_20', 'market_vol_ratio']


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
    # Accumulation / Distribution (主力 footprints) — each its own family
    'downside_rs':  'DownsideFam',
    'accum_trend':  'AccumFam',
    'coil':         'CoilFam',
    # 主力 (Tushare moneyflow) order-flow factors — distinct families; the corr-cull guards overlap
    'mf_cum20':     'MFCumFam',
    'mf_trend':     'MFTrendFam',
    'elg_cum20':    'MFElgFam',
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
        'accum_trend',    # rising 20d up-volume share (accumulation)
        'mf_cum20',       # 主力 net inflow, 20d cumulative (Tushare moneyflow)
        'mf_trend',       # 主力 inflow acceleration (recent vs longer pace)
        'elg_cum20',      # 超大单 net inflow, 20d cumulative
    ],
    'Reversion': [
        'reversal', 'rsi', 'smart_proxy', 'pv_divergence',
        'beta_60d',           # regime-adjusted: low beta = good in bear
        'clv_ma5',     # close location → intraday smart money direction
        'downside_rs', # holds up on down-market days (support / 抗跌)
    ],
    'Stability': [
        'pv_corr', 'vol_10', 'vol_20', 'vol_5',
        'skew_10', 'skew_20', 'amp_ma_5', 'amp_ma_10', 'amp_ma_20',
        'atr_ratio','skew_5',
        'coil',        # range compression (蓄势 / coiling)
    ],
}
