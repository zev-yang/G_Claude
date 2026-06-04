"""
data_loader.py — data-lake loading and panel assembly.
"""
import glob
import pandas as pd
import joblib
from tqdm import tqdm

from config import CONFIG


def load_single_robust(file_path):
    try:
        header = pd.read_csv(file_path, nrows=0)
        clean_cols = [c.strip().lower().replace('\ufeff', '').replace(' ', '') for c in header.columns]
        col_map = dict(zip(header.columns, clean_cols))
        
        target_map = {
            'code': ['code', 'ts_code', 'symbol', 'ticker'],
            'date': ['date', 'trade_date', 'datetime'],
            'open': ['open'], 'close': ['close'], 'high': ['high'], 'low': ['low'],
            'volume': ['volume', 'vol'],
            'amount': ['amount', 'amt', 'turnover'],
            'name': ['name', 'stock_name']
        }
        
        rename_dict = {}
        for raw, clean in col_map.items():
            for std, alts in target_map.items():
                if clean in alts:
                    rename_dict[raw] = std
                    break
        
        if not {'code', 'date', 'close', 'volume'}.issubset(rename_dict.values()): return None

        df = pd.read_csv(file_path, usecols=rename_dict.keys())
        df.rename(columns=rename_dict, inplace=True)
        
        val = str(df['code'].iloc[0])
        if '.' in val: val = val.split('.')[0]
        df['code'] = val.zfill(6)
        df['date'] = pd.to_datetime(df['date'])
        
        if 'name' not in df.columns: df['name'] = df['code']
        else: df['name'] = df['name'].astype(str)
        
        for c in ['open','high','low','close','volume','amount']:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce').astype('float32')
        
        df.dropna(subset=['close', 'date'], inplace=True)
        df.sort_values('date', inplace=True)
        
        if 'amount' not in df.columns:
            df['amount'] = df['close'] * df['volume'] * 100.0
            
        # UPGRADE Edit 1: keep enough history for a real backtest (was train_window + 100 = 400)
        req_len = max(CONFIG['max_history_days'], CONFIG['train_window'] + CONFIG['horizon'] + 50)
        if len(df) > req_len: df = df.iloc[-req_len:]
            
        return df
    except: return None


def load_universe_audit(path):
    print("\n📋 [1] Loading Data Lake...")
    files = glob.glob(path)
    if not files: raise ValueError(f"No files in {path}")
    
    dfs = joblib.Parallel(n_jobs=-1)(joblib.delayed(load_single_robust)(f) for f in tqdm(files))
    valid_dfs = [d for d in dfs if d is not None and len(d) > 200]
    
    if not valid_dfs: raise ValueError("Data Empty")
    
    panel = pd.concat(valid_dfs)
    panel.drop_duplicates(subset=['date', 'code'], inplace=True, keep='last')
    panel = panel.set_index(['date', 'code']).sort_index()
    print(f"  > Loaded: {len(panel):,} rows")
    return panel
# ==========================================================
# [新增] 🚀 通用混合战术避险系统 (Universal Tactical Hedge)
# ==========================================================
# ==========================================================
# [UPGRADE V8] 🚀 最终集成版：全天候自适应避险系统
# 深度整合 V6.1 的时序逻辑：波动率持续期过滤、两日反转确认、严苛信号压制
# 目标：解决 A 股“单日诱多”与“熔断式阴跌”风险
# ==========================================================
