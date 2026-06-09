"""
run.py — entry point. Orchestrates: load -> features -> backtest ->
production next-trade prediction. Run with:  python run.py
"""
import os
import sys
import gc
import traceback
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CONFIG, FAMILY_MAP, GROUPS_V3, RED, GREEN, RESET, _zscore, REGIME_FEATURES
from data_loader import load_universe_audit
from factors import AlphaLabV25_1
from safety import check_market_safety_v9
from logic_matrix import LogicMatrixPredictorV5, IntegratedAuditorV5
from portfolio import build_sector_clusters, diversify_picks, moneyflow_negative_screen
from backtest import DailyAuditor
from modeling import build_model, fit_model


def deep_clean_memory():
    import gc
    import pandas as pd
    print("\n🧹 [Memory Cleanup] Starting deep cleaning...")
    
    # 1. Targeted deletion of G35 variables
    targets = [
        'panel', 'df', 'train_full', 'tr_data', 'recent_data', 
        'macro_history', 'macro_history_prod', 'recent_macro_prod',
        'today_df', 'best', 'fused_scores', 'prod_fused', 'res_df', 
        'model', 'model_final', 'ic_eval_data', 'prod_ic_df'
    ]
    
    for t in targets:
        if t in globals():
            del globals()[t]

    # 2. Dynamic scanning with safe memory usage check
    for var_name in list(globals().keys()):
        if var_name.startswith('_') or var_name in ['CONFIG', 'RED', 'GREEN', 'RESET']:
            continue
            
        try:
            obj = globals()[var_name]
            if isinstance(obj, pd.DataFrame):
                usage = obj.memory_usage(deep=True).sum()
            elif isinstance(obj, pd.Series):
                usage = obj.memory_usage(deep=True) # Series usage is just an int
            else:
                continue
                
            if usage > 1024 * 1024: # 1MB
                del globals()[var_name]
        except:
            continue

    # 3. Double Garbage Collection
    gc.collect()
    gc.collect()
    print("✨ [Memory Cleanup] Completed. System Ready.")


if __name__ == "__main__":
    print(f"🚀 v25.1 Live Unlock | Forecast Fixed\n")
        # FAMILY_MAP is imported from config.
    try:
        # 1. Load
        panel = load_universe_audit(CONFIG['stock_data_path'])
        
        # 2. Features
        eng = AlphaLabV25_1()
        panel, feats = eng.run(panel)
        
        # 3. Audit
        auditor = DailyAuditor(panel, feats)
        auditor.check_ic()
        res_df, model = auditor.run_simulation()
        auditor.analyze(res_df)
             
# ... (Step 5 之后) ...
      # 4. 真实预测模块 (Real-World Forecast)
        print("\n🔮 [Step 6] PREDICT NEXT TRADE (Production Mode)...")
    
        # 【核心防御修复】：处理多重索引切片前，必须进行全局排序以防止 UnsortedIndexError
        panel = panel.sort_index()
        
        # 1. 先获取所有的交易日期，并确定“今天”（需要出信号的日子）
        all_dates_list = panel.index.get_level_values('date').unique().sort_values().tolist()
        last_real_date = all_dates_list[-1]  
        curr_idx = len(all_dates_list) - 1   # “今天”的索引位置
        
        # 2. 强制倒推 horizon + 1 天，彻底隔绝未来函数和未定型的 Target
        train_cutoff_idx = curr_idx - CONFIG['horizon'] - 1
        if train_cutoff_idx < 0:
            raise ValueError(f"数据量过少，无法支持 {CONFIG['horizon']} 天的 horizon 切片。")
        train_cutoff_date = all_dates_list[train_cutoff_idx]
        
        print(f"  > [Data Cutoff] Today: {last_real_date.date()} | Training Cutoff: {train_cutoff_date.date()}")
        
        # 3. 严格按日期截断后，再丢弃缺失值
        idx = pd.IndexSlice
        train_full = panel.loc[idx[:train_cutoff_date, :], :].dropna(subset=['target'])
        
        # last_date 更新为截断后的最后一天（用于后续统计，不再与今天混淆）
        last_train_date = train_full.index.get_level_values('date').max()
        
        # 取最近的一段窗口计算 ICIR
        eval_lookback = CONFIG['icir_window']
        recent_dates = train_full.index.get_level_values('date').unique().sort_values()[-eval_lookback:]
        recent_data = train_full.loc[pd.IndexSlice[recent_dates, :], :]

        print(f"  > Calculating ICIR for screening (Lookback: {len(recent_dates)} days)...")
        prod_daily_ics = []
        for d, group in recent_data.groupby(level='date'):
            res = {f: group[f].corr(group['target'], method='spearman') for f in feats}
            prod_daily_ics.append(res)
        
        prod_ic_df = pd.DataFrame(prod_daily_ics)
        
        # 构建统计表
        prod_stats = []
        for f in feats:
            m_ic = prod_ic_df[f].mean()
            s_ic = prod_ic_df[f].std()
            icir = m_ic / (s_ic + 1e-9)
            prod_stats.append({'Factor': f, 'MeanIC': m_ic, 'ICIR': icir, 'AbsICIR': abs(icir)})
        
        prod_stats_df = pd.DataFrame(prod_stats)

        # B. [核心修复]：候选筛选 + 相关性剔除
        # 1. 初选：绝对值 ICIR > 阈值
 # --- Production Elite Selection ---
        # 1. Sort by ICIR strength
        prod_stats_df['AbsICIR'] = prod_stats_df['ICIR'].abs()
        sorted_stats = prod_stats_df.sort_values('AbsICIR', ascending=False)
        candidate_list = sorted_stats['Factor'].tolist()
        
        # --- 操盘手版：结构化动态筛选 ---
# --- 操盘手版：全因子覆盖战队 (确保每一个注册因子都有出头之日) ---
        # GROUPS_V3 is imported from config.
        
        final_selected_feats = []
        
# === [Location] 优中选优：全局排序 + 组内上限 + 相关性去重 ===
        # 1. 提取当下量能状态
        last_real_date = panel.index.get_level_values('date').max()
        # 确保已提取 macro_history_prod
        macro_cols = ['mkt_breadth', 'cs_vol_ma5', 'market_vol_ratio', 'limit_down_count', 'mkt_low_level', 'brd_300', 'brd_1000']
        macro_history_prod = panel[macro_cols].groupby(level=0).first().sort_index()
        
        today_data_prod = macro_history_prod.loc[last_real_date]
        recent_macro_prod = macro_history_prod.loc[:last_real_date].tail(60)
        
        # 计算量能增量
        curr_mvr = today_data_prod['market_vol_ratio']
        prev_mvr_avg = recent_macro_prod['market_vol_ratio'].tail(3).mean()
        mvr_delta = curr_mvr - prev_mvr_avg
        
        # 2. 设置生产环境名额限制
        # 默认配置
        group_limits = {'Momentum': 4, 'Volume': 4, 'Reversion': 2, 'Stability': 2}
        
        if curr_mvr > 0.8 or mvr_delta > 0.5:
            group_limits = {'Momentum': 6, 'Volume': 4, 'Reversion': 2, 'Stability': 0}
            print(f"🔥 [生产-进攻模式] 市场放量/爆发 (MVR:{curr_mvr:.2f}, Delta:{mvr_delta:.2f})")
        elif curr_mvr < -1.0:
            group_limits = {'Momentum': 1, 'Volume': 1, 'Reversion': 4, 'Stability': 6}
            print(f"🛡️ [生产-防御模式] 市场极度缩量 (MVR:{curr_mvr:.2f})")
        else:
            print(f"⚖️ [生产-平衡模式] 市场量能平稳 (MVR:{curr_mvr:.2f})") 
        #   # FIX: Bear market breadth override
#   # When breadth is very low, force defensive posture regardless of volume
#   # This is the key fix that prevents v2's high-beta tech stock problem

       # 3. 执行动态因子选拔
        MAX_TOTAL_FEATS = 12
        CORR_THRESHOLD = 0.62
        final_selected_feats = []
        group_counts = {g: 0 for g in GROUPS_V3.keys()}
        selected_families = set() # <--- 新增：用于追踪已选中的家族
        # 按 AbsICIR 排序开始挑选
        sorted_prod_candidates = prod_stats_df.sort_values('AbsICIR', ascending=False)
        #   # FIX 2: hard ICIR floor — discard near-zero factors before selection
        #   # open_strength (0.010) and streak (0.074) were consuming slots in v2
        MIN_ICIR_FLOOR = 0.08
        sorted_prod_candidates = sorted_prod_candidates[
            sorted_prod_candidates['AbsICIR'] >= MIN_ICIR_FLOOR
        ]
        if len(sorted_prod_candidates) < 4:
            print(f"⚠️ Only {len(sorted_prod_candidates)} factors above floor — check data quality")
            
        for _, row in sorted_prod_candidates.iterrows():
            f = row['Factor']
            if len(final_selected_feats) >= MAX_TOTAL_FEATS: break
            # 1. 家族去重逻辑
            f_family = FAMILY_MAP.get(f, f)
            if f_family in selected_families:
                continue # 确保 StaFam 或 MomFam 只出一个代表

            # 2. 战队配额检查
            belong_to = next((gn for gn, f_list in GROUPS_V3.items() if f in f_list), None)
            if not belong_to or group_counts[belong_to] >= group_limits[belong_to]:
                continue
            
            # 3. 相关性检查 (此时建议 CORR_THRESHOLD 设为 0.60 左右)
            is_redundant = False
            if final_selected_feats:
                corrs = recent_data[final_selected_feats].corrwith(recent_data[f]).abs()
                if corrs.max() > CORR_THRESHOLD:
                    is_redundant = True
                    
            if not is_redundant:
                final_selected_feats.append(f)
                group_counts[belong_to] += 1
                selected_families.add(f_family) # <--- 锁定该家族
                
        
        print(f"🏆 [生产环境因子确定] 选入: {len(final_selected_feats)} 个因子，构成: {group_counts}")
        for g_name, count in group_counts.items():
            if count > 0:
                print(f"   > {g_name:10}: {count} 个因子")
            
        # 此时 final_selected_feats 应该有 8 个因子，且维度极其平衡
        #print(f"🏆 结构化选拔结果 (平衡维度): {final_selected_feats}")

        # C. 输出最终选择的因子和 ICIR 值
        #print("\n🏆 [Final Feature Selection Results]")
        #print("-" * 50)
        #final_display = prod_stats_df[prod_stats_df['Factor'].isin(final_selected_feats)].sort_values('AbsICIR', ascending=False)
        #print(final_display[['Factor', 'MeanIC', 'ICIR']].to_string(index=False))
        #print("-" * 50)
        #print(f"Total Features Selected: {len(final_selected_feats)}")

         # 1. 从 win_stats 提取入选因子的原始 ICIR（带正负号）
            # 注意：我们在 stats_list 里需要保存原始 ICIR 而不仅仅是 AbsICIR
        selected_info = []
        for f in final_selected_feats:
            # 从 win_stats 找到对应的 ICIR 值
            # 假设 win_stats 包含 'Factor' 和 'ICIR' 列
            f_row = prod_stats_df[prod_stats_df['Factor'] == f].iloc[0]
            val_icir = f_row.get('ICIR', f_row.get('AbsICIR', 0)) # 兼容性处理
            selected_info.append(f"{f}({val_icir:.3f})")
            
        # 2. 格式化输出字符串
        factor_detail_str = " | ".join(selected_info)
        summary_dist = ", ".join([f"{k}:{v}" for k, v in group_counts.items() if v > 0])
         # E. 最终预测下一交易日
        last_real_date = panel.index.get_level_values('date').max()    
        # 3. 使用 tqdm.write 一次性输出，防止 IOPub 报错
        tqdm.write("=" * 90)
        tqdm.write(f"📅 信号日期: {last_real_date.date()} | 因子总数: {len(final_selected_feats)} | 战队构成: [{summary_dist}]")
        tqdm.write(f"🚀 因子详情: {factor_detail_str}")
        tqdm.write("=" * 90)

        # D. 全量重训 (使用最终确定的 final_selected_feats)
        print(f"\n  > Retraining Model with {len(final_selected_feats)} features...")
# 🔮 [Step 6] 生产重训模型
        # UPGRADE: model from CONFIG['use_ranker'] (LGBMRanker vs LGBMRegressor).
        model_final = build_model()
        
        # 使用全量历史数据进行最后一次拟合
        # B. 全量重训 (Using Exponential Weighting)
        print("  > Retraining Model with Exponential History Weighting...")
        
        # Use geomspace: gives exponentially more weight to recent data
        # This helps the model adapt to the current market style faster
        w_final = np.geomspace(0.1, 1.0, len(train_full)) 
        # UPGRADE: append regime features (market-state context) to the model input.
        prod_feats = final_selected_feats + (REGIME_FEATURES if CONFIG['use_regime_features'] else [])
        fit_model(model_final, train_full[prod_feats], train_full['target'], w_final,
                  train_full.index.get_level_values('date').values)

        
        # E. 最终预测下一交易日
        #last_real_date = panel.index.get_level_values('date').max()
        today_df = panel.loc[last_real_date].copy()
        
        # 过滤流动性
        today_df = today_df[(today_df['close'] > 3.0) & (today_df['amount'] > 15e6)] # 15 million minimum daily turnover
        
        
        if len(today_df) > 0:
            # === MODIFIED: 预测时使用 prod_feats (factors + regime features) ===
            today_df['score'] = model_final.predict(today_df[prod_feats])
            # ... (后续逻辑) ...
 # ==========================================================
 # === [MODIFIED: Production 实盘统一调用避险模块] ===
# === [FIXED: Production 实盘统一调用避险模块] ===
            print(f"\n🛡 Checking Market Safety for {last_real_date}...")
        
# --- 在 Step 6 生产模式，判断 last_real_date 的逻辑中修改 ---

            # 1. 重新提取包含新指标的宏观序列
            macro_history_prod = panel[['mkt_breadth', 'cs_vol_ma5', 'market_vol_ratio', 
                                        'limit_down_count', 'mkt_low_level', 'brd_300', 'brd_1000']].groupby(level=0).first().sort_index()
            
            # 2. 准确定位当日和昨日
            recent_macro_prod = macro_history_prod.loc[:last_real_date].tail(60)
            today_row_prod = macro_history_prod.loc[last_real_date]
            
            # 获取昨日索引（避开非交易日干扰）
            all_macro_dates = macro_history_prod.index.tolist()
            curr_idx = all_macro_dates.index(last_real_date)
            prev_date_prod = all_macro_dates[curr_idx - 1] if curr_idx > 0 else last_real_date
            prev_row_prod = macro_history_prod.loc[prev_date_prod]
            
            # 3. 构建 Production 字典
            prod_today_dict = {
                'mkt_breadth': today_row_prod['mkt_breadth'],
                'market_vol_ratio': today_row_prod['market_vol_ratio'],
                'cs_vol_ma5': today_row_prod['cs_vol_ma5'],
                'limit_down_count': today_row_prod['limit_down_count'],
                'low_price': today_row_prod['mkt_low_level'],
                'prev_low': prev_row_prod['mkt_low_level'],
                'breadth_300': today_row_prod['brd_300'],
                'breadth_1000': today_row_prod['brd_1000']
            }
            
            # 4. 调用检测
            avg_icir_prod = prod_stats_df[prod_stats_df['Factor'].isin(final_selected_feats)]['AbsICIR'].mean()
            # UPGRADE Edit 5a: production hedge respects the same toggle. Keep enable_hedge=True
            # for live signals; flip to False only when studying the backtest's raw alpha.
            if CONFIG['enable_hedge']:
                skip_reason, mkt_str = check_market_safety_v9(prod_today_dict, recent_macro_prod, avg_icir_prod)
            else:
                skip_reason, mkt_str = None, "🟢 Hedge OFF"
            
            # --- 拦截执行 ---
            if skip_reason:
                print(f"🛑 [SIGNAL BLOCKED] {last_real_date.date()} -> 原因: {skip_reason}")
                print("⚠ 今日建议：强制空仓 (Wait on Cash)")
                # 这里实盘风控阻断，直接不进行文件保存或记录空仓文件
                import sys
                sys.exit(0)  # <--- 直接加这两行，完美解决，后面的代码完全不用动！
            else:
                print(f"✅ {last_real_date.date()} Market Environment: SAFE ({mkt_str})")
        # =======================================================
        # ...接后续文件保存代码
                # --- 原有逻辑：输出信号 ---
                # Output Top 10
            # --- 生产预测融合 (UPGRADE Edit 5b: same-scale z-blend, matches the backtest) ---
            lgbm_prod = today_df['score'].values   # raw LGBM score set just above
            if CONFIG['enable_logic_fusion']:
                logic_engine = LogicMatrixPredictorV5()
                logic_prod = np.array([
                    logic_engine.predict_diagnostics(row)['total_score']
                    for _, row in today_df.iterrows()
                ])
                fused_prod = _zscore(lgbm_prod) + CONFIG['logic_tilt'] * _zscore(logic_prod)
            else:
                fused_prod = lgbm_prod

            cluster_map_prod = build_sector_clusters(
                panel, last_real_date, lookback_days=60, corr_threshold=0.55
            )
            print(f"  Sector clusters built: {len(set(cluster_map_prod.values()))} clusters "
                f"for {len(cluster_map_prod)} stocks")
            today_df['score'] = fused_prod # 覆盖为融合分数
            # ── Layer-4 资金流 overlay (与回测一致): final = zscore(base) + MF_WEIGHT·mf_score ──
            if CONFIG.get('USE_MONEYFLOW', False) and 'mf_score' in today_df.columns:
                today_df['score'] = (_zscore(today_df['score'])
                                     + CONFIG.get('MF_WEIGHT', 0.15) * today_df['mf_score'].fillna(0.0).values)
            #today_breadth_prod = prod_today_dict.get('mkt_breadth', 0.5)
            # NEGATIVE SCREEN: drop over-accumulated names from the top pool before diversifying
            if CONFIG.get('moneyflow_role', 'screen') == 'screen':
                _pool_n = CONFIG.get('moneyflow_screen_pool', 50)
                _pool = moneyflow_negative_screen(
                    today_df, 'score',
                    CONFIG.get('moneyflow_screen_cols', ['elg_cum20']),
                    pool_n=_pool_n, pct=CONFIG.get('moneyflow_screen_pct', 0.90))
                print(f"  🧹 Moneyflow screen ({CONFIG.get('moneyflow_screen_cols', ['elg_cum20'])}): "
                      f"kept {len(_pool)} of top-{min(len(today_df), _pool_n)} (dropped over-accumulated)")
            else:
                _pool = today_df
            best = diversify_picks(
                _pool,
                score_col    = 'score',
                top_k        = 30,
                max_per_cluster = 2,
                cluster_map     = cluster_map_prod,
            )
 # === [核心修复点]：确保信号日期字符串使用真正的最后一天 (last_real_date) ===
            signal_date_str = pd.to_datetime(last_real_date).strftime('%Y-%m-%d')
            
            print(f"\n🏆 Top Picks for Tomorrow ({len(today_df)} scanned):")
            print(f"Signal Date: {signal_date_str}") # 打印核对
            print("-" * 140) # 延长分割线以适应更长的输出
            for c, r in best.iterrows():
                # VolRatio 
                vr = r['vol_ratio'] if 'vol_ratio' in r else 0
                # Name handle
                n = str(r['name']) if 'name' in r else str(c)
                
                # === 新增：动态提取入选因子的值 ===
                # 注意：这里的因子值是经过 rank(pct=True) 归一化后的分位数 (0~1)
                factor_vals = [f"{f}({r.get(f, 0.0):.3f})" for f in final_selected_feats]
                factor_str = " | ".join(factor_vals)
                
                print(f"{c:<8} {n[:4]:<8} {r['close']:<8.2f} {r['score']:.4f}   {vr:.2f}  {r['market_vol_ratio']:.3f}  |  {factor_str}")
            print("-" * 140)
            #audit最终结果
            engine = LogicMatrixPredictorV5()
            auditor = IntegratedAuditorV5(engine)
            auditor.run_audit(best)
             # ==============================================================================
            # [新增功能] 保存预测结果到 CSV
            # ==============================================================================
            csv_file = f'selected_stocks_{CONFIG["horizon"]}.csv'
            
            # 修正点：使用 last_date 转换日期，而不是 today_df
            #signal_date_str = pd.to_datetime(last_date).strftime('%Y-%m-%d')
            
            # 1. 准备当前日期的新数据 (从 best 变量中获取)
            new_records = []
            for code, row in best.iterrows():
                record = {
                    'Signal Date': signal_date_str,
                    'code': str(code),  # 强制转字符串，保留000开头
                    'Name': str(row.get('name', code)),
                    'Close': float(f"{row['close']:.2f}"),              # 2位小数
                    'Score': float(f"{row.get('score', 0.0):.3f}"),      # 3位小数（带默认值）
                    'VolRatio': float(f"{row.get('vol_ratio', 0.0):.3f}"),
                    'market_vol_ratio': float(f"{row.get('market_vol_ratio', 0.0):.3f}")
                }
                
                # === 修改：将【全量因子库】及其对应的值加入字典 ===
                # 使用 feats 替代 final_selected_feats
                for f in feats:
                    val = row.get(f, np.nan)
                    # 增加 pd.notna 判断，防止 NaN 格式化时报错
                    record[f] = float(f"{val:.3f}") if pd.notna(val) else None
                    
                new_records.append(record)
            
            new_df = pd.DataFrame(new_records)
            
            # 确保列顺序整齐，将【全量因子列】追加在基础列之后
            cols = ['Signal Date', 'code', 'Name', 'Close', 'Score', 'VolRatio', 'market_vol_ratio'] + feats
            # 补齐可能缺失的列
            for c in cols:
                if c not in new_df.columns: new_df[c] = None
            new_df = new_df[cols]
            

            # 2. 处理文件合并 (保留历史，更新今日)
            if os.path.exists(csv_file):
                try:
                    # 读取旧文件
                    history_df = pd.read_csv(csv_file, dtype={'code': str})
                    
                    # 【核心逻辑】：删除文件中“Signal Date”等于今天的旧记录
                    history_df = history_df[history_df['Signal Date'] != signal_date_str]
                    
                    # 追加新数据
                    final_df = pd.concat([history_df, new_df], ignore_index=True)
                except Exception as e:
                    print(f"⚠️ CSV读取失败，将覆盖为新数据: {e}")
                    final_df = new_df
            else:
                # 文件不存在，直接创建
                final_df = new_df
            
            # 3. 写入文件
            final_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"✅ 已保存 {len(best)} 只选股结果至 {csv_file} (日期: {signal_date_str})")
            print("强化量价验证：在 Breadth 超过 0.6 时，如果 MktVol 是负数，必须进入“减仓模式”。")
        else:
            print("❌ No valid stocks found for today (All filtered).")
    except SystemExit:
        pass  # === 新增：识别到风控系统的 sys.exit(0)，安静退出，不打印红字报错 ===        
    except:
        traceback.print_exc()
        
    deep_clean_memory()
