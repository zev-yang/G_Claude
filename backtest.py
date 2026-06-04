"""
backtest.py — walk-forward backtest engine (DailyAuditor).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

from config import CONFIG, FAMILY_MAP, GROUPS_V3, RED, GREEN, RESET, _zscore
from safety import check_market_safety_v9
from logic_matrix import LogicMatrixPredictorV5
from portfolio import build_sector_clusters, diversify_picks


class DailyAuditor:
    def __init__(self, panel, feats):
        self.panel = panel.sort_index()
        self.feats = feats
        self.dates = panel.index.get_level_values('date').unique().sort_values()
        # === MODIFIED: 增加一个变量存储最后一次筛选的特征，供生产使用 ===
        self.last_active_feats = feats 

    def check_ic(self):
        """此函数仅用于展示全样本下的因子表现，方便你观察"""
        valid_y = self.panel.dropna(subset=['target'])
        print("\n📊 [Step 3] Global IC & ICIR Analysis (Historical)...")
        
        # === MODIFIED: 核心 ICIR 计算逻辑 ===
        # 1. 计算每日 Cross-sectional IC
        daily_ics = []
        for d, group in valid_y.groupby(level='date'):
            if len(group) < 10: continue
            res = {'date': d}
            for f in self.feats:
                res[f] = group[f].corr(group['target'], method='spearman')
            daily_ics.append(res)
        
        ic_df = pd.DataFrame(daily_ics).set_index('date')
        
        # 2. 计算 ICIR
        stats = []
        for f in self.feats:
            m_ic = ic_df[f].mean()
            s_ic = ic_df[f].std()
            icir = m_ic / (s_ic + 1e-9)
            stats.append({'Factor': f, 'MeanIC': m_ic, 'ICIR': icir})
        
        print(pd.DataFrame(stats).sort_values('ICIR', ascending=False).to_string(index=False))

    def run_simulation(self):
        # 1. 确定回测时间范围
        start_date = pd.Timestamp(CONFIG['audit_start'])
        start_idx = max(self.dates.searchsorted(start_date), CONFIG['train_window'])
        last_idx = len(self.dates) - (CONFIG['horizon'] + 1)
        num_periods = (last_idx - start_idx) // CONFIG['horizon']
        
        if num_periods <= 0:
            print("⚠️ 数据不足，无法回测")
            return pd.DataFrame(), None
            
        start_idx = last_idx - num_periods * CONFIG['horizon'] - 1
        sim_dates = self.dates[start_idx:last_idx+1]
        
        print(f"\n🏃 [Step 4] Simulation Start (Elite Structural Selection)...")
        
        model = lgb.LGBMRegressor(
            n_estimators=300,        # <--- 增加迭代次数，100次可能欠拟合
            learning_rate=0.03,      # <--- 降低学习率，配合更多的迭代次数，搜索更精细
            max_depth=5,             # <--- 稍微加深，从3层调到5层，允许模型学习特征组合
            num_leaves=31,           # <--- 配合深度，增加叶子节点数
            min_data_in_leaf=50,      # <--- 关键新增：强制每个规律必须覆盖50只票
            reg_lambda=10,           # <--- 显著提高 L2 正则，防止由于深度增加导致的过拟合
            reg_alpha=2,             # <--- 引入 L1 正则，自动剔除无用特征
            colsample_bytree=0.7,    # <--- 降低随机特征采样比例，增加特征间的独立性观察
            subsample=0.8,           # <--- 引入行采样，增加鲁棒性
            verbose=-1, 
            random_state=42, 
            n_jobs=-1
        )
        
        idx = pd.IndexSlice
        logs = []
        step = CONFIG['horizon']
        trade_dates = sim_dates[::step]
        
        # 预定义战队分组

        # GROUPS_V3 is imported from config (shared, identical to production).
        for t in tqdm(trade_dates):
            # A. 准备数据窗口 (训练窗口 300天，评估窗口 60天)
            t_loc = self.dates.get_loc(t)
            # UPGRADE Edit 2: PURGE. Labels look `horizon` days forward, so the last training
            # row must end on/before t. Old code used t_loc-2 -> labels leaked ~7 days past the
            # signal date. This now matches the (correct) production cutoff: t_loc-horizon-1.
            train_end = self.dates[t_loc - CONFIG['horizon'] - 1 - CONFIG['embargo']]
            train_start = self.dates[max(0, t_loc - CONFIG['train_window'])]
            tr_data = self.panel.loc[idx[train_start:train_end, :], :].dropna(subset=['target'])
            
            if len(tr_data) < 500: continue
            
            ic_eval_start = self.dates[max(0, t_loc - CONFIG['icir_window'])]
            ic_eval_data = tr_data.loc[idx[ic_eval_start:train_end, :], :]
            
            # B. 计算 ICIR 选拔赛
            current_ics = []
            for d, group in ic_eval_data.groupby(level='date'):
                res = {f: group[f].corr(group['target'], method='spearman') for f in self.feats}
                current_ics.append(res)
            
            temp_ic_df = pd.DataFrame(current_ics)
            stats_list = []
            for f in self.feats:
                m_ic = temp_ic_df[f].mean()
                s_ic = temp_ic_df[f].std()
                icir = m_ic / (s_ic + 1e-9)
                stats_list.append({
                    'Factor': f, 
                    'ICIR': icir,        # <--- 必须保留原始正负值
                    'AbsICIR': abs(icir) # <--- 用于之前的排序逻辑
                })
            
            win_stats = pd.DataFrame(stats_list)
            sorted_candidates = win_stats.sort_values('AbsICIR', ascending=False)
            #   # FIX 2: hard ICIR floor — discard near-zero factors before selection
        #   # open_strength (0.010) and streak (0.074) were consuming slots in v2
            MIN_ICIR_FLOOR = 0.08
            sorted_candidates = sorted_candidates[
                sorted_candidates['AbsICIR'] >= MIN_ICIR_FLOOR
            ]
            if len(sorted_candidates) < 4:
                tqdm.write(f"  ⚠️ Only {len(sorted_candidates)} factors above "
                           f"ICIR floor at {t.date()}, using 0 return")
                # NOTE: 'valid' isn't built yet at this point, so the universe
                # benchmark can't be computed here; log NaN (period is skipped).
                logs.append({'date': t, 'Strat': 0.0, 'Bench': np.nan})
                continue  # (use 'continue' in simulation; 'pass' in Step 6)
            # C. 结构化优中选优逻辑
            MAX_TOTAL_FEATS = 12
            MAX_PER_GROUP = 4
            CORR_THRESHOLD = 0.62
            # --- 在 run_simulation 循环内部添加 ---
            # 1. 提取更全面的宏观历史
            macro_cols = ['mkt_breadth', 'cs_vol_ma5', 'market_vol_ratio', 'limit_down_count', 'mkt_low_level', 'brd_300', 'brd_1000']
            macro_history = self.panel[macro_cols].groupby(level=0).first().sort_index()
                # 获取当前日期 t 的宏观数据
            if t not in macro_history.index: continue
            today_data = macro_history.loc[t]  # <--- 【关键修复：在这里定义 today_data】
            recent_macro = macro_history.loc[:t].tail(60)
            # 1. 提取市场量能及其增量 (假设 macro_history 已在外部算好)
            curr_mvr = today_data['market_vol_ratio']
            # 计算最近3天的平均量能作为基准
            prev_mvr_avg = recent_macro['market_vol_ratio'].tail(3).mean()
            mvr_delta = curr_mvr - prev_mvr_avg # 【增量计算】
            
            # 2. 动态调整选拔名额 (进攻/防御切换)
            # 基础名额：Momentum:4, Volume:4, Reversion:2, Stability:2
            group_limits = {'Momentum': 4, 'Volume': 4, 'Reversion': 2, 'Stability': 2}
            
            # 如果 量能绝对值高 或 增量爆发 (>0.5个标准差) -> 进入进攻模式
            if curr_mvr > 0.8 or mvr_delta > 0.5:
                group_limits = {'Momentum': 6, 'Volume': 4, 'Reversion': 2, 'Stability': 0}
                tqdm.write(f"  🔥 [进攻模式] 市场放量/增量爆发 (MVR:{curr_mvr:.2f}, Delta:{mvr_delta:.2f})")
            # 如果 量能极度萎缩 -> 进入防御模式
            elif curr_mvr < -1.0:
                group_limits = {'Momentum': 1, 'Volume': 1, 'Reversion': 4, 'Stability': 6}
                tqdm.write(f"  🛡️ [防御模式] 市场极度缩量 (MVR:{curr_mvr:.2f})")
        #   # FIX: Bear market breadth override
#   # When breadth is very low, force defensive posture regardless of volume
#   # This is the key fix that prevents v2's high-beta tech stock problem
           
            final_active_feats = []
            group_counts = {g: 0 for g in GROUPS_V3.keys()}
            selected_families = set()  # <--- 新增：用于追踪已选中的家族
            for _, row in sorted_candidates.iterrows():
                f = row['Factor']
                if len(final_active_feats) >= MAX_TOTAL_FEATS: break
                # 1. 家族去重逻辑 (核心手术)
                f_family = FAMILY_MAP.get(f, f) # 如果没定义，则自成一派
                if f_family in selected_families:
                    continue # 该家族已有最强因子入选，跳过同族其他窗口的因子

                # 2. 战队配额检查
                belong_to = next((gn for gn, f_list in GROUPS_V3.items() if f in f_list), None)
                if not belong_to or group_counts[belong_to] >= group_limits[belong_to]: continue
                
                # 3. 相关性检查
                is_redundant = False
                if final_active_feats:
                    corrs = ic_eval_data[final_active_feats].corrwith(ic_eval_data[f]).abs()
                    if corrs.max() > CORR_THRESHOLD:
                        is_redundant = True
                
                if not is_redundant:
                    final_active_feats.append(f)
                    group_counts[belong_to] += 1
                    selected_families.add(f_family) # <--- 锁定该家族
            # UPGRADE Edit 7: a duplicated copy of the quota+correlation block used to sit here.
            # It was a no-op (self-correlation = 1.0 blocked any second add) but confusing; removed.

            # --- 修正点1：此处开始退回一级缩进，脱离选拔循环 ---
 # 1. 从 win_stats 提取入选因子的原始 ICIR（带正负号）
            # 注意：我们在 stats_list 里需要保存原始 ICIR 而不仅仅是 AbsICIR
            selected_info = []
            for f in final_active_feats:
                # 从 win_stats 找到对应的 ICIR 值
                # 假设 win_stats 包含 'Factor' 和 'ICIR' 列
                f_row = win_stats[win_stats['Factor'] == f].iloc[0]
                val_icir = f_row.get('ICIR', f_row.get('AbsICIR', 0)) # 兼容性处理
                selected_info.append(f"{f}({val_icir:.3f})")
            
            # 2. 格式化输出字符串
            factor_detail_str = " | ".join(selected_info)
            summary_dist = ", ".join([f"{k}:{v}" for k, v in group_counts.items() if v > 0])
            
            # 3. 使用 tqdm.write 一次性输出，防止 IOPub 报错
            tqdm.write("=" * 90)
            tqdm.write(f"📅 信号日期: {t.date()} | 因子总数: {len(final_active_feats)} | 战队构成: [{summary_dist}]")
            tqdm.write(f"🚀 因子详情: {factor_detail_str}")
            tqdm.write("=" * 90)

            # D. 模型训练 (修正点2：权重长度匹配 tr_data)
            #w = np.linspace(0.1, 1.0, len(tr_data))
            w = np.geomspace(0.1, 1.0, len(tr_data))
            model.fit(tr_data[final_active_feats], tr_data['target'], sample_weight=w)
            
            # E. 预测与回测执行
            try:
                today = self.panel.loc[t].copy()
            except: continue
            
            #valid = today[(today['adv'] > 5e6) & (today['close'] > 2)]
            valid = today[(today['adv'] > 15e6) & (today['close'] > 3.0)]
            if len(valid) < 5: continue
            
            # UPGRADE Edit 4a: removed a redundant model.predict here; lgbm_scores is computed
            # once below, after the safety check passes.
 # === [新增：空仓避险逻辑] ===
            
 # ==========================================================
# ==========================================================
# ==========================================================
            # [UPGRADE V3] 🚀 生产级：混合战术避险系统 (Hybrid Tactical Hedge)
            # ==========================================================
            # === [MODIFIED: Simulation 统一调用避险模块] ===
# === [FIXED: Simulation 统一调用避险模块] ===
# --- 在 DailyAuditor.run_simulation 的 for t in tqdm(trade_dates): 循环内修改 ---


            
            # 2. 获取当前和前一日数据
            t_loc = self.dates.get_loc(t)
            prev_t = self.dates[t_loc - 1] if t_loc > 0 else t
            today_row = macro_history.loc[t]
            prev_row = macro_history.loc[prev_t]
            
            # 3. 构建适配 V8 的字典
            today_data_dict = {
                'mkt_breadth': today_row['mkt_breadth'],
                'market_vol_ratio': today_row['market_vol_ratio'],
                'cs_vol_ma5': today_row['cs_vol_ma5'],
                'limit_down_count': today_row['limit_down_count'],
                'low_price': today_row['mkt_low_level'],
                'prev_low': prev_row['mkt_low_level'],
                'breadth_300': today_row['brd_300'],
                'breadth_1000': today_row['brd_1000']
            }
            
            # 4. 调用 V8 避险
            recent_macro_window = macro_history.loc[:t].tail(60)
            avg_abs_icir = win_stats[win_stats['Factor'].isin(final_active_feats)]['AbsICIR'].mean()
            
            # UPGRADE Edit 3: hedge is now toggleable. Set CONFIG['enable_hedge']=False to
            # measure raw alpha with NO safety filter (5/8 periods were blocked before).
            if CONFIG['enable_hedge']:
                skip_reason, market_env_str = check_market_safety_v9(today_data_dict, recent_macro_window, avg_abs_icir)
            else:
                skip_reason, market_env_str = None, "🟢 Hedge OFF (raw alpha mode)"

            # --- 拦截执行 ---
            if skip_reason:
                tqdm.write(f"🛑 [SIGNAL BLOCKED] {t.date()} -> 原因: {skip_reason}")
                # 记录为空仓收益
                logs.append({'date': t, 'Strat': 0.0, 'Bench': valid['ret_pnl'].mean()})
                continue
            else:
                if market_env_str != "⚠ 历史宏观数据不足，跳过避险系统检测。":
                    tqdm.write(f"✅ {t.date()} Market Environment: SAFE ({market_env_str})")
            # =======================================================

             # --- 修正后的回测 Top K 逻辑 (UPGRADE Edit 4b: same-scale z-blend) ---
            # Old fusion multiplied LGBM by exp(l) / (1 + 1.5*l): asymmetric (up to 2.5x up,
            # 0.37x down) and let a static, momentum-biased heuristic dominate the adaptive
            # model by magnitude. Now both signals are z-scored and blended; logic only TILTS
            # the ML rank, weighted by CONFIG['logic_tilt']. fused_score becomes a z (ranking
            # is scale-invariant, so downstream selection is unaffected).
            lgbm_scores = model.predict(valid[final_active_feats])
            if CONFIG['enable_logic_fusion']:
                logic_engine = LogicMatrixPredictorV5()
                logic_scores = np.array([
                    logic_engine.predict_diagnostics(row)['total_score']
                    for _, row in valid.iterrows()
                ])
                valid['fused_score'] = _zscore(lgbm_scores) + CONFIG['logic_tilt'] * _zscore(logic_scores)
            else:
                valid['fused_score'] = lgbm_scores
            today_breadth = today_data_dict.get('mkt_breadth', 0.5)
            cluster_map = build_sector_clusters(
                self.panel, t, lookback_days=60, corr_threshold=0.55
            )
            top = diversify_picks(
                valid,
                score_col    = 'fused_score',
                top_k        = CONFIG['top_k'],
                max_per_cluster = 2,
                cluster_map     = cluster_map,
            ).copy()
            top['pred_score'] = top['fused_score'] # 统一列名方便后续打印
            
            # --- 修正点3：优化打印输出，防止 IOPub 溢出 ---
            # 拼接所有信息后再打印，减少网络数据包频率
            output_lines = [f"\n📊 {t.date()} 选股报告 (Top 30):"]
            for i, (code, row) in enumerate(top.iloc[::-1].iterrows(), 1):
                stock_name = str(row.get('name', code))[:4]
                try:
                    sell_price = self.panel.loc[(t, code), 'open_exit']
                    ret_pct = (sell_price / row['open_entry'] - 1.0) * 100
                    color_code = RED if ret_pct > 0 else GREEN
                    ret_str = f"{color_code}{ret_pct:.2f}%{RESET}"
                except:
                    ret_str = "N/A"
                
                line = (f"  #{i:2d} {code:<7} {stock_name:<4} Score:{row['pred_score']:.4f} "
                        f"Price:{row['open_entry']:>6.2f} PnL:{ret_str:>8} "
                        f"MktVol:{row.get('market_vol_ratio', 0):.2f}")
                output_lines.append(line)
            
            tqdm.write("\n".join(output_lines)) # 一次性输出整天结果

            # F. 收益记录
            if not top['ret_pnl'].isnull().any():
                r_strat = top['ret_pnl'].mean() - 2 * CONFIG['cost_bps']
                r_bench = valid['ret_pnl'].mean()
                logs.append({'date': t, 'Strat': r_strat, 'Bench': r_bench})
            
        return pd.DataFrame(logs).set_index('date'), model

    def analyze(self, df):
        if df.empty: return
        df['Eq_Strat'] = (1+df['Strat']).cumprod()
        
        ann = df['Strat'].mean() * (252/CONFIG['horizon'])
        sh = ann / (df['Strat'].std() * np.sqrt(252/CONFIG['horizon']) + 1e-9)
        dd = (df['Eq_Strat']/df['Eq_Strat'].cummax()-1).min()
        
        print("\n📊 [Step 5] Audit Results:")
        print(f"Strat CAGR  : {ann:.2%}")
        print(f"Sharpe Ratio: {sh:.2f}")
        print(f"Max Drawdown: {dd:.2%}")
