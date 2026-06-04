"""
logic_matrix.py — rule-based overlay (LogicMatrixPredictorV5) and the
integrated diagnostic auditor (IntegratedAuditorV5).
"""
import gc
import operator
import numpy as np
import pandas as pd


class LogicMatrixPredictorV5:
    """
    负责核心规则计算、类别评分及动态权重分配
    适配数据：已执行过 rank(pct=True) 的 0-1 归一化特征
    """
    def __init__(self):
        # 类别定义
        self.categories = ['momentum', 'liquidity', 'volatility', 'structure', 'market']
        # 基础权重分配
        self.base_weights = {'momentum': 0.25, 'liquidity': 0.25, 'volatility': 0.15, 'structure': 0.20, 'market': 0.15}
        
        # 定义核心规则库 (基于专家经验的逻辑组合)
        # 规则格式: (lambda函数, 分数贡献, 规则名称说明)
        self.rules = {
            'momentum': [
                (lambda x: x['mom_acc'] > 0.75 and x['near_high'] > 0.8, 1.0, "加速突破高点"),
                # UPGRADE Edit 6: dist_high is rank-normalized to [0,1]; the old "> -0.05"
                # was always True, collapsing this rule to just ret_10 > 0.7. Use a real rank threshold.
                (lambda x: x['ret_10'] > 0.7 and x['dist_high'] > 0.7, 0.8, "中期趋势强劲且临近突破"),
                (lambda x: x['mom_acc'] < 0.25 and x['ret_20'] < 0.3, -1.0, "趋势严重走坏")
            ],
            'liquidity': [
                (lambda x: x['buy_force'] > 0.7 and x['vol_z'] > 0.7, 0.9, "资金强力介入"),
                (lambda x: x['illiq_ma10'] < 0.3 and x['smart_proxy'] > 0.6, 0.7, "机构低阻力拉升"),
                (lambda x: x['vol_ratio'] < 0.3 and x['buy_force'] < 0.4, -0.8, "流动性陷阱/无人接盘")
            ],
            'volatility': [
                (lambda x: x['vol_20'] < 0.3 and x['amp_ma_5'] < 0.3, 0.6, "极度缩量蓄势"),
                (lambda x: x['vol_10'] > 0.8 and x['skew_10'] < 0.2, -0.9, "高位放量派发风险")
            ],
            'structure': [
                (lambda x: x['rsi'] < 0.25 and x['reversal'] > 0.75, 1.0, "严重超跌超卖反转"),
                (lambda x: x['turnover_cv'] < 0.4, 0.3, "换手平稳安全")
            ],

            'market': [
#       # Original rule: low breadth + shrinking volume
                (lambda x: x['mkt_breadth'] < 0.35 and x['market_vol_ratio'] < -0.6, -1.2, "市场缩量阴跌风险"),

#       # NEW FIX: extreme bear — fires even when volume is flat/slightly positive
#       # In v2, breadth=29% + vol_ratio=0.14 (not negative) → rule never fired
#       # This patch ensures bear-market penalty always applies below 0.30
                (lambda x: x['mkt_breadth'] < 0.30,-0.8, "市场极端低迷 (广度<30%)"),
#
#       # NEW: high beta in bear market — explicit penalty
#       # Requires beta_60d to be in the row (it is after v3 feature engineering)
                (lambda x: x.get('beta_60d', 0.5) > 0.75 and x['mkt_breadth'] < 0.40,-0.6, "熊市高Beta个股风险"),
            ]
        }

    def predict_diagnostics(self, row: pd.Series):
        """执行预测并返回详细诊断包"""
        cat_scores = {}
        hits = {}
        
        # 1. 逐类别计算得分
        for cat in self.categories:
            score = 0.0
            triggered = []
            for func, val, desc in self.rules.get(cat, []):
                try:
                    if func(row):
                        score += val
                        triggered.append(f"{desc}({val:+.1f})")
                except: continue
            cat_scores[cat] = np.clip(score, -1.0, 1.0)
            hits[cat] = triggered

        # 2. 动态权重调整
        weights = self.base_weights.copy()
        m_breadth = row.get('mkt_breadth', 0.5)
        if m_breadth < 0.35: # 环境恶劣时：压低动量权重，提升避险意识
            weights['momentum'] -= 0.1; weights['market'] += 0.1
        
        # 3. 最终判定
        total_score = sum(cat_scores[cat] * weights[cat] for cat in self.categories)
        
        # 信号映射
        if total_score >= 0.55: signal = 'STRONG_BUY'; conf = 0.7 + (total_score-0.55)*0.5
        elif total_score >= 0.2: signal = 'BUY'; conf = 0.5 + (total_score-0.2)*0.6
        elif total_score > -0.2: signal = 'NEUTRAL'; conf = 0.5
        else: signal = 'SELL'; conf = 0.6

        return {
            'total_score': total_score,
            'signal': signal,
            'confidence': conf,
            'cat_scores': cat_scores,
            'weights': weights,
            'hits': hits
        }

# ==========================================================
# 2. 集成审计器 (The Diagnostic Interface)
# ==========================================================
class IntegratedAuditorV5:
    """
    V5.0 集成系统：整合因子定义、逻辑矩阵中间结果与 LGBM 评分
    """
    def __init__(self, predictor: LogicMatrixPredictorV5):
        self.predictor = predictor
        # 安全算子映射
        self.ops_map = {'>': operator.gt, '<': operator.lt}
        
        # --- 因子元数据全量对齐 (适配 Rank 0-1) ---
        self.meta = {
            # 动量类
            'mom_acc':    {'name': '动量加速度', 'bull': ('>', 0.75), 'bear': ('<', 0.25), 'meaning': '短期趋势强于中期趋势，值高说明强势加速'},
            'ret_10':     {'name': '10日收益率', 'bull': ('>', 0.70), 'bear': ('<', 0.30), 'meaning': '最近10天累计收益排名'},
            'dist_high':  {'name': '距离60日高', 'bull': ('>', 0.85), 'bear': ('<', 0.20), 'meaning': '值越接近1说明距离60日高点越近'},
            'bias_20':    {'name': '20日乖离率', 'bull': ('>', 0.80), 'bear': ('<', 0.20), 'meaning': '收盘价相对20日均线的偏离排名'},
            'near_high':  {'name': '20日高点位', 'bull': ('>', 0.90), 'bear': ('<', 0.30), 'meaning': '值近1表示有突破潜力'},
            # 资金类
            'vol_ratio':  {'name': '成交量比',   'bull': ('>', 0.75), 'bear': ('<', 0.25), 'meaning': '当日成交量相对均量的排名'},
            'buy_force':  {'name': '买入力度',   'bull': ('>', 0.70), 'bear': ('<', 0.30), 'meaning': '近5日上涨日成交量占比排名'},
            'pv_corr':    {'name': '价量相关性', 'bull': ('>', 0.65), 'bear': ('<', 0.20), 'meaning': '价格与成交量的滚动正相关性'},
            'illiq_ma10': {'name': '非流动性',   'bull': ('<', 0.25), 'bear': ('>', 0.75), 'meaning': '越低表示冲击成本越小，拉升越容易'},
            'smart_proxy':{'name': '机构因子',   'bull': ('>', 0.70), 'bear': ('<', 0.30), 'meaning': '收盘价/VWAP，值高代表机构介入'},
            'vol_z':      {'name': '成交量能量', 'bull': ('>', 0.75), 'bear': ('<', 0.25), 'meaning': '成交量相对自身波动的能量强度'},
            # 波动类
            'vol_20':     {'name': '20日波动',   'bull': ('<', 0.30), 'bear': ('>', 0.80), 'meaning': '低波动在A股通常代表筹码稳定'},
            'skew_10':    {'name': '10日偏度',   'bull': ('>', 0.70), 'bear': ('<', 0.30), 'meaning': '正偏代表上涨潜力，负偏代表下跌风险'},
            'amp_ma_5':   {'name': '5日平均振幅', 'bull': ('<', 0.30), 'bear': ('>', 0.80), 'meaning': '振幅小代表筹码稳定'},
            # 结构类
            'rsi':        {'name': 'RSI指标',    'bull': ('<', 0.20), 'bear': ('>', 0.80), 'meaning': '超卖区看涨，超买区警惕'},
            'reversal':   {'name': '反转强度',   'bull': ('>', 0.80), 'bear': ('<', 0.20), 'meaning': '负涨跌幅，值高意味着昨日大跌'},
            'turnover_cv':{'name': '换手稳定性', 'bull': ('<', 0.30), 'bear': ('>', 0.70), 'meaning': '低CV表示量能稳定，无妖股特征'},
            # 市场类
            'mkt_breadth':{'name': '市场宽度',   'bull': ('>', 0.60), 'bear': ('<', 0.35), 'meaning': '股价高于5日线的个股占比'},
            'market_vol_ratio': {'name': '市场量比', 'bull': ('>', 0.50), 'bear': ('<', -0.6), 'meaning': '全市场总成交额偏离度'}
        }

    def _get_status(self, feat, val):
        m = self.meta.get(feat)
        if not m: return ""
        # 判定好坏
        if self.ops_map[m['bull'][0]](val, m['bull'][1]): return f"✅{m['name']}({val:.2f}):{m['meaning']}"
        if self.ops_map[m['bear'][0]](val, m['bear'][1]): return f"❌{m['name']}({val:.2f}):{m['meaning']}"
        return ""

    def run_audit(self, top_df: pd.DataFrame):
        try:
            date = top_df.index.get_level_values('date').unique()[0]
        except: date = "Unknown"

        print("\n" + "█"*125)
        print(f"🚀 [V5.0 集成决策诊断报告] 日期: {date} | 核心引擎: LogicMatrixV5")
        print("█"*125)

        # 1. 市场宏观风控
        r0 = top_df.iloc[0]
        mb, mvr = r0.get('mkt_breadth', 0.5), r0.get('market_vol_ratio', 0.0)
        risk = (40 if mb < 0.35 else 0) + (30 if mvr < -0.7 else 0)
        
        print(f"🌐 市场风控：赚钱效应 {mb:.2%} | 大盘量能 {mvr:.2f} | 环境风险分: {risk}")
        if risk >= 60:
            print("🛑 【一票否决】缩量阴跌环境，避险系统强制关停！今日不建议任何交易。")
            #return
            is_blocked = True # 标记但不退出
        print("-" * 110)
        # 2. 个股深度审计
        print(f"\n{'No':<3} {'Code':<9} {'LGBM':<8} {'Logic':<8} {'Conf':<6} {'Signal':<10} {'核心诊断'}")
        print("-" * 125)

        for i, (idx, row) in enumerate(top_df.iterrows()):
            diag = self.predictor.predict_diagnostics(row)
            code = idx[1] if isinstance(idx, tuple) else idx
            
            risk_icon = "🔥" if diag['total_score'] < 0 else "🟢"
            print(f"#{i+1:<2} {code:<9} {row['score']:>8.4f} {diag['total_score']:>8.2f} {diag['confidence']:>5.0%}  {risk_icon}{diag['signal']:<10}", end="")
            
            # 因子诊断：提取最显著的两个标记
            details = [self._get_status(f, row[f]) for f in self.meta.keys() if f in row]
            details = [d for d in details if d != ""]
            print(f" {details[0] if details else ''}")
            
            if i < 8: # 对前 8 名展示深度归因
                attr_str = "        ↳ [维度分] "
                for cat in self.predictor.categories:
                    sc, w = diag['cat_scores'][cat], diag['weights'][cat]
                    icon = "🔺" if sc > 0.2 else "🔻" if sc < -0.2 else "🔹"
                    attr_str += f"{cat[:3].upper()}:{icon}{sc:+.1f}({w:.0%})  "
                print(attr_str)
                
                all_hits = []
                for cat in self.predictor.categories: all_hits.extend(diag['hits'][cat])
                if all_hits: print(f"        ↳ [规则命中] {' | '.join(all_hits)}")
                
                if len(details) > 1:
                    for d in details[1:3]: print(f"        ↳ [因子意义] {d}")
            
            print("-" * 125)
            if i == 14: break

        gc.collect()
# ===================== MAIN =====================
