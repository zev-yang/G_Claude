"""
safety.py — market-regime safety / hedge filter (check_market_safety_v9).
"""
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd


def check_market_safety_v9(
    today_data: Dict,           # 当日数据字典
    recent_macro: pd.DataFrame, # 历史宏观数据（至少60日）
    avg_abs_icir: float         # 策略因子拥挤度
) -> Tuple[Optional[str], str]:
    
    # ---------------------------- 1. 数据健康检查与 Winsorize ----------------------------
    required_cols = ['mkt_breadth', 'cs_vol_ma5', 'market_vol_ratio', 'limit_down_count']
    if len(recent_macro) < 60:
        return None, "⚠ 历史数据不足60日，跳过避险系统检测。"
    
    hist_df = recent_macro.copy()
    for col in required_cols:
        if col in hist_df.columns:
            q99, q01 = hist_df[col].quantile(0.99), hist_df[col].quantile(0.01)
            hist_df[col] = hist_df[col].clip(q01, q99)

    # ---------------------------- 2. 指标提取 ----------------------------
    curr_brd = today_data['mkt_breadth']
    curr_vol_ratio = today_data['market_vol_ratio']
    curr_cs_vol = today_data['cs_vol_ma5']
    ld_count = today_data.get('limit_down_count', 0)
    
    brd_300 = today_data.get('breadth_300', curr_brd)
    brd_1000 = today_data.get('breadth_1000', curr_brd)
    today_low = today_data.get('low_price', 0.0)
    prev_low = today_data.get('prev_low', 0.0)
    
    prev_brd = hist_df['mkt_breadth'].iloc[-1]
    brd_acceleration = curr_brd - prev_brd  # 广度加速度
    # ---------------------------- 3. 滚动分位数计算 ----------------------------
    hist_60 = hist_df.tail(60)
    q_brd_10 = hist_60['mkt_breadth'].quantile(0.10)
    q_brd_20 = hist_60['mkt_breadth'].quantile(0.20)
    q_vol_05 = hist_60['market_vol_ratio'].quantile(0.05)
    q_vol_90 = hist_60['market_vol_ratio'].quantile(0.90)
    q_cs_vol_95 = hist_60['cs_vol_ma5'].quantile(0.95)
    q_ld_90 = hist_60['limit_down_count'].quantile(0.90)
    
    prev_brd_raw = hist_df['mkt_breadth'].iloc[-1]
    brd_recovery = curr_brd - prev_brd_raw
    q_rec_75 = hist_df['mkt_breadth'].diff().tail(60).quantile(0.75)

    # ---------------------------- 4. 跌停趋势与状态定义 (核心修复：移至此处) ----------------------------
    yesterday_data = recent_macro.iloc[-1] 
    prev_ld_count = yesterday_data.get('limit_down_count', 0)
    
    # 定义：是否发生跌停潮 (跌停数 > 历史90分位 且 绝对家数 > 80)
    is_ld_surge = (ld_count > q_ld_90) and (ld_count > 80)
    
    # 定义：情绪是否在明显修复 (跌停数比昨天减少40%以上 且 绝对数 < 150)
    ld_improving = (ld_count < prev_ld_count * 0.6) and (ld_count < 150)
    # [NEW: V9 Proactive Risk Detection]
    # Calculate Cross-Sectional Volatility Acceleration
    curr_cs_vol = today_data['cs_vol_ma5']
    hist_cs_vol_med = recent_macro['cs_vol_ma5'].median()
    vol_surge_ratio = curr_cs_vol / (hist_cs_vol_med + 1e-9)
    # ---------------------------- 5. 预期滑点模型 ----------------------------
    base_slippage = 0.0005
    liquidity_factor = np.clip(curr_brd / 0.5, 0.5, 2.0)
    volume_factor = np.clip(curr_vol_ratio / hist_60['market_vol_ratio'].median(), 0.3, 1.5)
    ld_penalty = 1 + (ld_count / 100)
    est_slip = base_slippage * (1/liquidity_factor) * (1/volume_factor) * ld_penalty

    # ---------------------------- 6. 决策矩阵 ----------------------------
    env_str = f"广度:{curr_brd:.0%}(1000:{brd_1000:.0%}) | 量比:{curr_vol_ratio:.2f} | 跌停:{ld_count}"

    # --- 第一层：豁免权 (必须放在拦截逻辑之前) ---
    # --- 在决策矩阵的第一层：豁免权部分增加 ---
    # 逻辑：处于绝对冰点(curr_brd < 0.2)，但今天广度修复显著(> 5%)，视为首日反转
    if curr_brd < 0.20 and brd_acceleration > 0.05:
        return None, f"✅ [首日修复] 冰点反转信号 (Acc:{brd_acceleration:+.2%})，豁免拦截"
    # 修复逻辑优先：如果跌停数大幅回落，直接放行，不看其他拦截
    if ld_improving:
         return None, f"✅ [情绪修复] 跌停回落 ({prev_ld_count:.0f}->{ld_count}) ({env_str})"

    # [NEW: Breadth-Volatility Divergence]
    # If market is "broad" (>60%) but volatility is surging (>1.5x normal)
    # This is a sign of a "Distribution Phase" (institutions dumping into retail)
    if today_data['mkt_breadth'] > 0.60 and vol_surge_ratio > 1.5:
        return "⚠️ 高位放量大分歧 (Distribution Phase)", f"VolSurge:{vol_surge_ratio:.2f}"

    # [NEW: The "Freezing Path" - Pre-emptive Limit Down Detection]
    # If limit downs are increasing 3 days in a row, even if count is low
    ld_history = recent_macro['limit_down_count'].tail(3).values
    if len(ld_history) >= 3 and (today_data['limit_down_count'] > ld_history[-1] > ld_history[-2]):
        if today_data['limit_down_count'] > 50:
            return "📉 跌停家数连续攀升 (Sentiment Erosion)", f"LD:{today_data['limit_down_count']}"

    # 反转确认：满足冰点且价格不创新低，且不是处于恶化的跌停潮中
    if (len(hist_df) >= 2):
        p1_brd = hist_df['mkt_breadth'].iloc[-1]
        p1_vol = hist_df['market_vol_ratio'].iloc[-1]
        p2_brd = hist_df['mkt_breadth'].iloc[-2]
        p1_rec = p1_brd - p2_brd
        prev_condition = (p1_brd < q_brd_10) and (p1_vol > q_vol_90 or p1_rec > q_rec_75)
    else:
        prev_condition = False

    curr_bottom_signal = (curr_brd < q_brd_10) and (curr_vol_ratio > q_vol_90 or brd_recovery > q_rec_75)
    reversal_confirmed = curr_bottom_signal and (prev_condition or today_low > prev_low) and not is_ld_surge
    if reversal_confirmed:
        return None, f"✅ [确认反转] 两日确认/价格企稳 ({env_str})"

    # 恐慌赶底
    hist_cs_vol_tail = hist_df['cs_vol_ma5'].tail(20).values
    consecutive_spike = sum(1 for val in reversed(hist_cs_vol_tail) if val > q_cs_vol_95)
    vol_declining = curr_cs_vol < hist_df['cs_vol_ma5'].iloc[-1]
    is_panic_raw = (curr_brd < q_brd_10) and (curr_cs_vol > q_cs_vol_95)
    if is_panic_raw and (consecutive_spike < 2 or vol_declining) and not is_ld_surge:
        return None, f"✅ [恐慌赶底] 波动率收敛 ({env_str})"

    # --- 第二层：强拦截 ---
        # --- 新增：高位坠落拦截 ---
    # 获取过去3天的广度均值
    prev_brd_avg = recent_macro['mkt_breadth'].tail(3).mean()
    # 如果处于高位(0.6以上)且广度比3日均值下降超过 8%
    if curr_brd > 0.60 and (curr_brd - prev_brd_avg) < -0.08:
        return "⚠️ 高位广度拐头 (Rollover)", f"Brd:{curr_brd:.2f}, Delta:{(curr_brd-prev_brd_avg):.2f}"

    # --- 新增：分歧度拦截 (LGBM 易失效区) ---
    hist_cs_vol_med = recent_macro['cs_vol_ma5'].median()
    vol_surge_ratio = curr_cs_vol / (hist_cs_vol_med + 1e-9)
    # 如果横截面波动率是中位数的 1.6 倍，说明市场乱了，AI 在瞎猜
    if vol_surge_ratio > 1.6:
        return "🌪️ 市场剧烈分歧 (Dispersion)", f"VolRatio:{vol_surge_ratio:.2f}"
        # 获取最近4天的量能标准化序列 (Z-score)
  # ---------------------------- 1. 提取序列 ----------------------------
    # 提取最近10天的量能和广度，用于平滑计算
    vol_seq = recent_macro['market_vol_ratio'].tail(10)
    brd_seq = recent_macro['mkt_breadth'].tail(10)
    
    if len(vol_seq) < 5: return None, "数据不足"

    # ---------------------------- 2. 趋势计算 (Smoothing) ----------------------------
    # A. 量能斜率：计算最近3天量能的均值 vs 之前3天
    vol_ma3_now = vol_seq[-3:].mean()
    vol_ma3_prev = vol_seq[-6:-3].mean()
    vol_slope = vol_ma3_now - vol_ma3_prev # 负值代表量能在衰减
    
    # B. 广度动能：广度是否在走下坡路
    brd_ma3_now = brd_seq[-3:].mean()
    brd_ma3_prev = brd_seq[-6:-3].mean()
    brd_slope = brd_ma3_now - brd_ma3_prev

    # C. 确定“缩量力度”
    # 不再判断 d1<d2<d3，而是判断 3日累计降幅是否超过阈值
    # 或者 3日均线 明显低于 10日均线
    vol_vs_longterm = vol_seq[-3:].mean() - vol_seq.mean()

    # ---------------------------- 3. 增强型决策矩阵 ----------------------------
    
    # 【拦截逻辑 A】：量价双杀趋势（解决 3/13 阴跌初期）
    # 只要 3日量能斜率 < -0.3 且 3日广度斜率 < -0.05
    if vol_slope < -0.30 and brd_slope < -0.05:
        return "🛑 趋势拦截：量价双向衰减 (Divergence Start)", \
               f"VolSlope: {vol_slope:.2f}, BrdSlope: {brd_slope:.2f}"

    # 【拦截逻辑 B】：存量博弈下的流动性枯竭
    # 广度低于 40% 且 3日均量 处于 Z-Score 负值区
    if today_data['mkt_breadth'] < 0.40 and vol_ma3_now < -0.5:
         return "💤 风险拦截：存量枯竭环境", f"Brd: {today_data['mkt_breadth']:.2f}, VolMA3: {vol_ma3_now:.2f}"

    # 【拦截逻辑 C】：高位缩量背离
    if today_data['mkt_breadth'] > 0.60 and vol_slope < -0.5:
        return "⚠️ 风险拦截：高位缩量背离", "警惕机构悄悄撤退"

    # --- 1. 高位坠落检测 (更敏感但容错) ---
    max_brd_5d = recent_macro['mkt_breadth'].tail(5).max()
    # 调低门槛到 -0.10，但增加一个“成交额验证”
    if max_brd_5d > 0.65 and (curr_brd - max_brd_5d) < -0.10:
        if today_data['market_vol_ratio'] < 0: # 只有缩量下跌才判定为派发风险
            return "⚠️ 高位缩量回撤", "Distribution"

    # --- 2. 分歧度检测 (下调至 1.5) ---
    hist_cs_vol_med = recent_macro['cs_vol_ma5'].median()
    if curr_cs_vol > hist_cs_vol_med * 1.5 and curr_brd > 0.25: # 只有非冰点期的剧烈分歧才拦截
        return "🌪️ 分歧过大", "Unstable"

    # --- 3. 【核心改进：增加再入场机制】 ---
    # 即使广度很低，但如果今天出现了强力修复（增加 > 5%），则强制判定为安全
    if curr_brd < 0.30 and brd_acceleration > 0.05:
        return None, "✅ 冰点首日强修复 (Re-entry)"
        
    if est_slip > 0.0035: # 稍微放宽滑点到 0.35%
        return f"💧 滑点过高 ({est_slip:.3%})", env_str

    if is_ld_surge:
        return f"💀 跌停潮 ({ld_count}家) 锁死流动性", env_str

    divergence = brd_300 - brd_1000
    if abs(divergence) > 0.45 or (brd_1000 < 0.15 and brd_300 > 0.45):
         return f"📉 极端分化 (Divergence:{divergence:+.2f})", env_str

    if curr_brd < q_brd_10:
        return f"❄ 绝对冰点无修复", env_str

    # 地量见地价放行
    if curr_vol_ratio < q_vol_05:
        if curr_brd < q_brd_20:
             return None, f"⏸️ 地量见地价，建议轻仓试探"
        else:
             return f"🥀 无量阴跌", env_str

    # 高位风险
    if curr_brd > 0.75: # 稍微调高高位阈值
        if curr_vol_ratio < q_vol_05:
            return f"🔥 高位缩量背离", env_str
        if avg_abs_icir > 3.5:
            return f"🔥 因子拥挤", env_str

    
#   # FIX: persistent bear market warning (breadth < 35% for 5+ days)
    brd_last5 = hist_df['mkt_breadth'].tail(5)
    if (brd_last5 < 0.35).all():
        avg_brd5 = brd_last5.mean()
        return None, (
            f"⚠️ 持续熊市 (5日广度均{avg_brd5:.0%}<35%)，"
            f"建议仓位≤50% ({env_str})"
        )
#
#   # Weak recovery from very low breadth → half position
    if curr_brd < 0.40 and brd_recovery > 0.03:
        return None, f"📈 弱市修复中，建议半仓 ({env_str})"
#
    return None, f"🟢 安全 ({env_str})"
