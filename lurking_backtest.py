# -*- coding: utf-8 -*-
"""
lurking_backtest.py — 潜伏模式【独立】月度调仓回测引擎 (与 V25 完全隔离).

和 V25 的根本区别:
  · 调仓: 每月第一个交易日 (V25 是每 horizon 天)
  · 持仓: 3 / 6 / 12 个月 (V25 是 8 天)
  · 选股: 在质量池内按 hidden_alpha_neutral 选前 N 只等权 (V25 全市场 LGBM rank)
  · 成本: 单边 0.1% (潜伏换手低, 成本占比小)

★ 同样的纪律 (潜伏样本更短更珍贵, 反过拟合更重要):
  · 自带【留出期 OOS 分割】: 前 70% 调仓月开发, 后 30% 留出, 分别报 CAGR/Sharpe/MaxDD;
  · 多持仓期 (3/6/12月) 各自独立回测, 看哪个期限有效 + 是否区间稳健, 不挑单点最高;
  · 停牌处理: 调仓日无价 -> 跳过该股; 持仓期内退市 -> 按最后可得价了结。

输入 (全部已由前几块产出):
  price_panel:  (date,code)->hfq close (V25 data_loader 的 close)
  alpha_panel:  (date,code)->hidden_alpha_neutral (lurking_synthesis.neutralize 的输出)
  top_n:        每期持有只数 (默认 30)

注意: 这是回测引擎本身; 上层"装配脚本"(把数据喂进来跑) 是下一步, 单独交付。
"""
import numpy as np
import pandas as pd

COST_ONESIDE = 0.001       # 单边 0.1%
HOLD_MONTHS = (3, 6, 12)   # 三个持仓期各自回测
TOP_N_DEFAULT = 30
HOLDOUT_FRAC = 0.70


def _rebalance_dates(all_dates):
    """每月第一个交易日。"""
    s = pd.Series(all_dates, index=all_dates)
    return s.groupby([s.index.year, s.index.month]).first().tolist()


def _fwd_return(price, d0, d1, code):
    """code 从 d0 到 d1 的收益; 缺价(停牌/退市)用区间内最后可得价。None 表示无法计算。"""
    try:
        s = price.xs(code, level='code')
    except KeyError:
        return None
    s = s[(s.index >= d0) & (s.index <= d1)]
    if len(s) < 2 or pd.isna(s.iloc[0]):
        return None
    p0 = s.iloc[0]
    p1 = s.dropna().iloc[-1]            # 退市/停牌 -> 最后可得价
    return p1 / p0 - 1.0


def _period_metrics(rec_df):
    """重叠持仓篮子的正确年化指标。

    每个篮子收益是【持仓 days 天的累计收益】, 不能当"一期"简单乘频率年化(那是原 bug)。
    正确做法:
      · 年化收益(CAGR): 每个篮子收益按其持仓天数几何年化 (1+r)^(365/days)-1, 取均值;
      · 年化波动 + Sharpe + MaxDD: 全部用【不重叠】子序列 (按持仓期首尾相接采样) 计算,
        因为重叠篮子的收益高度自相关, 直接算 std 会被严重低估 -> Sharpe 虚高(上一版 bug)。
        不重叠序列里每个观测是独立的一个持仓期收益, 年化用 √(每年持仓期数)。
    这样 3/6/12 月口径一致、可比, 且 Sharpe 不再因重叠而爆炸。
    """
    r_all = rec_df['ret'].to_numpy(dtype=float)
    d_all = rec_df['days'].to_numpy(dtype=float)
    valid = ~np.isnan(r_all) & (d_all > 0)
    r_all, d_all = r_all[valid], d_all[valid]
    if len(r_all) < 2:
        return dict(cagr=float('nan'), sharpe=float('nan'), maxdd=float('nan'), n=int(len(r_all)))

    # CAGR: 每篮按实际持仓天数几何年化, 取均值 (跨持仓期口径一致)
    cagr = float(np.nanmean(np.power(1.0 + r_all, 365.0 / d_all) - 1.0))

    # 不重叠子序列 (首尾相接) -> 独立观测, 用于波动/Sharpe/MaxDD
    sub = rec_df.sort_values('entry')
    picked_r, picked_days, last_exit = [], [], None
    for _, row in sub.iterrows():
        if last_exit is None or row['entry'] >= last_exit:
            picked_r.append(row['ret']); picked_days.append(row['days']); last_exit = row['exit']
    picked_r = np.array(picked_r, dtype=float)

    if len(picked_r) >= 2:
        avg_days = np.nanmean(picked_days)
        ppy = 365.0 / avg_days if avg_days > 0 else 1.0      # 每年独立持仓期数
        ann_ret_np = np.nanmean(picked_r) * ppy
        vol_ann = np.nanstd(picked_r) * np.sqrt(ppy)
        sharpe = float(ann_ret_np / (vol_ann + 1e-9))
        eq = np.cumprod(1.0 + picked_r)
        dd = float((eq / np.maximum.accumulate(eq) - 1.0).min())
    else:
        sharpe, dd = float('nan'), float('nan')
    return dict(cagr=cagr, sharpe=sharpe, maxdd=dd, n=int(len(r_all)))


def backtest(price_panel, alpha_panel, top_n=TOP_N_DEFAULT, verbose=True):
    """逐持仓期(3/6/12月)月度调仓回测; 每期输出 全样本 + 开发 + 留出(OOS) 三段指标。"""
    price = price_panel.sort_index()
    if isinstance(price, pd.DataFrame):
        price = price.iloc[:, 0]
    alpha = alpha_panel.sort_index()
    if isinstance(alpha, pd.DataFrame):
        alpha = alpha.iloc[:, 0]

    all_dates = price.index.get_level_values('date').unique().sort_values()
    rebal = _rebalance_dates(all_dates)
    results = {}

    for months in HOLD_MONTHS:
        # 每期持有 months 个月; 每月调仓买入, 持有 months 后卖出 (滚动, 持仓重叠)。
        # 关键修复: 记录每个篮子的【入场日/出场日/区间天数】, 年化按【实际持仓天数】而非"调仓频率"。
        recs = []     # (entry_date, exit_date, holding_days, net_return)
        for rd in rebal:
            try:
                cs = alpha.xs(rd, level='date').dropna()
            except KeyError:
                continue
            if len(cs) < top_n:
                continue
            picks = cs.nlargest(top_n).index.tolist()
            exit_target = rd + pd.DateOffset(months=months)
            future = all_dates[all_dates >= exit_target]
            if len(future) == 0:
                continue                          # 未来不足 months -> 该篮子未到期, 跳过(无前视)
            exit_d = future[0]
            rets = [_fwd_return(price, rd, exit_d, c) for c in picks]
            rets = [x for x in rets if x is not None]
            if not rets:
                continue
            net = np.mean(rets) - 2 * COST_ONESIDE     # 买卖各一次单边成本
            hold_days = (exit_d - rd).days
            recs.append((rd, exit_d, hold_days, net))

        if len(recs) < 4:
            results[months] = dict(note='样本不足', n=len(recs))
            continue

        rec_df = pd.DataFrame(recs, columns=['entry', 'exit', 'days', 'ret']).sort_values('entry')
        results[months] = dict(
            full=_period_metrics(rec_df),
            dev=_period_metrics(rec_df.iloc[:int(len(rec_df) * HOLDOUT_FRAC)]),
            # 留出期按【出场日】落在后段切割: 避免重叠篮子把末端单边行情重复计入开发/留出
            oos=_period_metrics(rec_df[rec_df['exit'] >= rec_df['exit'].quantile(HOLDOUT_FRAC)]),
            n=len(rec_df),
            span=(rec_df['entry'].min(), rec_df['entry'].max()))

    if verbose:
        _print_report(results, top_n)
    return results


def _print_report(results, top_n):
    print(f"\n{'='*78}\n=== 潜伏模式 月度调仓回测 (top_n={top_n}, 单边成本{COST_ONESIDE:.1%}) ===")
    print(f"{'持仓期':>6}{'调仓数':>7}{'全样本CAGR':>12}{'全样本Sh':>10}"
          f"{'留出CAGR':>11}{'留出Sh':>9}{'留出MaxDD':>11}")
    for m in HOLD_MONTHS:
        r = results.get(m, {})
        if 'full' not in r:
            print(f"{m:>5}月  {r.get('note','-'):>40}")
            continue
        f, o = r['full'], r['oos']
        print(f"{m:>5}月{r['n']:>7}{f['cagr']:>11.1%}{f['sharpe']:>10.2f}"
              f"{o['cagr']:>10.1%}{o['sharpe']:>9.2f}{o['maxdd']:>11.1%}")
    print("\n判读 (同 V25 纪律): 看【留出期 OOS】是否为正且在 3/6/12 月间方向稳健;")
    print("  全样本好但留出期崩 = 过拟合; 不挑留出最高的单点持仓期当生产参数。")
