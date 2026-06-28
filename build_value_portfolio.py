# -*- coding: utf-8 -*-
"""
build_value_portfolio.py — 生产组合: 输出【当期该持有名单 + 本季买卖 diff + 下次调仓日】。

严格实现【已验证回测】的那套参数, 不调、不加未验证项:
  · 季度调仓 (1/4/7/10月首个交易日)        ← 回测: 换手腰斩、回撤更浅、IR 0.51
  · 信号 = 行业+size 中性化 value (纯value)  ← 回测: @0.3%/边 IR 0.51 (raw 翻负, 不用)
  · 域 = 全市场 剔ST + 剔 circ_mv 最小30%   ← 已验证现实域
  · 持仓 = 域内最便宜 decile(10%) 等权       ← 分散、单票轻、压冲击成本

工作流(每季): 先 python fetch_longhist.py 增量更新数据 -> python build_value_portfolio.py
   -> 得到新名单 + 与上季 diff(买/卖哪些) -> 照着下单。
持仓存 value_portfolio_holdings.csv, 供下季算 diff。

[待验证扩展, 未默认开] 叠加质量池(便宜的好公司)/正交卫星(accruals/gm_mom): 须各自长样本重测过关再加, 不在此 bolt-on。
"""
import os
import datetime as dt

import numpy as np
import pandas as pd

import eval_value_longhist as V

# ── 冻结的生产参数 (源自已验证回测) ──
EXCL_SMALL_PCT = 0.30
TOP_PCT = 0.10
COST_PER_SIDE = 0.003          # 换手成本监控用
HOLD_FILE = './value_portfolio_holdings.csv'
# ─────────────────────────────────────


def quarterly_rebals(dates):
    s = pd.Series(dates, index=dates)
    monthly = s.groupby([s.index.year, s.index.month]).first()
    return monthly[pd.Index(monthly.index.get_level_values(1)).isin([1, 4, 7, 10])].tolist()


def main():
    print("① 读长历史 + 构建 中性化 value / circ_mv / ST集 ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind = ind_df['industry']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])
    name_map = ind_df['name']

    dates = value_raw.index.get_level_values('date').unique().sort_values()
    qrebals = quarterly_rebals(dates)
    today = pd.Timestamp(dt.date.today())
    cur = max(d for d in qrebals if d <= today)
    future_q = [d for d in qrebals if d > cur]
    nxt = future_q[0] if future_q else cur + pd.DateOffset(months=3)
    print(f"   数据至 {dates.max().date()} | 当期调仓日 {cur.date()} | 下次调仓 {nxt.date()}\n")

    # 当期选股
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, [cur]))
    v = value_neu.xs(cur, level='date').dropna()
    cm = circ.xs(cur).reindex(v.index)
    tradeable = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    tradeable = tradeable[~tradeable.isin(st)]
    vin = v.reindex(tradeable).dropna()
    picks = vin.nlargest(max(1, int(len(vin) * TOP_PCT))).index
    w = 1.0 / len(picks)

    # 富化(当期 pe/pb/circ_mv + value 分)
    dbx = db[db['date'] == cur].set_index('code')
    out = pd.DataFrame({'ts_code': [c + ('.SH' if c[0] == '6' else '.SZ') for c in picks],
                        'code': picks, 'weight': w,
                        'value分': vin.reindex(picks).round(3).values,
                        'pe_ttm': dbx['pe_ttm'].reindex(picks).round(1).values,
                        'pb': dbx['pb'].reindex(picks).round(2).values,
                        'circ_mv亿': (dbx['circ_mv'].reindex(picks) / 1e4).round(1).values,
                        'name': name_map.reindex(picks).values})
    out = out.sort_values('value分', ascending=False).reset_index(drop=True)
    print(f"② 当期持仓: {len(out)} 只, 每只 {w:.2%} (域内 {len(vin)} 只可交易, 取最便宜 {TOP_PCT:.0%})")
    print(out[['ts_code', 'name', 'weight', 'value分', 'pe_ttm', 'pb', 'circ_mv亿']].head(25).to_string(index=False))
    if len(out) > 25:
        print(f"   ... 共 {len(out)} 只 (完整见 csv)")

    # 与上期 diff
    cur_set = set(picks)
    if os.path.exists(HOLD_FILE):
        prev = pd.read_csv(HOLD_FILE, dtype={'code': str})
        prev_date = pd.Timestamp(prev['rebal_date'].iloc[0])
        prev_set = set(prev['code'])
        if prev_date < cur:
            buys, sells = cur_set - prev_set, prev_set - cur_set
            n_prev = len(prev_set)
            turnover = sum(w for _ in buys) + sum(1.0 / n_prev for _ in sells) if n_prev else 1.0
            print(f"\n③ 本季调仓 diff (上期 {prev_date.date()} -> 当期 {cur.date()}):")
            print(f"   买入 {len(buys)} 只 | 卖出 {len(sells)} 只 | 保留 {len(cur_set & prev_set)} 只")
            print(f"   双边换手 ≈ {turnover:.0%} | 估算成本 ≈ {turnover * COST_PER_SIDE:.2%} (按 {COST_PER_SIDE:.1%}/边)")
            if buys:
                print(f"   买入: {', '.join(sorted(c+('.SH' if c[0]=='6' else '.SZ') for c in buys))[:300]}")
            if sells:
                print(f"   卖出: {', '.join(sorted(c+('.SH' if c[0]=='6' else '.SZ') for c in sells))[:300]}")
        elif prev_date == cur:
            print(f"\n③ 已是本季({cur.date()})持仓, 无新交易 (下次调仓 {nxt.date()})。")
    else:
        print(f"\n③ 首次建仓: 全部 {len(out)} 只为初始买入。")

    # 存当期持仓 (供下季 diff) + 导出
    save = out[['code', 'ts_code', 'weight', 'name']].copy()
    save['rebal_date'] = cur.date()
    save.to_csv(HOLD_FILE, index=False, encoding='utf-8-sig')
    out.to_csv('./value_portfolio_current.csv', index=False, encoding='utf-8-sig')
    print(f"\n持仓已存 {HOLD_FILE} (下季算 diff用); 当期明细 -> value_portfolio_current.csv")
    print(f"⚠️ 下季流程: {nxt.date()} 前先跑 fetch_longhist.py 增量更新, 再跑本脚本拿新名单+diff。")
    print("   纪律: 参数已冻(季度/中性value/decile/剔ST微盘); value 会迟到, 须扛数年超额回撤; 成本控 ≤0.3%/边。")


if __name__ == '__main__':
    main()
