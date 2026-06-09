"""
market_position.py — Layer-1 大盘仓位 (market timing via aggregate money flow).

Builds a daily position coefficient (0 / 0.5 / 1.0) from MARKET-WIDE 主力净流入, used to scale
strategy exposure: go lighter / flat when aggregate smart-money flow is weak, full when strong.
Reads the SAME raw moneyflow partials as moneyflow_factors.py — so NO moneyflow_dc dependency,
and history reaches back to 2022 instead of 2023-09.

★ Thresholds HARD-CODED (写死), point-in-time. The 252-day rank uses ONLY trailing data — no
full-sample fitting. We OBSERVE whether it flags known bad regimes (e.g. 2026-03); we do NOT
tune the thresholds to make it do so (that would be data-snooping / 违反铁律).

Per the design:
    mkt_strength[t] = sum(主力净额, last 5d) / mean(当日总买入额, last 20d)
    rank[t]         = trailing-252d percentile rank of mkt_strength[t]   (point-in-time)
    position[t]     = 1.0 if rank>=0.70 ; 0.5 if 0.40<=rank<0.70 ; 0.0 if rank<0.40
position[t] is decided from data up to day t and applied to the NEXT holding window.

Standalone check:  python market_position.py tushare_cache/_partial/moneyflow
"""
import numpy as np
import pandas as pd
from moneyflow_factors import load_moneyflow   # reuse the same fastparquet reader

# ── hard-coded (写死) — DO NOT tune on the backtest ───────────────────────────────────
MP_NET_WINDOW    = 5      # 主力净额累计窗口
MP_TURN_WINDOW   = 20     # 当日总买入额均值窗口 (分母)
MP_RANK_LOOKBACK = 252    # 滚动分位回看
MP_FULL          = 0.70   # rank>=0.70 -> 满仓 1.0
MP_HALF          = 0.40   # 0.40<=rank<0.70 -> 半仓 0.5 ; <0.40 -> 空仓 0.0


def _trailing_pct_rank(s, win):
    """Point-in-time percentile rank: fraction of the trailing `win`-day window (incl. today)
    that today's value is >= . Uses only past+current data, never future."""
    return s.rolling(win, min_periods=win).apply(
        lambda x: (x[-1] >= x).mean(), raw=True)


def market_position_series(src='tushare_cache/_partial/moneyflow'):
    """Raw moneyflow partials -> daily position Series (index=date, value ∈ {0.0,0.5,1.0}; NaN before warmup)."""
    mf = load_moneyflow(src)
    date = pd.to_datetime(mf['trade_date'].astype(str), format='%Y%m%d')
    main_net = (mf['buy_lg_amount'] + mf['buy_elg_amount']) - (mf['sell_lg_amount'] + mf['sell_elg_amount'])
    total    = (mf['buy_sm_amount'] + mf['buy_md_amount'] + mf['buy_lg_amount'] + mf['buy_elg_amount'])

    daily = pd.DataFrame({'date': date,
                          'main_net': main_net.astype('float64'),
                          'total': total.astype('float64')})
    g = daily.groupby('date').sum().sort_index()          # market-wide aggregate per day

    net5   = g['main_net'].rolling(MP_NET_WINDOW,  min_periods=MP_NET_WINDOW).sum()
    turn20 = g['total'].rolling(MP_TURN_WINDOW, min_periods=MP_TURN_WINDOW).mean().replace(0, np.nan)
    strength = net5 / turn20
    rank = _trailing_pct_rank(strength, MP_RANK_LOOKBACK)

    pos = pd.Series(np.where(rank >= MP_FULL, 1.0,
                    np.where(rank >= MP_HALF, 0.5, 0.0)), index=g.index, dtype='float64')
    pos[rank.isna()] = np.nan                              # before 252-day warmup -> undefined
    return pos.rename('position'), strength.rename('mkt_strength'), rank.rename('mkt_rank')


if __name__ == '__main__':
    import sys
    src = sys.argv[1] if len(sys.argv) > 1 else 'tushare_cache/_partial/moneyflow'
    pos, strength, rank = market_position_series(src)
    n = pos.notna().sum()
    print(f"market position: {len(pos)} days, {n} with signal "
          f"({pos.index.min().date()}..{pos.index.max().date()})")
    vc = pos.value_counts(dropna=True).sort_index()
    print("position distribution:", {float(k): int(v) for k, v in vc.items()})
    # show the most recent 10 days
    tail = pd.concat([strength, rank, pos], axis=1).dropna().tail(10)
    print("\nlast 10 days:\n", tail.to_string())
