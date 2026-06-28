# -*- coding: utf-8 -*-
"""
eval_value_portfolio.py — 把已确认的 value 因子做成【可落地的大盘多头 value-tilt 组合】, 扣现实成本。

承接 eval_value_longhist (value 已 PASS): 这里不再问"有没有 alpha", 而是问"扣成本/限大盘后, 还能拿多少净超额"。
形态 (落地 + 归因干净):
  · 选股域: 每个调仓日按 circ_mv 取前 N_LARGE 只 (PIT 大盘域 ≈ 沪深300 规模); A 股做空难, 只做多头 tilt。
  · 组合: 域内按【行业+size 中性化 value】取最便宜 TOP_PCT 等权, 持有到下次调仓。
  · 基准: ① 沪深300 指数 (000300.SH, 真基准); ② 同一大盘域等权 (size-clean 参照 -> 纯 value 归因)。
  · 成本: 按换手 × 单边成本, 扫 0/0.1/0.2/0.3%; 月度 vs 季度调仓对比 (低换手省成本)。
复用 eval_value_longhist 的构建块, 不重写。沪深300 指数自动拉取并缓存。
"""
import os
import datetime as dt

import numpy as np
import pandas as pd

import eval_value_longhist as V          # 复用 _read / build_hfq / build_value / neutralize / _xs_rank
import fetch_moneyflow_extra as F        # 仅用于拉沪深300指数

N_LARGE = 300
TOP_PCT = 0.30
COSTS = [0.0, 0.001, 0.002, 0.003]       # 单边成本
FREQS = {'月度': 1, '季度': 3}            # 调仓间隔(月)
IDX_FILE = './tushare_cache/_longhist/index/000300.parquet'


def load_csi300():
    if os.path.exists(IDX_FILE):
        df = pd.read_parquet(IDX_FILE, engine=V.ENGINE)
    else:
        os.makedirs(os.path.dirname(IDX_FILE), exist_ok=True)
        pro = F.get_pro()
        today = dt.date.today().strftime('%Y%m%d')
        df = F.call(pro, 'index_daily', ts_code='000300.SH', start_date='20100101', end_date=today)
        df.to_parquet(IDX_FILE, engine=V.ENGINE)
    df['date'] = pd.to_datetime(df['trade_date'])
    return df.set_index('date')['close'].sort_index()


def _turnover(prev_w, new_w):
    """双边换手 Σ|Δw| (买+卖)。"""
    idx = prev_w.index.union(new_w.index)
    return float((new_w.reindex(idx).fillna(0) - prev_w.reindex(idx).fillna(0)).abs().sum())


def _metrics(port_r, bench_r, ppy):
    """组合 vs 基准: 年化收益/超额, 跟踪误差, IR, 超额最大回撤。"""
    pr, br = np.asarray(port_r, float), np.asarray(bench_r, float)
    n = len(pr)
    p_ann = (np.prod(1 + pr)) ** (ppy / n) - 1
    b_ann = (np.prod(1 + br)) ** (ppy / n) - 1
    exc = pr - br
    te = exc.std() * np.sqrt(ppy)
    ir = (exc.mean() * ppy) / (te + 1e-12)
    rel = np.cumprod(1 + exc)
    dd = (rel / np.maximum.accumulate(rel) - 1).min()
    return p_ann, p_ann - b_ann, te, ir, dd


def main():
    print("① 读长历史 + 构建 价格 / 中性化 value / circ_mv ...")
    daily = V._read('daily', ['close']); adj = V._read('adj_factor', ['adj_factor'])
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    price = V.build_hfq(daily, adj)
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')['industry']

    dates = price.index.get_level_values('date').unique().sort_values()
    s = pd.Series(dates, index=dates)
    rebal_all = s.groupby([s.index.year, s.index.month]).first().tolist()
    wide = price.unstack('code')
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, rebal_all))
    idx_close = load_csi300()
    print(f"   样本 {dates.min().date()}~{dates.max().date()} | 大盘域前 {N_LARGE} | 最便宜 {TOP_PCT:.0%} 等权\n")

    print(f"净超额 (vs 沪深300 / vs 大盘域等权); 单边成本扫描")
    hdr = f"{'频率':>5}{'成本/边':>8}{'组合年化':>9}{'超额vs300':>10}{'超额vsEW':>10}{'跟踪误差':>9}{'IR':>6}{'超额回撤':>9}{'年换手':>8}"
    print(hdr); print('-' * len(hdr))

    for fname, step in FREQS.items():
        rebal = rebal_all[::step]
        rebal = [d for d in rebal if d in value_neu.index.get_level_values('date')]
        gross, bench, lcew, turns = [], [], [], []
        prev_w = pd.Series(dtype=float)
        for i in range(len(rebal) - 1):
            d, d1 = rebal[i], rebal[i + 1]
            try:
                v = value_neu.xs(d, level='date').dropna()
                cm = circ.xs(d).reindex(v.index).dropna()
            except KeyError:
                continue
            univ = cm.nlargest(N_LARGE).index                       # 大盘域 (PIT)
            vin = v.reindex(univ).dropna()
            if len(vin) < 20:
                continue
            picks = vin.nlargest(max(1, int(len(vin) * TOP_PCT))).index   # 域内最便宜
            try:
                p0, p1 = wide.loc[d, picks], wide.loc[d1, picks]
            except KeyError:
                continue
            r = (p1 / p0 - 1).mean()                                # 等权组合收益
            ew = (wide.loc[d1, univ] / wide.loc[d, univ] - 1).mean()  # 大盘域等权(参照基准)
            new_w = pd.Series(1.0 / len(picks), index=picks)
            turns.append(_turnover(prev_w, new_w)); prev_w = new_w
            gross.append(r); lcew.append(ew)
            bench.append(idx_close.asof(d1) / idx_close.asof(d) - 1)  # 沪深300 同期
        gross, bench, lcew, turns = map(lambda x: np.array(x, float), (gross, bench, lcew, turns))
        ppy = 12.0 / step
        avg_turn = np.nanmean(turns) * ppy
        for c in COSTS:
            net = gross - turns * c                                  # 扣换手成本
            p_ann, exc300, te, ir, dd = _metrics(net, bench, ppy)
            _, excEW, _, _, _ = _metrics(net, lcew, ppy)
            print(f"{fname:>5}{c*100:>7.1f}%{p_ann:>8.1%}{exc300:>+10.1%}{excEW:>+10.1%}{te:>8.1%}{ir:>6.2f}{dd:>8.1%}{avg_turn:>7.0%}")
    print("\n判读:")
    print("  · 看【vs 大盘域等权】那列 = 纯 value tilt 净效应 (归因最干净); vs沪深300 还含选股域与300成分的差异。")
    print("  · IR(扣成本后)>0.5 才算有落地价值; 成本升高/季度换手低 -> 净超额怎么变, 决定调仓频率。")
    print("  · A 股做空难, 这是多头 tilt 的可落地数; 别拿长样本多空 t=8 当到手收益。value 会迟到, 须扛长回撤。")


if __name__ == '__main__':
    main()
