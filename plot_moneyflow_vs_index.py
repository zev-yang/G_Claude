# -*- coding: utf-8 -*-
"""
plot_moneyflow_vs_index.py — 全市场分层资金流 vs 沪深指数 (回测期 ~2 年) 一图看清。

读现成数据, 不新增任何下载依赖:
  · 资金流: tushare_cache/_partial/moneyflow/*.parquet (raw 分层 buy/sell, 万元)
  · 指数:   优先 Tushare index_daily (上证指数 000001.SH / 深证成指 399001.SZ, 需 TUSHARE_TOKEN);
            拿不到就用你自己的数据湖 stock_data_all 做等权指数代理 (6 开头=沪, 0/3 开头=深)
            — 等权口径其实更贴近你策略的持仓方式。

输出:
  · moneyflow_vs_index.png — 4 联图: ①沪深指数(归一) ②各层净流入20日滚动和
    ③各层累计净流入 ④全市场成交额
  · moneyflow_daily_agg.csv — 日度聚合数据 (想自己再切随便用)
  · 终端打印【观察用】相关性表: 各层净流入 vs 指数 同日/未来1日/未来5日收益

★ 铁律提示: 这是探索性观察, 不是调参依据。图上看着像信号的东西, 必须走同样的
  证伪流程 (结构 A/B) 才能进系统 — 我们已经在个股层证伪过 4 种资金流用法了。

用法:  python plot_moneyflow_vs_index.py [--days 504]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

MF_DIR   = 'tushare_cache/_partial/moneyflow'
LAKE_DIR = 'stock_data_all'
TIERS = {'elg': ('超大单', 'tab:red'), 'lg': ('大单', 'tab:orange'),
         'md': ('中单', 'tab:green'),  'sm': ('小单', 'tab:blue')}


# ── 1) 资金流日度聚合 ────────────────────────────────────────────────────────────
def load_mf_daily(days):
    files = sorted(glob.glob(os.path.join(MF_DIR, '*.parquet')))
    if not files:
        raise SystemExit(f"找不到资金流数据: {MF_DIR} — 先跑 run_data_update.py")
    files = files[-days:]                                   # 一天一个文件, 取最近 N 天
    rows = []
    for f in files:
        df = pd.read_parquet(f, engine='fastparquet')
        date = pd.to_datetime(str(df['trade_date'].iloc[0]), format='%Y%m%d')
        g = {}
        for t in TIERS:
            b, s = f'buy_{t}_amount', f'sell_{t}_amount'
            g[f'{t}_net'] = (df[b].sum() - df[s].sum()) / 1e4 if (b in df and s in df) else np.nan
        # 各层净额恒等和为 0 (每笔成交买卖双方各属一层) -> 缺哪层就用恒等式补
        for t in TIERS:
            if np.isnan(g[f'{t}_net']):
                others = [g[f'{o}_net'] for o in TIERS if o != t]
                if not any(np.isnan(x) for x in others):
                    g[f'{t}_net'] = -sum(others)
        buys = [c for c in df.columns if c.startswith('buy_') and c.endswith('_amount')]
        g['turnover'] = df[buys].sum().sum() / 1e4           # 万元 -> 亿元 (总买入额=总成交额)
        g['date'] = date
        rows.append(g)
    out = pd.DataFrame(rows).set_index('date').sort_index()
    out['main_net'] = out['lg_net'] + out['elg_net']         # 主力 = 大单 + 超大单
    return out


# ── 2) 指数: Tushare 优先, 数据湖等权代理兜底 ─────────────────────────────────────
def load_indices(start, end):
    try:
        import tushare as ts
        tok = os.environ.get('TUSHARE_TOKEN')
        if not tok:
            raise RuntimeError('no TUSHARE_TOKEN')
        pro = ts.pro_api(tok)
        out = {}
        for code, name in [('000001.SH', '上证指数'), ('399001.SZ', '深证成指')]:
            d = pro.index_daily(ts_code=code, start_date=start.strftime('%Y%m%d'),
                                end_date=end.strftime('%Y%m%d'))
            s = d.set_index(pd.to_datetime(d['trade_date'], format='%Y%m%d'))['close'].sort_index()
            out[name] = s.astype(float)
        print("  指数来源: Tushare index_daily (官方指数)")
        return out
    except Exception as e:
        print(f"  Tushare 指数不可用 ({e!r}) -> 用数据湖等权代理 (更贴近你的等权持仓口径)")
        return _lake_proxy_indices(start, end)


def _lake_proxy_indices(start, end):
    rets = {'沪市等权(湖代理)': [], '深市等权(湖代理)': []}
    files = glob.glob(os.path.join(LAKE_DIR, '*.csv'))
    if not files:
        raise SystemExit(f"湖也没有数据 ({LAKE_DIR}) — 无法构建指数")
    for f in files:
        code = os.path.basename(f)[:-4]
        key = '沪市等权(湖代理)' if code.startswith('6') else (
              '深市等权(湖代理)' if code[0] in '03' else None)
        if key is None:
            continue
        try:
            d = pd.read_csv(f, usecols=['date', 'close']).tail(600)
            d['date'] = pd.to_datetime(d['date'])
            d = d[(d['date'] >= start) & (d['date'] <= end)].set_index('date')
            r = d['close'].astype(float).pct_change()
            rets[key].append(r[(r.abs() < 0.25)])            # 剔除送转/数据毛刺
        except Exception:
            continue
    out = {}
    for key, lst in rets.items():
        mat = pd.concat(lst, axis=1)
        idx = (1 + mat.mean(axis=1).fillna(0)).cumprod() * 100
        out[key] = idx
    return out


# ── 3) 画图 + 观察表 ───────────────────────────────────────────────────────────
def main(days):
    mf = load_mf_daily(days)
    idx = load_indices(mf.index.min(), mf.index.max())
    sh_name = list(idx.keys())[0]
    sh = idx[sh_name].reindex(mf.index).ffill()

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True,
                             gridspec_kw={'height_ratios': [2, 2, 2, 1]})
    fig.suptitle(f"全市场分层资金流 vs 沪深指数  ({mf.index.min().date()} ~ {mf.index.max().date()})",
                 fontsize=14)

    ax = axes[0]                                             # ① 指数
    for name, s in idx.items():
        ax.plot(s.index, s / s.iloc[0] * 100, label=name, lw=1.4)
    ax.set_ylabel('指数 (起点=100)'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)

    ax = axes[1]                                             # ② 各层净流入 20 日滚动和
    for t, (label, color) in TIERS.items():
        ax.plot(mf.index, mf[f'{t}_net'].rolling(20).sum(), label=label, color=color, lw=1.3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('净流入 20日滚动和 (亿元)'); ax.legend(loc='upper left', ncol=4); ax.grid(alpha=0.3)

    ax = axes[2]                                             # ③ 累计净流入
    for t, (label, color) in TIERS.items():
        ax.plot(mf.index, mf[f'{t}_net'].cumsum(), label=label, color=color, lw=1.3)
    ax.plot(mf.index, mf['main_net'].cumsum(), label='主力(大+超大)', color='purple', lw=1.6, ls='--')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_ylabel('累计净流入 (亿元)'); ax.legend(loc='upper left', ncol=5); ax.grid(alpha=0.3)

    ax = axes[3]                                             # ④ 成交额
    ax.bar(mf.index, mf['turnover'], width=1.0, color='grey', alpha=0.45)
    ax.plot(mf.index, mf['turnover'].rolling(20).mean(), color='black', lw=1.2, label='MA20')
    ax.set_ylabel('成交额 (亿元)'); ax.legend(loc='upper left'); ax.grid(alpha=0.3)

    for ax in axes:                                          # 标注: 2026-03 惨淡段 + 924 行情
        ax.axvspan(pd.Timestamp('2026-03-01'), pd.Timestamp('2026-03-31'),
                   color='red', alpha=0.08)
        if mf.index.min() <= pd.Timestamp('2024-09-24') <= mf.index.max():
            ax.axvline(pd.Timestamp('2024-09-24'), color='green', ls=':', lw=1)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.tight_layout()
    fig.savefig('moneyflow_vs_index.png', dpi=130)
    mf.to_csv('moneyflow_daily_agg.csv', encoding='utf-8-sig')
    print(f"\n✅ 图已保存: {os.path.abspath('moneyflow_vs_index.png')}")
    print(f"✅ 日度数据: {os.path.abspath('moneyflow_daily_agg.csv')}")

    # ── 观察用相关性表 (仅描述, 不作调参依据) ──
    ret = sh.pct_change()
    print(f"\n📐 [观察] 各层净流入 vs {sh_name}收益 的相关系数 (N={len(mf)}):")
    print(f"  {'层级':<14}{'同日':>8}{'未来1日':>10}{'未来5日':>10}")
    for t, (label, _) in list(TIERS.items()) + [('main', ('主力(大+超大)', None))]:
        f = mf[f'{t}_net']
        c0 = f.corr(ret)
        c1 = f.corr(ret.shift(-1))
        c5 = f.corr(ret.shift(-1).rolling(5).sum().shift(-4))
        print(f"  {label:<14}{c0:>8.3f}{c1:>10.3f}{c5:>10.3f}")
    print("  (同日相关高=资金流是跟随; 未来列≈0=没有领先性 — 与我们个股层的证伪一致则收口)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=504, help='回看交易日数 (默认 ~2年)')
    main(ap.parse_args().days)
