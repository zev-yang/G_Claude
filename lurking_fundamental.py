# -*- coding: utf-8 -*-
"""
lurking_fundamental.py —— 潜伏 的基本面因子工具包 (与 V25 完全隔离, 不碰 LGBM)。

三步, 各自独立:
  1) build_factors(panel)  -> (date, code) float32 因子面板 [5 个 Lab_ 因子], PIT 对齐
  2) gate(factors, fwd, ...) -> 在【潜伏调仓频率】上评估: ICIR / IC_ac1 / IC_t_NW / 正交相关
  3) composite(factors, use=幸存因子) -> (date, code) 综合分 = 可得因子的横截面 rank 均值

输出格式刻意对齐你的 fundamental_factors.py: (date, code)-MultiIndex, float32, sort_index。
所以可以像 fundamentals_panel() 一样直接喂进 潜伏 的选股 (rank/加权), 无需 LGBM。

数据来源 (你的 cache 约定):
  - 三表: tushare_cache/{balancesheet,income,cashflow}.parquet  (fetch_financials_tushare 合并产物)
  - 市值: tushare_cache/_partial/daily_basic/*.parquet           (优先 total_mv, 退 circ_mv)

PIT 三道防线 (与 V25 的 lab_factors_fundamental 一致):
  公告日(f_ann_date)对齐 merge_asof(backward) · YTD->TTM · 首次披露去重(不用重述)。

FIELD_MAP 已全部核实: balancesheet 用上传 parquet 实测, income/cashflow 用官方文档。
"""
import os
import glob

import numpy as np
import pandas as pd

CACHE = 'tushare_cache'
ENGINE = 'fastparquet'
MV_UNIT_SCALE = 1e4          # daily_basic 市值 万元 -> 元
EPS = 1e-9

# 已核实的字段名 ----------------------------------------------------------------
F = {
    'contract_liab': 'contract_liab', 'adv_receipts': 'adv_receipts',
    'goodwill': 'goodwill', 'money_cap': 'money_cap', 'total_assets': 'total_assets',
    'ib_debt': ['st_borr', 'lt_borr', 'non_cur_liab_due_1y', 'bond_payable'],  # 缺的自动跳过
    'n_income': 'n_income', 'cfo': 'n_cashflow_act', 'capex': 'c_pay_acq_const_fiolta',
}

# 符号: 构造成"越高越好"。composite 里再 rank, 方向必须先统一。
#   正向: contract_liab_yoy, fcf_yield, net_cash   |  取负: accruals_cf, goodwill_ratio


# ============================================================================
# 0. 载入 + PIT 工具
# ============================================================================
def _norm_panel(panel):
    """接受 (date,code)-MultiIndex 或带 date/code 列的 DataFrame -> 唯一 (date,code) 网格。"""
    if isinstance(panel, pd.MultiIndex):
        p = panel.to_frame(index=False)
    elif isinstance(panel, pd.DataFrame):
        p = panel.reset_index() if isinstance(panel.index, pd.MultiIndex) else panel.copy()
    else:
        raise TypeError("panel 需为 (date,code) MultiIndex 或含 date/code 列的 DataFrame")
    p = p[['date', 'code']].copy()
    p['date'] = pd.to_datetime(p['date'])
    p['code'] = p['code'].astype(str)
    return p.drop_duplicates().sort_values('date')


def load_stmt(name, cache=CACHE):
    """读某张报表 -> 加 avail_date(最早公告日) + code, 首次披露去重 (PIT)。
    优先直接 glob _partial/<name>/*.parquet (与你 daily_basic 一致, 不依赖合并产物);
    无分片才回退合并文件。分片在 fetch 时已过滤 report_type==1 + 排除 .BJ。"""
    pdir = os.path.join(cache, '_partial', name)
    files = sorted(glob.glob(os.path.join(pdir, '*.parquet')))
    if files:
        df = pd.concat([pd.read_parquet(f, engine=ENGINE) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(os.path.join(cache, f'{name}.parquet'), engine=ENGINE)
    for c in ('f_ann_date', 'ann_date', 'end_date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    anns = [c for c in ('f_ann_date', 'ann_date') if c in df.columns]
    df['avail_date'] = df[anns].min(axis=1)
    if 'code' not in df.columns:
        df['code'] = df['ts_code'].astype(str).str[:6]
    df = df.dropna(subset=['avail_date', 'end_date'])
    df = df.sort_values('avail_date').drop_duplicates(['code', 'end_date'], keep='first')
    return df


def load_mktcap(cache=CACHE):
    """daily_basic 分片 -> (date, code, mktcap[元])。优先 total_mv, 退 circ_mv。"""
    files = sorted(glob.glob(os.path.join(cache, '_partial', 'daily_basic', '*.parquet')))
    if not files:
        raise FileNotFoundError("无 daily_basic 分片 — 先跑 run_data_update.py")
    parts = []
    for f in files:
        d = pd.read_parquet(f, engine=ENGINE)
        col = 'total_mv' if 'total_mv' in d.columns else 'circ_mv'
        parts.append(d[['ts_code', 'trade_date', col]].rename(columns={col: '_mv'}))
    db = pd.concat(parts, ignore_index=True)
    db['date'] = pd.to_datetime(db['trade_date'])
    db['code'] = db['ts_code'].astype(str).str[:6]
    db['mktcap'] = pd.to_numeric(db['_mv'], errors='coerce') * MV_UNIT_SCALE
    return db[['date', 'code', 'mktcap']]


def _ttm(df, col):
    """YTD 累计 -> TTM = 上年年报 + 本期累计 − 上年同期累计。"""
    b = df[['code', 'end_date', col]].dropna(subset=['end_date']).drop_duplicates(['code', 'end_date']).copy()
    b['year'] = b['end_date'].dt.year
    b['month'] = b['end_date'].dt.month
    same = b[['code', 'year', 'month', col]].rename(columns={col: '_py'})
    same['year'] += 1
    out = b.merge(same, on=['code', 'year', 'month'], how='left')
    ann = b.loc[b['month'] == 12, ['code', 'year', col]].rename(columns={col: '_ann'})
    ann['year'] += 1
    out = out.merge(ann, on=['code', 'year'], how='left')
    out['ttm'] = out['_ann'] + out[col] - out['_py']
    return out[['code', 'end_date', 'ttm']]


def _yoy(df, col):
    """资产负债表 stock 项同比 (同季对同季)。"""
    b = df[['code', 'end_date', col]].dropna(subset=['end_date']).drop_duplicates(['code', 'end_date']).copy()
    b['year'] = b['end_date'].dt.year
    b['month'] = b['end_date'].dt.month
    prev = b[['code', 'year', 'month', col]].rename(columns={col: '_py'})
    prev['year'] += 1
    out = b.merge(prev, on=['code', 'year', 'month'], how='left')
    out['yoy'] = out[col] / out['_py'] - 1.0
    out.loc[out['_py'].abs() < EPS, 'yoy'] = np.nan
    return out[['code', 'end_date', 'yoy']]


def _pit(panel, stmt, q, name):
    """q[code,end_date,<val>] 接回 avail_date 后, PIT 展开到 panel[date,code]。"""
    valcol = [c for c in q.columns if c not in ('code', 'end_date')][0]
    v = q.merge(stmt[['code', 'end_date', 'avail_date']].drop_duplicates(['code', 'end_date']),
                on=['code', 'end_date'], how='left').rename(columns={valcol: name})
    v = v[['code', 'avail_date', name]].dropna(subset=['avail_date']).sort_values('avail_date')
    out = pd.merge_asof(panel.sort_values('date'), v, left_on='date', right_on='avail_date',
                        by='code', direction='backward')
    return out[['date', 'code', name]]


# ============================================================================
# 1. 五个因子 -> (date, code) 面板
# ============================================================================
def build_factors(panel, cache=CACHE):
    """返回 (date, code)-MultiIndex float32 面板, 列 = 5 个 Lab_ 因子 (方向已统一为越高越好)。"""
    g = _norm_panel(panel)
    bs = load_stmt('balancesheet', cache)
    inc = load_stmt('income', cache)
    cf = load_stmt('cashflow', cache)
    mc = load_mktcap(cache)

    # 合同负债 (旧期用预收款项兜底) 同比
    bs2 = bs.copy()
    bs2['_cl'] = bs2[F['contract_liab']]
    if F['adv_receipts'] in bs2.columns:
        bs2['_cl'] = bs2['_cl'].fillna(bs2[F['adv_receipts']])
    cl = _pit(g, bs2, _yoy(bs2, '_cl'), 'Lab_contract_liab_yoy')

    # accruals_cf = (NI − CFO)/总资产, TTM; 取负 (高 accruals=差)
    ni = _pit(g, inc, _ttm(inc, F['n_income']).rename(columns={'ttm': '_ni'}), '_ni')
    co = _pit(g, cf, _ttm(cf, F['cfo']).rename(columns={'ttm': '_cfo'}), '_cfo')
    ta = _pit(g, bs, bs[['code', 'end_date', F['total_assets']]].rename(columns={F['total_assets']: '_ta'}), '_ta')
    acc = ni.merge(co, on=['date', 'code']).merge(ta, on=['date', 'code'])
    acc['Lab_accruals_cf'] = np.where(acc['_ta'].abs() > EPS,
                                      -((acc['_ni'] - acc['_cfo']) / acc['_ta']), np.nan)
    acc = acc[['date', 'code', 'Lab_accruals_cf']]

    # fcf_yield = (CFO − capex)_ttm / 市值
    cx = _pit(g, cf, _ttm(cf, F['capex']).rename(columns={'ttm': '_cx'}), '_cx')
    fcf = co.merge(cx, on=['date', 'code']).merge(mc, on=['date', 'code'], how='left')
    fcf['_fcf'] = fcf['_cfo'] - fcf['_cx'].fillna(0.0)
    fcf['Lab_fcf_yield'] = np.where(fcf['mktcap'].abs() > EPS, fcf['_fcf'] / fcf['mktcap'], np.nan)
    fcf = fcf[['date', 'code', 'Lab_fcf_yield']]

    # net_cash = (货币资金 − 有息负债)/总资产
    parts = [c for c in F['ib_debt'] if c in bs.columns]
    bs3 = bs.copy()
    bs3['_ibd'] = bs3[parts].fillna(0.0).sum(axis=1) if parts else 0.0
    cash = _pit(g, bs3, bs3[['code', 'end_date', F['money_cap']]].rename(columns={F['money_cap']: '_cash'}), '_cash')
    debt = _pit(g, bs3, bs3[['code', 'end_date', '_ibd']], '_debt')
    nc = cash.merge(debt, on=['date', 'code']).merge(ta, on=['date', 'code'])
    nc['Lab_net_cash'] = np.where(nc['_ta'].abs() > EPS, (nc['_cash'] - nc['_debt']) / nc['_ta'], np.nan)
    nc = nc[['date', 'code', 'Lab_net_cash']]

    # goodwill_ratio = 商誉/总资产; 取负 (高商誉=减值风险=差); 无并购=0
    bs4 = bs.copy()
    bs4['_gw'] = bs4[F['goodwill']].fillna(0.0)
    gw = _pit(g, bs4, bs4[['code', 'end_date', '_gw']], '_gw').merge(ta, on=['date', 'code'])
    gw['Lab_goodwill_ratio'] = np.where(gw['_ta'].abs() > EPS, -(gw['_gw'] / gw['_ta']), np.nan)
    gw = gw[['date', 'code', 'Lab_goodwill_ratio']]

    out = (g.merge(cl, on=['date', 'code'], how='left')
            .merge(acc, on=['date', 'code'], how='left')
            .merge(fcf, on=['date', 'code'], how='left')
            .merge(nc, on=['date', 'code'], how='left')
            .merge(gw, on=['date', 'code'], how='left'))
    cols = ['Lab_contract_liab_yoy', 'Lab_accruals_cf', 'Lab_fcf_yield',
            'Lab_net_cash', 'Lab_goodwill_ratio']
    out[cols] = out[cols].astype('float32')
    return out.set_index(['date', 'code'])[cols].sort_index()


# ============================================================================
# 2. gate —— 在【潜伏调仓频率】上评估 (预注册阈值, 写死)
# ============================================================================
ICIR_MIN = 0.25
CORR_MAX = 0.60
TSTAT_ADVISORY = 1.96   # |IC_t_NW|>=1.96 ≈ p<0.05 (诊断参考)


def _nw_t(ic, lag):
    x = ic.dropna().values
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = (e @ e) / n
    L = int(min(lag, n - 1))
    for l in range(1, L + 1):
        var += 2 * (1 - l / (L + 1.0)) * (e[l:] @ e[:-l]) / n
    se = np.sqrt(var / n)
    return x.mean() / se if se > 0 else np.nan


def gate(factors, fwd, existing=None, rebalance_dates=None, nw_lag=6):
    """
    factors: build_factors 输出 (或其子集), (date,code)-index。
    fwd:     DataFrame [date, code, fwd_ret] —— ★用潜伏持有期的长 horizon 收益, 不是 8 天。
    existing: 已有因子宽表 index=(date,code) (正交检验; None 跳过)。
    rebalance_dates: 潜伏调仓日 (强烈建议传, 否则慢因子 ICIR 虚高)。
    nw_lag:  Newey-West 滞后, 单位=调仓数 (6mo+月度调仓 -> 6)。
    读 IC_t_NW, 不是 ICIR_full。
    """
    from scipy.stats import spearmanr
    fac = factors.reset_index()
    rows = []
    for name in [c for c in fac.columns if c.startswith('Lab_')]:
        m = fac[['date', 'code', name]].merge(fwd, on=['date', 'code'], how='inner').dropna(subset=[name, 'fwd_ret'])
        if rebalance_dates is not None:
            m = m[m['date'].isin(pd.to_datetime(pd.Index(rebalance_dates)))]
        ics = {}
        for dt, gdf in m.groupby('date'):
            if gdf[name].nunique() >= 5:
                ic, _ = spearmanr(gdf[name], gdf['fwd_ret'])
                if np.isfinite(ic):
                    ics[dt] = ic
        ic = pd.Series(ics).sort_index()
        icir = ic.mean() / ic.std(ddof=1) if len(ic) > 1 and ic.std(ddof=1) > 0 else np.nan
        ac1 = ic.autocorr(1) if len(ic) > 2 else np.nan
        tnw = _nw_t(ic, nw_lag)
        corr, cwith = _max_corr(fac, existing, name) if existing is not None else (np.nan, None)
        rows.append(dict(
            factor=name, n_obs=len(ic),
            ICIR_full=round(icir, 3) if np.isfinite(icir) else np.nan,
            IC_ac1=round(ac1, 2) if np.isfinite(ac1) else np.nan,
            IC_t_NW=round(tnw, 2) if np.isfinite(tnw) else np.nan,
            maxabs_corr=round(corr, 3) if np.isfinite(corr) else np.nan, corr_with=cwith,
            PASS_icir=bool(np.isfinite(icir) and abs(icir) >= ICIR_MIN),
            PASS_corr=(existing is None) or bool(np.isfinite(corr) and abs(corr) < CORR_MAX),
            NW_signif=bool(np.isfinite(tnw) and abs(tnw) >= TSTAT_ADVISORY),
        ))
    return pd.DataFrame(rows)


def _max_corr(fac, existing, name):
    from scipy.stats import spearmanr
    f = fac.set_index(['date', 'code'])[name]
    j = existing.join(f.rename('_lab'), how='inner').dropna(subset=['_lab'])
    best, who = 0.0, None
    for col in existing.columns:
        cs = []
        for _, gdf in j[[col, '_lab']].dropna().groupby(level=0):
            if len(gdf) >= 5:
                c, _ = spearmanr(gdf[col], gdf['_lab'])
                if np.isfinite(c):
                    cs.append(c)
        if cs and abs(np.mean(cs)) > abs(best):
            best, who = float(np.mean(cs)), col
    return best, who


# ============================================================================
# 3. composite —— 可得因子的横截面 rank 均值 (透明, 零超参, 不碰 LGBM)
# ============================================================================
def composite(factors, use=None, min_factors=2):
    """
    每个因子按日做横截面 pct-rank ([0,1], 越大越好), 再取每只票【可得因子】的均值。
    -> 缺因子的票(如金融股缺 contract_liab/net_cash)拿公平均值, 不被白白拉低。
    use: 用哪些因子 (传 gate 幸存者; None=全 5 个)。
    min_factors: 一只票至少要有这么多个非 NaN 因子才给分, 否则 NaN。
    返回 Series[(date,code) -> 综合分]。
    """
    cols = use or [c for c in factors.columns if c.startswith('Lab_')]
    ranks = factors[cols].groupby(level='date').rank(pct=True)   # 横截面 rank, NaN 保留
    avail = ranks.notna().sum(axis=1)
    score = ranks.mean(axis=1, skipna=True)                       # 可得因子均值
    score[avail < min_factors] = np.nan
    return score.rename('lurking_fund_score').astype('float32')


# ============================================================================
# smoke test (合成数据, 验证 PIT/TTM/yoy/composite 逻辑; 非结果)
# ============================================================================
if __name__ == '__main__':
    print(">>> smoke test on synthetic data — validates logic, NOT a result.\n")
    rng = np.random.default_rng(0)
    codes = [f'{i:06d}' for i in range(30)]
    dates = pd.bdate_range('2024-01-02', periods=60)
    panel = pd.MultiIndex.from_product([dates, codes], names=['date', 'code']).to_frame(index=False)

    eds = pd.to_datetime(['2022-12-31', '2023-03-31', '2023-06-30', '2023-09-30',
                          '2023-12-31', '2024-03-31'])
    def mk(cols):
        r = []
        for c in codes:
            for ed in eds:
                row = {'ts_code': c + '.SZ', 'end_date': ed.strftime('%Y%m%d'),
                       'f_ann_date': (ed + pd.Timedelta(days=30)).strftime('%Y%m%d')}
                for k in cols:
                    row[k] = rng.normal(1e9, 2e8)
                r.append(row)
        return pd.DataFrame(r)

    os.makedirs(f'{CACHE}/_partial/daily_basic', exist_ok=True)
    mk(['contract_liab', 'adv_receipts', 'goodwill', 'money_cap', 'total_assets',
        'st_borr', 'lt_borr']).to_parquet(f'{CACHE}/balancesheet.parquet', engine=ENGINE)
    mk(['n_income']).to_parquet(f'{CACHE}/income.parquet', engine=ENGINE)
    mk(['n_cashflow_act', 'c_pay_acq_const_fiolta']).to_parquet(f'{CACHE}/cashflow.parquet', engine=ENGINE)
    db = panel.copy()
    db['ts_code'] = db['code'] + '.SZ'
    db['trade_date'] = db['date'].dt.strftime('%Y%m%d')
    db['total_mv'] = rng.normal(5e5, 1e5, len(db))
    db.to_parquet(f'{CACHE}/_partial/daily_basic/x.parquet', engine=ENGINE)

    fac = build_factors(panel)
    print("因子面板:", fac.shape, "| 列:", list(fac.columns))
    print("非空率:\n", (fac.notna().mean() * 100).round(1).to_string())
    sc = composite(fac)
    print("\ncomposite 非空:", int(sc.notna().sum()), "/", len(sc), "| 范围:",
          round(float(sc.min()), 3), "-", round(float(sc.max()), 3))
    print("\n✅ 逻辑跑通 (PIT/TTM/yoy/composite)。真实评估等三表数据 + 潜伏 fwd/调仓日。")
