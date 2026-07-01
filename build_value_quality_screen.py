# -*- coding: utf-8 -*-
"""
build_value_quality_screen.py — 集中选股【候选短名单 + 决策信息卡】: value便宜 ∩ F-score健康 + 质量/流动性信息列。

定位: 研究工具, 非机械组合。保证"便宜+财务健康"下限(避雷), 最终挑10-15只靠你判断。
预注册(写死): 域=全市场剔ST+剔circ_mv最小30%; 候选=域内最便宜value decile(10%) ∩ F-score>=5。
  【纪律】新增三列只作决策卡【信息】, 不叠硬门槛、不做行业择时 —— 由你挑票时参考。
决策卡: PE/PB自身5年分位、value域内排名、F分+7项、负债率、现金流、ROA、流通市值、
  + 日均成交额20d(流动性,标<5000万) + 现金流为正年数(近3年,0-3) + 分红年数(近3年,0-3)。
复用 eval_value_longhist / eval_fscore 口径。全程 fastparquet。
"""
import datetime as dt
import glob
import os

import numpy as np
import pandas as pd

import eval_value_longhist as V

FUND_DIR = './tushare_cache/_partial/fundamentals'
CF_DIR = './tushare_cache/_partial/cashflow'
DIV_DIR = './tushare_cache/_partial/dividend'
EXCL_SMALL_PCT = 0.30
CHEAP_PCT = 0.10
FSCORE_HEALTHY = 5
YEARS = (2023, 2024, 2025)
AMT_MIN_YI = 0.5
OUT_CSV = './value_quality_screen.csv'
CRIT = ['ROA>0', 'CFO>0', 'ΔROA>0', 'CFO>NI', 'Δ负债<0', 'Δ毛利>0', 'Δ周转>0']


def value_pcts(db):
    d = db.set_index(['date', 'code']).sort_index()
    out = {}
    for col in ('pe_ttm', 'pb'):
        w = d[col].where(d[col] > 0).unstack('code').sort_index()
        out[col] = w.rolling(V.PCT_WIN, min_periods=V.PCT_MINP).rank(pct=True).iloc[-1]
    return out['pe_ttm'], out['pb']


def avg_amount_20d():
    amt = V._read('daily', ['amount'])
    w = amt.set_index(['date', 'code'])['amount'].unstack('code').sort_index()
    return w.iloc[-20:].mean() / 1e5   # 千元 -> 亿元


def cashflow_pos_years():
    parts = []
    for f in glob.glob(f'{CF_DIR}/*.parquet'):
        per = os.path.splitext(os.path.basename(f))[0]
        if per.endswith('1231') and per[:4].isdigit() and int(per[:4]) in YEARS:
            try:
                parts.append(pd.read_parquet(f, columns=['ts_code', 'end_date', 'n_cashflow_act'], engine='fastparquet'))
            except Exception:
                pass
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts, ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['yr'] = df['end_date'].astype(str).str[:4]
    df = df.drop_duplicates(['code', 'yr'], keep='last')
    df['pos'] = pd.to_numeric(df['n_cashflow_act'], errors='coerce') > 0
    return df.groupby('code')['pos'].sum().astype(int)


def dividend_years():
    fs = glob.glob(f'{DIV_DIR}/*.parquet')
    parts = []
    for f in fs:
        try:
            parts.append(pd.read_parquet(f, engine='fastparquet'))
        except Exception:
            pass
    if not parts:
        return pd.Series(dtype=float)
    df = pd.concat(parts, ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    if 'div_proc' in df.columns:
        df = df[df['div_proc'].astype(str).str.contains('实施', na=False)]
    cc = 'cash_div_tax' if 'cash_div_tax' in df.columns else ('cash_div' if 'cash_div' in df.columns else None)
    if cc is None:
        return pd.Series(dtype=float)
    df['cash'] = pd.to_numeric(df[cc], errors='coerce')
    df = df[df['cash'] > 0]
    dc = 'ex_date' if 'ex_date' in df.columns else 'ann_date'
    df['yr'] = pd.to_datetime(df[dc].astype(str), errors='coerce').dt.year
    df = df[df['yr'].isin(YEARS)]
    return df.groupby('code')['yr'].nunique()


def build_fscore_detail(today):
    fs = sorted(glob.glob(f'{FUND_DIR}/*.parquet'))
    df = pd.concat([pd.read_parquet(f, engine='fastparquet') for f in fs], ignore_index=True)
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
    df['end'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
    for c in ['roa', 'ocfps', 'ocf_to_profit', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ann', 'end', 'roa']).sort_values(['code', 'end'])
    g = df.groupby('code')
    cc = {'ROA>0': df['roa'] > 0, 'CFO>0': df['ocfps'] > 0,
          'ΔROA>0': (df['roa'] - g['roa'].shift(4)) > 0, 'CFO>NI': df['ocf_to_profit'] > 1,
          'Δ负债<0': (df['debt_to_assets'] - g['debt_to_assets'].shift(4)) < 0,
          'Δ毛利>0': (df['grossprofit_margin'] - g['grossprofit_margin'].shift(4)) > 0,
          'Δ周转>0': (df['assets_turn'] - g['assets_turn'].shift(4)) > 0}
    for k, v in cc.items():
        df[k] = v.astype(int)
    df['fscore'] = df[list(cc)].sum(axis=1)
    sub = df[df['ann'] <= today]
    return sub.loc[sub.groupby('code')['end'].idxmax()].set_index('code')[['fscore'] + CRIT + ['debt_to_assets', 'ocfps', 'roa']]


def main():
    print("① 读长历史 + 构建 中性化value / 5年分位 / F-score / 质量信息列 ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind, name_map = ind_df['industry'], ind_df['name']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])

    cur = value_raw.index.get_level_values('date').max()
    today = pd.Timestamp(dt.date.today())
    print(f"   数据至 {cur.date()}\n")

    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, [cur])).xs(cur, level='date').dropna()
    pe_pct, pb_pct = value_pcts(db)
    cm = circ.xs(cur)
    amt20 = avg_amount_20d()
    cfy = cashflow_pos_years()
    divy = dividend_years()
    print(f"   质量列覆盖: 现金流年数 {len(cfy)} 股 | 分红年数 {len(divy)} 股 | 成交额 {amt20.notna().sum()} 股")

    domain = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    domain = [c for c in domain if c not in st and c in value_neu.index]
    vd = value_neu.reindex(domain).dropna()
    vd_rank = vd.rank(pct=True)
    cheap = vd[vd >= vd.quantile(1 - CHEAP_PCT)].index
    print(f"② 域 {len(domain)} 只 -> 最便宜decile {len(cheap)} 只")

    fdet = build_fscore_detail(today)
    cand = [c for c in cheap if c in fdet.index and fdet.loc[c, 'fscore'] >= FSCORE_HEALTHY]
    print(f"③ 候选 = 便宜 ∩ F-score>={FSCORE_HEALTHY}(健康): {len(cand)} 只\n")
    if not cand:
        raise SystemExit("无候选")

    rows = []
    for c in cand:
        f = fdet.loc[c]
        mark = ''.join('✓' if f[k] else '·' for k in CRIT)
        a = float(amt20.get(c, np.nan))
        rows.append({
            '代码': c, '名称': str(name_map.get(c, c)), '行业': str(ind.get(c, '')),
            'PE分位': round(float(pe_pct.get(c, np.nan)), 2), 'PB分位': round(float(pb_pct.get(c, np.nan)), 2),
            'value便宜排名': round(float(vd_rank.get(c, np.nan)), 2),
            'F分': int(f['fscore']), '7项明细': mark,
            '成交额亿20d': round(a, 2) if not np.isnan(a) else np.nan,
            '流动性提示': '⚠<5000万' if (not np.isnan(a) and a < AMT_MIN_YI) else '',
            '现金流正年数': int(cfy.get(c, 0)), '分红年数': int(divy.get(c, 0)),
            '负债率': round(float(f['debt_to_assets']), 1),
            '现金流ocfps': round(float(f['ocfps']), 2), 'ROA': round(float(f['roa']), 1),
            '流通市值亿': round(float(cm.get(c, np.nan)) / 1e4, 1),
        })
    out = pd.DataFrame(rows).sort_values('value便宜排名', ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f"=== 候选短名单 ({len(out)} 只, 按便宜度排序; 7项: {'/'.join(CRIT)}) ===")
    with pd.option_context('display.max_rows', None, 'display.width', 240, 'display.unicode.east_asian_width', True):
        print(out.head(40).to_string(index=False))
    if len(out) > 40:
        print(f"\n  (仅显示最便宜40只; 完整 {len(out)} 只见 {OUT_CSV})")
    print(f"\n💾 已存 {OUT_CSV}")
    print("\n用法 (这是候选池, 非持仓单; 三个新列是【信息】不是硬门槛):")
    print("  · 便宜: PE/PB分位低 + value便宜排名高。健康: F分高(5-7)+7项。")
    print("  · 流动性: 成交额亿20d, '⚠<5000万'的票你的资金可能买卖困难, 优先避开。")
    print("  · 质量避雷: 现金流正年数(近3年,越接近3越好=持续造血)、分红年数(越接近3越好=真金白银派现,利润可信)。")
    print("  · 行业扎堆自己看着分散(人工判断, 未写进规则)。从中挑10-15只你看得懂的集中持有。")


if __name__ == '__main__':
    main()
