# -*- coding: utf-8 -*-
"""
build_value_quality_screen.py — 集中选股【候选短名单 + 决策信息卡】: value便宜 ∩ F-score健康。

定位(重要): 这不是机械组合策略, 是给你10万账户做集中选股的研究工具。
  · value(已验证16年t=8, 组合级真alpha): 5年PE/PB分位, 越便宜越高 -> 保证"便宜"下限。
  · F-score(7项健康度, 稳健性检验定性): 阈值/成本两关过、滚动增量方向正但偏弱 -> 不进组合, 但去陷阱效果稳
    (回撤-10.6%->-4.8~-6.3%三阈值都降、严过滤下子期全正、抗成本) -> 作"避雷"健康下限。
  -> 名单保证"便宜+财务健康"下限(客观避雷), 最终挑哪10-15只靠你的判断(你的真实优势)。

预注册(写死, 不调): 域=全市场剔ST+剔circ_mv最小30%(同生产组合已验证域);
  候选 = 域内最便宜 value decile(10%) ∩ F-score>=5(健康); 按便宜度排序输出。
每只附决策卡: PE/PB自身5年分位、value域内排名、F分+7项明细、负债率、经营现金流、ROA、流通市值。
复用 eval_value_longhist 构建块 + eval_fscore 的F-score口径。全程 fastparquet。
"""
import datetime as dt
import glob

import numpy as np
import pandas as pd

import eval_value_longhist as V

FUND_DIR = './tushare_cache/_partial/fundamentals'
EXCL_SMALL_PCT = 0.30          # 剔流通市值最小30%
CHEAP_PCT = 0.10               # 便宜池: value 最便宜 decile
FSCORE_HEALTHY = 5             # 健康下限 (0-7; >=5 财务健康)
OUT_CSV = './value_quality_screen.csv'
CRIT = ['ROA>0', 'CFO>0', 'ΔROA>0', 'CFO>NI', 'Δ负债<0', 'Δ毛利>0', 'Δ周转>0']


def value_pcts(db):
    """返回最新日 PE/PB 各自5年分位 (低=便宜), Series indexed by code。"""
    d = db.set_index(['date', 'code']).sort_index()
    out = {}
    for col in ('pe_ttm', 'pb'):
        w = d[col].where(d[col] > 0).unstack('code').sort_index()
        pct = w.rolling(V.PCT_WIN, min_periods=V.PCT_MINP).rank(pct=True)
        out[col] = pct.iloc[-1]
    return out['pe_ttm'], out['pb']


def build_fscore_detail(today):
    """as-of today, 每股最新一期财报的 F-score + 7项明细 + 关键原始值。"""
    fs = sorted(glob.glob(f'{FUND_DIR}/*.parquet'))
    if not fs:
        raise SystemExit(f"无 {FUND_DIR}")
    df = pd.concat([pd.read_parquet(f, engine='fastparquet') for f in fs], ignore_index=True)
    need = ['roa', 'ocfps', 'ocf_to_profit', 'debt_to_assets', 'grossprofit_margin', 'assets_turn']
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise SystemExit(f"fundamentals 缺 {miss}")
    df['code'] = df['ts_code'].astype(str).str[:6]
    df['ann'] = pd.to_datetime(df['ann_date'].astype(str), errors='coerce')
    df['end'] = pd.to_datetime(df['end_date'].astype(str), errors='coerce')
    for c in need:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['ann', 'end', 'roa']).sort_values(['code', 'end'])
    g = df.groupby('code')
    crit_cols = {
        'ROA>0': df['roa'] > 0,
        'CFO>0': df['ocfps'] > 0,
        'ΔROA>0': (df['roa'] - g['roa'].shift(4)) > 0,
        'CFO>NI': df['ocf_to_profit'] > 1,
        'Δ负债<0': (df['debt_to_assets'] - g['debt_to_assets'].shift(4)) < 0,
        'Δ毛利>0': (df['grossprofit_margin'] - g['grossprofit_margin'].shift(4)) > 0,
        'Δ周转>0': (df['assets_turn'] - g['assets_turn'].shift(4)) > 0,
    }
    for k, v in crit_cols.items():
        df[k] = v.astype(int)
    df['fscore'] = df[list(crit_cols)].sum(axis=1)
    sub = df[df['ann'] <= today]
    idx = sub.groupby('code')['end'].idxmax()         # 每股最新一期
    det = sub.loc[idx].set_index('code')
    return det[['fscore'] + CRIT + ['debt_to_assets', 'ocfps', 'roa', 'end']]


def main():
    print("① 读长历史 + 构建 中性化 value / 5年PE-PB分位 / F-score明细 ...")
    db = V._read('daily_basic', ['pe_ttm', 'pb', 'circ_mv'])
    value_raw = V.build_value(db[['date', 'code', 'pe_ttm', 'pb']])
    lncap = np.log(db.set_index(['date', 'code'])['circ_mv'].clip(lower=1)).rename('lncap')
    circ = db.set_index(['date', 'code'])['circ_mv'].sort_index()
    ind_df = pd.read_parquet(V.IND_FILE, engine=V.ENGINE).drop_duplicates('code').set_index('code')
    ind, name_map = ind_df['industry'], ind_df['name']
    st = set(ind_df.index[ind_df['name'].astype(str).str.contains('ST')])

    dates = value_raw.index.get_level_values('date').unique().sort_values()
    cur = dates.max()
    today = pd.Timestamp(dt.date.today())
    print(f"   数据至 {cur.date()}\n")

    # 当期 value (中性化排名 + 原始分位)
    value_neu = V._xs_rank(V.neutralize(value_raw, ind, lncap, [cur])).xs(cur, level='date').dropna()
    pe_pct, pb_pct = value_pcts(db)
    cm = circ.xs(cur)

    # 域: 剔ST + 剔最小30%市值
    domain = cm[cm >= cm.quantile(EXCL_SMALL_PCT)].index
    domain = [c for c in domain if c not in st and c in value_neu.index]
    vd = value_neu.reindex(domain).dropna()
    vd_rank = vd.rank(pct=True)                         # 域内便宜度 (1=最便宜)
    cheap = vd[vd >= vd.quantile(1 - CHEAP_PCT)].index  # 最便宜 decile
    print(f"② 域 {len(domain)} 只 (剔ST+剔小市值) | 最便宜 decile {len(cheap)} 只")

    # F-score 健康过滤
    fdet = build_fscore_detail(today)
    print(f"   F-score 覆盖 {len(fdet)} 只 | as-of {today.date()}")
    cand = [c for c in cheap if c in fdet.index and fdet.loc[c, 'fscore'] >= FSCORE_HEALTHY]
    print(f"③ 候选 = 便宜 ∩ F-score>={FSCORE_HEALTHY}(健康): {len(cand)} 只\n")
    if not cand:
        raise SystemExit("无候选 — 检查 fundamentals 覆盖 / 放宽 FSCORE_HEALTHY")

    # 决策卡
    rows = []
    for c in cand:
        f = fdet.loc[c]
        mark = ''.join('✓' if f[k] else '·' for k in CRIT)
        rows.append({
            '代码': c, '名称': str(name_map.get(c, c)), '行业': str(ind.get(c, '')),
            'PE分位': round(float(pe_pct.get(c, np.nan)), 2),
            'PB分位': round(float(pb_pct.get(c, np.nan)), 2),
            'value便宜排名': round(float(vd_rank.get(c, np.nan)), 2),
            'F分': int(f['fscore']), '7项明细': mark,
            '负债率': round(float(f['debt_to_assets']), 1),
            '现金流ocfps': round(float(f['ocfps']), 2),
            'ROA': round(float(f['roa']), 1),
            '流通市值亿': round(float(cm.get(c, np.nan)) / 1e4, 1),
        })
    out = pd.DataFrame(rows).sort_values(['value便宜排名'], ascending=False).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    print(f"=== 候选短名单 ({len(out)} 只, 按便宜度排序; 7项: {'/'.join(CRIT)}) ===")
    show = out.head(40)
    with pd.option_context('display.max_rows', None, 'display.width', 200, 'display.unicode.east_asian_width', True):
        print(show.to_string(index=False))
    if len(out) > 40:
        print(f"\n  (仅显示最便宜40只; 完整 {len(out)} 只见 {OUT_CSV})")
    print(f"\n💾 已存 {OUT_CSV}")
    print("\n用法: 这是【候选池】不是【持仓单】。从中挑你看得懂的 10-15 只集中持有:")
    print("  · PE/PB分位低 = 相对自身历史便宜; value便宜排名高 = 域内最便宜的一批。")
    print("  · F分高(5-7) + 7项看健康在哪/弱在哪; 负债率低、现金流(ocfps)>0、ROA>0 = 财务扎实。")
    print("  · 名单已保证'便宜+健康'客观下限(避雷); 最终选择靠你对生意的理解。")
    print("  · 谨记: 集中持有靠的是你的研究深度, 不是breadth; 这工具帮你缩小尽调范围、筛掉陷阱。")


if __name__ == '__main__':
    main()
