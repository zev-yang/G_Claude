# -*- coding: utf-8 -*-
"""
verify_adjustment.py — 实证检验数据湖 (stock_data_all) 是否为【未复权】价格。

检验一 (无需网络): 除权缺口扫描
  主板/中小板非ST股票涨跌停 ±10% -> 若数据已复权, 相邻收盘涨跌幅不可能 < -11%。
  未复权数据在送转/配股除权日会出现 -15% ~ -55% 的机械跳空 (10转10 => -50%)。
  抓到此类缺口 = 未复权铁证。注: 现金分红缺口通常只有 -1%~-3%, 藏在正常波动里,
  本检验抓的是送转/配股这类大缺口。

检验二 (可选, 需 TUSHARE_TOKEN): 与 Tushare 三方对账
  随机抽 3 只股票较早的历史段:
    湖价格 vs Tushare daily(未复权)  -> 应【一致】(同源同口径)
    湖价格 vs Tushare pro_bar(qfq)   -> 应【不一致】(若一致说明湖已是前复权)

用法: python verify_adjustment.py
"""
import glob
import os

import numpy as np
import pandas as pd

LAKE = './stock_data_all'


def scan_exright_gaps():
    files = glob.glob(os.path.join(LAKE, '*.csv'))
    if not files:
        raise SystemExit(f"找不到数据湖: {LAKE}")
    print(f"检验一: 扫描 {len(files)} 个文件中的除权缺口 ...")
    hits = []
    for fp in files:
        code = os.path.basename(fp)[:-4]
        if code[:2] not in ('60', '00'):          # 只看 ±10% 板 (60 沪主板 / 00 深主板+中小板)
            continue
        try:
            d = pd.read_csv(fp, usecols=['date', 'close', 'name'])
        except Exception:
            continue
        if len(d) < 2:
            continue
        if d['name'].astype(str).str.contains('ST').any():   # ST 历史另有 5% 限制, 跳过避免误判
            continue
        chg = d['close'].pct_change()
        bad = d[(chg < -0.11)]
        for _, r in bad.iterrows():
            hits.append((code, str(r['name']), r['date'],
                         float(chg.loc[r.name])))
    hits = pd.DataFrame(hits, columns=['code', 'name', 'date', 'drop']).sort_values('drop')
    print(f"\n  发现疑似除权缺口 (主板非ST, 单日 < -11%): {len(hits)} 个")
    if len(hits):
        print("  最极端 15 例 (送转除权的典型形态: -20% ~ -55%):")
        print(hits.head(15).to_string(index=False,
              formatters={'drop': lambda x: f'{x:.1%}'}))
        recent = hits[hits['date'] >= '2024-05-01']
        print(f"\n  其中落在回测窗口内 (2024-05 之后): {len(recent)} 个")
        print("\n  ✅ 结论: 数据为【未复权】原始价 — 复权数据不可能出现超跌停缺口。")
        print("     这些缺口正在污染 target 与动量类因子 (持有人并未真亏, 模型却按暴跌学习)。")
    else:
        print("  未发现超限缺口 — 不能就此断定已复权 (送转近年减少), 请继续检验二交叉对账。")
    return hits


def cross_check_tushare():
    tok = os.environ.get('TUSHARE_TOKEN')
    if not tok:
        print("\n检验二: 跳过 (未设 TUSHARE_TOKEN 环境变量)。检验一已足够下结论。")
        return
    try:
        import tushare as ts
        pro = ts.pro_api(tok)
        files = sorted(glob.glob(os.path.join(LAKE, '6005*.csv')))[:3] or \
                sorted(glob.glob(os.path.join(LAKE, '*.csv')))[:3]
        print("\n检验二: 与 Tushare 对账 (历史段前 60 个交易日)")
        for fp in files:
            code = os.path.basename(fp)[:-4]
            ts_code = code + ('.SH' if code.startswith('6') else '.SZ')
            lake = pd.read_csv(fp, usecols=['date', 'close'])
            lake = lake[lake['date'] >= '2024-06-01'].head(60)
            if lake.empty:
                continue
            s, e = lake['date'].iloc[0].replace('-', ''), lake['date'].iloc[-1].replace('-', '')
            raw = pro.daily(ts_code=ts_code, start_date=s, end_date=e)
            qfq = ts.pro_bar(ts_code=ts_code, adj='qfq', start_date=s, end_date=e)
            m_raw = lake.merge(raw.assign(date=pd.to_datetime(raw['trade_date']).dt.strftime('%Y-%m-%d')),
                               on='date', suffixes=('_lake', '_ts'))
            m_qfq = lake.merge(qfq.assign(date=pd.to_datetime(qfq['trade_date']).dt.strftime('%Y-%m-%d')),
                               on='date', suffixes=('_lake', '_ts'))
            d_raw = (m_raw['close_lake'] - m_raw['close_ts']).abs().max()
            d_qfq = (m_qfq['close_lake'] - m_qfq['close_ts']).abs().max()
            print(f"  {code}: |湖−未复权| max={d_raw:.4f} (应≈0) | |湖−前复权| max={d_qfq:.4f} (有分红送转则应>0)")
    except Exception as e:
        print(f"\n检验二失败 ({e!r}) — 不影响检验一的结论。")


if __name__ == '__main__':
    scan_exright_gaps()
    cross_check_tushare()
