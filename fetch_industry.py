# -*- coding: utf-8 -*-
"""
fetch_industry.py — 个股申万一级行业分类 (Tushare stock_basic.industry). 【潜伏模式中性化用】

用途: 潜伏模式 Layer5 合成后要做【行业中性化】(把 hidden_alpha 减去同行业均值, 避免选出来
  全是某个超跌行业)。中性化需要每只股票的行业归属。

接口选择 (简单够用): stock_basic 的 industry 字段就是申万一级行业, 一次调用拉全市场,
  每股一个标签, 秒级完成。它是【当前】静态归属, 不含历史变更 —— 但一级行业归属极少变,
  对中性化精度足够。若日后要时点准确的历史行业, 再升级到 index_member_all (带生效日期)。

输出: 单文件 tushare_cache/_partial/industry/stock_industry.parquet
  列: code(6位), ts_code, name, industry(申万一级), market, list_date
  含全部上市状态(L/D/P), 剔北交所, 与 universe 一致。
"""
import os

import pandas as pd

import fetch_moneyflow_extra as F

OUT_DIR_DEFAULT = './tushare_cache/_partial/industry'
OUT_FILE = 'stock_industry.parquet'


def _out_dir():
    try:
        from config import CONFIG
        return CONFIG.get('industry_path', OUT_DIR_DEFAULT)
    except Exception:
        return OUT_DIR_DEFAULT


def update_industry(pro=None, out_dir=None):
    pro = pro or F.get_pro()
    out_dir = out_dir or _out_dir()
    os.makedirs(out_dir, exist_ok=True)

    frames = []
    for status in ('L', 'D', 'P'):       # 上市/退市/暂停 (防幸存者偏差, 与财务 fetcher 一致)
        try:
            df = F.call(pro, 'stock_basic', exchange='', list_status=status,
                        fields='ts_code,name,industry,market,list_date')
            if df is not None and not df.empty:
                frames.append(df)
        except Exception as e:
            print(f"  [industry] {status} 拉取失败: {e!r}")

    if not frames:
        print("  [industry] SKIP: stock_basic 无数据 (检查权限)")
        return

    allf = pd.concat(frames, ignore_index=True)
    allf = allf[allf['ts_code'].str[-3:].isin(['.SZ', '.SH'])]      # 剔北交所
    allf['code'] = allf['ts_code'].astype(str).str[:6]
    allf = allf.drop_duplicates(subset='code', keep='first')
    n_ind = allf['industry'].notna().sum()
    allf[['code', 'ts_code', 'name', 'industry', 'market', 'list_date']].to_parquet(
        os.path.join(out_dir, OUT_FILE), engine='fastparquet', index=False)
    print(f"  [industry] ✅ {len(allf)} 只股票, {n_ind} 只有行业标签, "
          f"{allf['industry'].nunique()} 个申万一级行业 -> {OUT_FILE}")


def load_industry(src=None):
    """读行业表 -> code -> industry 的 Series (供中性化用)。"""
    src = src or _out_dir()
    path = os.path.join(src, OUT_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 不存在 — 先跑 fetch_industry.py")
    df = pd.read_parquet(path, engine='fastparquet')
    return df.set_index('code')['industry']


if __name__ == '__main__':
    update_industry()
