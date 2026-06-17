
#import tushare as ts
#pro = ts.pro_api('db04790b0214c9122022fbd224d720e9cfa1fdcccb74edd4216f6bca')
import glob, os
import pandas as pd

files = sorted(glob.glob('tushare_cache/_partial/daily_basic/*.parquet'))
print(f"分片文件数: {len(files)}")
if files:
    print(f"最早文件: {os.path.basename(files[0])}")
    print(f"最晚文件: {os.path.basename(files[-1])}")
    # 读最早和最晚各一片，确认实际日期 + 字段
    first = pd.read_parquet(files[0])
    last = pd.read_parquet(files[-1])
    print(f"最早数据日期: {first['trade_date'].iloc[0]}")
    print(f"最晚数据日期: {last['trade_date'].iloc[0]}")
    print(f"字段里有 PE/PB 吗: {[c for c in first.columns if c in ('pe_ttm','pb','circ_mv')]}")
    # 跨度
    d0 = pd.to_datetime(str(first['trade_date'].iloc[0]))
    d1 = pd.to_datetime(str(last['trade_date'].iloc[0]))
    print(f"覆盖跨度: {(d1-d0).days/365.25:.1f} 年")