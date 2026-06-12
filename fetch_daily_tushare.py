"""
fetch_daily_tushare.py — A股日线增量 (Tushare `daily`, doc_id=27) 直接续写进 TDX 数据湖。

目标: run_data_update.py 一条命令把【行情】也更新掉, 不再单独跑 TDX notebook。
落盘格式与 TDX 下载器逐字节兼容 — stock_data_all/{code}.csv, 9 列:
    date,open,high,low,close,volume,amount,code,name
    · code   无后缀 6 位 (Tushare 的 '000001.SZ' -> '000001')
    · date   'YYYY-MM-DD'
    · amount 元 (Tushare 千元 ×1000)
    · volume 单位【运行时自动校准】: 取湖里最新一天与 Tushare 同日成交量对比,
      中位比值 ≈1 -> 湖与 Tushare 同为手; ≈100 -> 湖为股, Tushare 手×100。
      绝不拍脑袋假设 — 单位接缝错 100 倍会打穿所有滚动量比因子。
    · 价格   两边都是未复权原始价 (doc_id=27 明示"本接口是未复权行情"), 可直接拼接。

范围边界: 本脚本只做【增量】(湖最新日期 -> 今天)。空湖/全量回填仍是 TDX 下载器的活
(Tushare 单次 6000 条, 拉 2000 年至今全市场要 6000+ 次调用, 不划算)。

只保留 .SZ/.SH (剔除北交所 .BJ — TDX 湖里本来就没有, 保持 universe 一致)。
新股 (湖里没有的 code) 自动建新文件, 名称从 stock_basic 取, 取不到则用 code 兜底。
发布时滞: daily 在交易日 15~16 点入库; 遇到第一个空日期就【停】(绝不跳过造成中间空洞),
下次运行自动续上 — 与 run_data_update 的 RECENT_GUARD 哲学一致。
"""
import os
import glob
import datetime as dt

import numpy as np
import pandas as pd

import fetch_moneyflow_extra as F   # 复用 get_pro / call(分页+重试)

LAKE_DIR_DEFAULT = './stock_data_all'
LAKE_COLS = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'code', 'name']
_CAL_N = 30          # 用多少只股票做成交量单位校准
_ACCEPT = {1.0: (0.85, 1.15), 100.0: (85.0, 115.0)}   # 接受的比值带


def _lake_dir():
    try:
        from config import CONFIG
        p = CONFIG.get('stock_data_path', LAKE_DIR_DEFAULT)
        return os.path.dirname(p) if p.endswith('.csv') else p
    except Exception:
        return LAKE_DIR_DEFAULT


def _scan_lake(lake_dir):
    """逐文件读最后一行 (seek 尾部, 与 TDX 下载器同法) ->
    {code: dict(last_date='YYYY-MM-DD', volume=float, name=str)}"""
    state = {}
    for path in glob.glob(os.path.join(lake_dir, '*.csv')):
        code = os.path.basename(path)[:-4]
        if not (len(code) == 6 and code.isdigit()):
            continue
        try:
            with open(path, 'rb') as f:
                try:
                    f.seek(-400, os.SEEK_END)
                except OSError:
                    f.seek(0)
                lines = [l for l in f.readlines() if l.strip()]
            if not lines:
                continue
            parts = lines[-1].decode('utf-8', errors='ignore').strip().split(',')
            if len(parts) < 7 or len(parts[0]) != 10 or not parts[0][:4].isdigit():
                continue
            state[code] = dict(last_date=parts[0],
                               volume=float(parts[5]) if parts[5] else 0.0,
                               name=parts[8].strip().strip('"') if len(parts) >= 9 else code)
        except Exception:
            continue
    return state


def _tushare_to_lake(df, vol_scale):
    """Tushare daily 行 -> 湖 schema (除 name 外); 单位换算 + TDX 同款清洗。"""
    out = pd.DataFrame({
        'date':   pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d').dt.strftime('%Y-%m-%d'),
        'open':   pd.to_numeric(df['open'],  errors='coerce').fillna(0).round(4),
        'high':   pd.to_numeric(df['high'],  errors='coerce').fillna(0).round(4),
        'low':    pd.to_numeric(df['low'],   errors='coerce').fillna(0).round(4),
        'close':  pd.to_numeric(df['close'], errors='coerce').fillna(0).round(4),
        'volume': (pd.to_numeric(df['vol'], errors='coerce').fillna(0) * vol_scale)
                  .round(0).astype('int64'),
        'amount': (pd.to_numeric(df['amount'], errors='coerce').fillna(0) * 1000.0).round(2),
        'code':   df['ts_code'].astype(str).str[:6],
    })
    return out


def _calibrate_volume_scale(pro, state):
    """用湖最新一天 vs Tushare 同日成交量的中位比值, 判定 vol 换算倍数 (1 或 100)。"""
    frontier = max(v['last_date'] for v in state.values())
    f_yyyymmdd = frontier.replace('-', '')
    ts_day = F.call(pro, 'daily', trade_date=f_yyyymmdd)
    if ts_day.empty:
        raise RuntimeError(f"calibration failed: tushare daily({f_yyyymmdd}) returned empty")
    ts_day = ts_day[ts_day['ts_code'].str[-3:].isin(['.SZ', '.SH'])]
    ts_vol = dict(zip(ts_day['ts_code'].str[:6], pd.to_numeric(ts_day['vol'], errors='coerce')))

    ratios = []
    for code, st in state.items():
        if st['last_date'] != frontier or st['volume'] <= 0:
            continue
        tv = ts_vol.get(code)
        if tv and tv > 0:
            ratios.append(st['volume'] / tv)
        if len(ratios) >= _CAL_N:
            break
    if len(ratios) < 5:
        raise RuntimeError(f"calibration failed: only {len(ratios)} overlapping stocks on {frontier}")
    med = float(np.median(ratios))
    for scale, (lo, hi) in _ACCEPT.items():
        if lo <= med <= hi:
            unit = '手 (×1)' if scale == 1.0 else '股 (Tushare 手 ×100)'
            print(f"  [daily->lake] volume 校准: 湖/Tushare 中位比值 {med:.2f} -> 湖单位={unit}")
            return scale
    raise RuntimeError(
        f"volume 校准失败: 中位比值 {med:.2f} 既不接近 1 也不接近 100 — "
        f"湖数据单位异常, 拒绝写入以免污染 (样本 {len(ratios)} 只, 日期 {frontier})")


def _stock_names(pro):
    """新股名称 map; 拿不到权限就 {} (兜底 name=code)。"""
    try:
        sb = F.call(pro, 'stock_basic', fields='ts_code,name')
        return dict(zip(sb['ts_code'].astype(str).str[:6], sb['name'].astype(str)))
    except Exception as e:
        print(f"  [daily->lake] stock_basic unavailable ({e!r}) -> 新股名称用 code 兜底")
        return {}


def update_daily_lake(pro=None, lake_dir=None):
    pro = pro or F.get_pro()
    lake_dir = lake_dir or _lake_dir()
    if not os.path.isdir(lake_dir):
        print(f"  [daily->lake] SKIP: 数据湖目录不存在 ({lake_dir}) — 先用 TDX 下载器建全量历史")
        return
    state = _scan_lake(lake_dir)
    if not state:
        print(f"  [daily->lake] SKIP: 湖为空 — 增量无锚点, 全量回填请用 TDX 下载器")
        return

    frontier = max(v['last_date'] for v in state.values())
    today = dt.datetime.now().strftime('%Y%m%d')
    start = (dt.datetime.strptime(frontier, '%Y-%m-%d') + dt.timedelta(days=1)).strftime('%Y%m%d')
    if start > today:
        print(f"  [daily->lake] 已是最新 (frontier {frontier})")
        return

    cal = F.call(pro, 'trade_cal', exchange='', start_date=start, end_date=today, is_open='1')
    days = sorted(cal['cal_date'].astype(str).tolist()) if not cal.empty else []
    if not days:
        print(f"  [daily->lake] 已是最新 (frontier {frontier}, 其后无交易日)")
        return
    print(f"  [daily->lake] frontier {frontier} -> 待补 {len(days)} 个交易日 ({days[0]}..{days[-1]})")

    vol_scale = _calibrate_volume_scale(pro, state)
    names = _stock_names(pro)

    chunks = []
    for d in days:                                  # 升序; 第一个空日即停, 保证无中间空洞
        df = F.call(pro, 'daily', trade_date=d)
        if df.empty:
            print(f"  [daily->lake] {d} 尚未入库 (15~16点发布) — 到此为止, 下次运行续传")
            break
        df = df[df['ts_code'].str[-3:].isin(['.SZ', '.SH'])]   # 剔除北交所, 与湖 universe 一致
        chunks.append(_tushare_to_lake(df, vol_scale))
    if not chunks:
        return
    new = pd.concat(chunks, ignore_index=True)

    appended = created = rows = 0
    for code, grp in new.groupby('code'):
        st = state.get(code)
        if st is not None:
            grp = grp[grp['date'] > st['last_date']]           # 防重复 append
        if grp.empty:
            continue
        grp = grp.sort_values('date').copy()
        grp['name'] = (st['name'] if st else names.get(code, code))
        path = os.path.join(lake_dir, f"{code}.csv")
        is_new = st is None
        grp[LAKE_COLS].to_csv(path, mode='a' if not is_new else 'w',
                              header=is_new, index=False, encoding='utf-8-sig')
        rows += len(grp)
        appended += (not is_new)
        created += is_new
    print(f"  [daily->lake] ✅ 追加 {rows:,} 行 -> 更新 {appended:,} 只 / 新建 {created} 只 "
          f"(覆盖到 {new['date'].max()})")


if __name__ == '__main__':
    update_daily_lake()
