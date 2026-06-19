# -*- coding: utf-8 -*-
"""
run_lab_gate.py  ——  对 Lab_ 因子跑预注册 gate（含诚实显著性诊断）。

铁律
----
阈值 **预注册、写死**，绝不在本脚本输出上回调。这是固定阈值的 PASS/FAIL 筛子，
不是调参——和 lab_smart_intraday / lab_upper_shadow 当初被判负用同一把尺。

⚠️ 跑分在**你的机器、你的 lake** 上。我没有你的数据，不替你产出 ICIR/corr 数字。
   __main__ 里是随机噪声 demo，只证明能跑通——不是结果。

⚠️⚠️ 这个 demo 还暴露了一个真问题（见文件末尾 & 我消息里的说明）：
   对**季度更新**的 fundamental 因子，逐日 IC 序列高度自相关（因子一个季度才动一次），
   逐日 ICIR 会把噪声判成 alpha——随机数据都能冲到 |ICIR|≈0.5。所以本脚本除了
   报你预注册的 ICIR，还报 **IC 一阶自相关(IC_ac1)** 和 **Newey-West 调整后的 IC t 值**，
   作为诚实的显著性参考。你 0.25 的门槛对慢因子可能太松——是否收紧由你 pre-register，
   我不替你偷改。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from lab_factors_fundamental import LAB_FUNDAMENTAL_FACTORS, FIELD_MAP

# ============== 预注册阈值（DO NOT TUNE） ==============
ICIR_MIN = 0.25          # |ICIR| >= 0.25  （你预注册的门槛）
CORR_MAX = 0.60          # max |corr 与任一已有因子| < 0.60
HORIZON = 8              # 注意：这是 V25 的 horizon。fundamental 因子是给潜伏(月度/长持)用的，
                         #       建议改成潜伏的评估 horizon，并配 rebalance_dates 去重叠（见下）。
ICIR_WINDOW = 60
NW_LAG = HORIZON         # Newey-West 滞后；前向收益重叠 -> 至少取 horizon。慢因子可调大。
TSTAT_ADVISORY = 1.96    # |t_NW|>=1.96 ≈ p<0.05（仅诊断参考，非你预注册门槛）
# ======================================================

TD = "trade_date"
TC = FIELD_MAP["ts_code"]


def forward_return(close_panel, horizon=HORIZON):
    """从宽表收盘价 [index=trade_date, columns=ts_code] 算前向 horizon 日收益。
    已有 fwd_ret 长表则跳过本函数。"""
    fwd = close_panel.shift(-horizon) / close_panel - 1.0
    out = fwd.stack().rename("fwd_ret").reset_index()
    out.columns = [TD, TC, "fwd_ret"]
    return out


def _ic_series(factor_long, fwd_long, fcol, rebalance_dates=None):
    """逐(评估)日横截面 Spearman rank-IC。
    rebalance_dates 不为 None 时，只在这些日子算 IC —— 这才是潜伏(月度调仓)
    应该看的频率：去掉了日频重叠，独立观测数 ≈ 年数×调仓次数。"""
    m = factor_long.merge(fwd_long, on=[TD, TC], how="inner").dropna(subset=[fcol, "fwd_ret"])
    if rebalance_dates is not None:
        rb = pd.to_datetime(pd.Index(rebalance_dates))
        m = m[m[TD].isin(rb)]
    ics = {}
    for dt, g in m.groupby(TD):
        if g[fcol].nunique() < 5:
            continue
        ic, _ = spearmanr(g[fcol], g["fwd_ret"])
        if np.isfinite(ic):
            ics[dt] = ic
    return pd.Series(ics).sort_index() if ics else pd.Series(dtype=float)


def _icir_full(ic):
    if ic.empty or ic.std(ddof=1) == 0 or not np.isfinite(ic.std(ddof=1)):
        return np.nan
    return ic.mean() / ic.std(ddof=1)


def _icir_roll_med(ic):
    if len(ic) < ICIR_WINDOW:
        return np.nan
    roll = ic.rolling(ICIR_WINDOW).mean() / ic.rolling(ICIR_WINDOW).std(ddof=1)
    return roll.median()


def _ic_ac1(ic):
    """IC 序列一阶自相关。越高 -> 逐日 ICIR 越不可信(有效样本越少)。"""
    if len(ic) < 3:
        return np.nan
    return ic.autocorr(lag=1)


def _nw_tstat(ic, lag=NW_LAG):
    """Newey-West(HAC) 调整后、IC 均值的 t 值。处理重叠/持续带来的自相关。"""
    x = ic.dropna().values
    n = len(x)
    if n < 3:
        return np.nan
    e = x - x.mean()
    var = (e @ e) / n
    L = int(min(lag, n - 1))
    for l in range(1, L + 1):
        w = 1.0 - l / (L + 1.0)
        var += 2.0 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(var / n)
    return x.mean() / se if se > 0 else np.nan


def max_abs_corr(factor_long, existing_wide, fcol):
    """与已有因子的时均横截面 Spearman 相关，取 max|.|。"""
    if existing_wide is None or existing_wide.shape[1] == 0:
        return np.nan, None
    f = factor_long.set_index([TD, TC])[fcol]
    joined = existing_wide.join(f.rename("_lab"), how="inner").dropna(subset=["_lab"])
    best_c, best_name = 0.0, None
    for name in existing_wide.columns:
        sub = joined[[name, "_lab"]].dropna()
        if sub.empty:
            continue
        cs = []
        for _, g in sub.groupby(level=0):
            if len(g) < 5:
                continue
            c, _ = spearmanr(g[name], g["_lab"])
            if np.isfinite(c):
                cs.append(c)
        if cs:
            c = float(np.mean(cs))
            if abs(c) > abs(best_c):
                best_c, best_name = c, name
    return best_c, best_name


def run_gate(raw_tables, panel, fwd_long, existing_wide=None, rebalance_dates=None, nw_lag=NW_LAG):
    """
    raw_tables: dict 键 balancesheet/income/cashflow/daily_basic/holdernumber/holdertrade。
    panel:      [trade_date, ts_code] 票池。
    fwd_long:   [trade_date, ts_code, fwd_ret]。
    existing_wide: 已有因子宽表 index=[trade_date,ts_code]（正交检验；None 跳过）。
    rebalance_dates: 给潜伏用——只在调仓日评估 IC（强烈建议传，否则慢因子 ICIR 虚高）。
    nw_lag:     Newey-West 滞后，单位=评估观测(调仓)数。月度调仓+多月 horizon 时，
                设成 ≈ horizon 的月数（如 6mo→6）以修残余重叠。默认 NW_LAG。
    """
    rows = []
    for name, (fn, needs) in LAB_FUNDAMENTAL_FACTORS.items():
        try:
            fac = fn(*([raw_tables[t] for t in needs] + [panel]))
        except Exception as e:
            rows.append(dict(factor=name, error=f"{type(e).__name__}: {e}")); continue

        ic = _ic_series(fac, fwd_long, name, rebalance_dates=rebalance_dates)
        icir = _icir_full(ic)
        ac1 = _ic_ac1(ic)
        tnw = _nw_tstat(ic, lag=nw_lag)
        corr, cwith = max_abs_corr(fac, existing_wide, name)

        pass_icir = np.isfinite(icir) and abs(icir) >= ICIR_MIN              # 你的预注册门槛
        pass_corr = (existing_wide is None) or (np.isfinite(corr) and abs(corr) < CORR_MAX)
        rows.append(dict(
            factor=name, n_obs=len(ic),
            IC_mean=round(ic.mean(), 4) if len(ic) else np.nan,
            ICIR_full=round(icir, 3) if np.isfinite(icir) else np.nan,
            ICIR_roll_med=round(_icir_roll_med(ic), 3) if np.isfinite(_icir_roll_med(ic)) else np.nan,
            IC_ac1=round(ac1, 2) if np.isfinite(ac1) else np.nan,            # 自相关诊断
            IC_t_NW=round(tnw, 2) if np.isfinite(tnw) else np.nan,           # 诚实显著性
            maxabs_corr=round(corr, 3) if np.isfinite(corr) else np.nan,
            corr_with=cwith,
            PASS_icir=bool(pass_icir), PASS_corr=bool(pass_corr),
            VERDICT_preReg="PASS" if (pass_icir and pass_corr) else "FAIL",
            NW_significant=bool(np.isfinite(tnw) and abs(tnw) >= TSTAT_ADVISORY),
        ))
    df = pd.DataFrame(rows)
    cols = ["factor", "n_obs", "IC_mean", "ICIR_full", "ICIR_roll_med", "IC_ac1",
            "IC_t_NW", "maxabs_corr", "corr_with", "PASS_icir", "PASS_corr",
            "VERDICT_preReg", "NW_significant"]
    return df[[c for c in cols if c in df.columns]]


# ===========================================================================
# DEMO（随机噪声）：演示"慢因子的逐日 ICIR 会虚高"，并不是结果
# ===========================================================================
if __name__ == "__main__":
    print(">>> DEMO on RANDOM noise — proves harness runs & exposes the inflation. NOT a result.\n")
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-01-03", periods=90)
    codes = [f"{i:06d}.SZ" for i in range(40)]
    panel = pd.MultiIndex.from_product([dates, codes], names=[TD, TC]).to_frame(index=False)
    px = pd.DataFrame(rng.lognormal(0, 0.02, (len(dates), len(codes))).cumprod(axis=0),
                      index=dates, columns=codes)
    fwd = forward_return(px)

    def _stmt(eds, cols):
        rows = []
        for c in codes:
            for ed in eds:
                r = {TC: c, "end_date": ed, "f_ann_date": ed + pd.Timedelta(days=30)}
                for k in cols:
                    r[k] = rng.normal(1e9, 2e8)
                rows.append(r)
        return pd.DataFrame(rows)

    eds = pd.to_datetime(["2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30",
                          "2021-12-31", "2022-03-31"])
    raw = dict(
        balancesheet=_stmt(eds, ["contract_liab", "adv_receipts", "goodwill", "money_cap",
                                 "total_assets", "st_borr", "lt_borr"]),
        income=_stmt(eds, ["n_income"]),
        cashflow=_stmt(eds, ["n_cashflow_act", "c_pay_acq_const_fiolta"]),
        holdernumber=_stmt(eds, ["holder_num"]),
        daily_basic=panel.assign(total_mv=rng.normal(5e5, 1e5, len(panel))),
        holdertrade=pd.DataFrame({TC: rng.choice(codes, 20), "f_ann_date": rng.choice(dates, 20),
                                  "in_de": rng.choice(["IN", "DE"], 20),
                                  "change_vol": rng.normal(1e6, 2e5, 20),
                                  "avg_price": rng.normal(20, 3, 20)}),
    )
    rep = run_gate(raw, panel, fwd, existing_wide=None)
    print(rep.to_string(index=False))
    print("\n读法：随机数据下 ICIR_full 仍能到 ±0.3~0.6 且'PASS_icir'=True，")
    print("但 IC_ac1 很高（自相关）、IC_t_NW 接近 0（NW 一调整就现原形）。")
    print("=> 逐日 ICIR 对季度因子虚高；信 IC_t_NW，且应在潜伏调仓日评估(传 rebalance_dates)。")
