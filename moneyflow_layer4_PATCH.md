# Layer-4 资金流 overlay — 集成补丁

配套文件（单独给出，均已在合成数据上做 ship 级验证）：
- **`moneyflow_factors.py`**（整文件替换）— 在上次基础上新增 3 个 Layer-4 helper：`mf_strength_8`、`mf_accel`、`sm_outflow_rate`。
- **`fetch_moneyflow_extra.py`**（新文件）— 下载 `moneyflow_dc` / `moneyflow_ind_dc`(行业+概念) / `daily_basic` 到 `tushare_cache/_partial/`，复用你 `fetch_tushare.py` 的断点续传范式。

本文件 = `config.py` / `factors.py` / `backtest.py` / `run.py` 的 FIND→REPLACE 补丁。

**口径锁定**：`mf_score = 0.6·z(mf_strength_8) + 0.2·z(mf_accel) + 0.2·retail_contrary`；叠加 `final = zscore(base) + MF_WEIGHT·mf_score`，**不进 LGBM**；分母用「当日总买入额」(ratio-of-sums)；阈值/权重**全写死**；A/B 只作 go/no-go 观察。Layer-1/2/3 **defer**（config 里留 False 开关）。`USE_MONEYFLOW` 与现有 `moneyflow_role`(screen) / `mf_dipbuy` **相互独立**。

---

## 0) 先跑数据下载（你在 G49_Claude 目录执行）

```powershell
$env:TUSHARE_TOKEN="<your token>"
python fetch_moneyflow_extra.py
```
- `moneyflow_dc` 历史 ~2023-09 起，脚本会**自动二分跳过空前缀**；早于该日的个股流仍用现有 raw `moneyflow`。
- ⚠ `moneyflow_dc` / `moneyflow_ind_dc` 需足够 Tushare 积分；若持续报权限错说明账号未开通该接口（Layer-4 现在**不依赖**这两个，可先不管，等做 Layer-1/2/3 再开）。
- **Layer-4 现在只用 raw `moneyflow`**（你已有），所以即使上面下载没跑，下面的因子也能立刻回测。

---

## 1) `config.py` — Layer-4 开关（全写死）

**FIND**：
```python
    "results_dir": "./results_v25_1_production",
```
**REPLACE WITH**：
```python
    # ════════ Layer-4 资金流 overlay (线性叠加, 不喂 LGBM) ════════════════════════════
    # 全部写死、point-in-time、A/B 只作 go/no-go 观察 (绝不在全样本上调 MF_WEIGHT 等参数)。
    "USE_MONEYFLOW": True,        # 总开关; False=不叠加 (与现有 screen/mf_dipbuy 独立)
    "MF_WEIGHT": 0.15,           # 写死: final = zscore(base) + MF_WEIGHT·mf_score
    "MF_W_STRENGTH": 0.6,        # 写死: mf_score 内 strength 权重
    "MF_W_ACCEL": 0.2,           # 写死: accel 权重 (mf_accel = strength_8 - strength_20)
    "MF_W_RETAIL": 0.2,          # 写死: retail_contrary 权重
    "RETAIL_CONTRARY_ENABLE": True,
    "RETAIL_CONTRARY_PERCENTILE": 0.90,  # 写死: 小单净流出【当日横截面】前 10%
    "RETAIL_CONTRARY_VOL_RATIO": 1.2,    # 写死: 放量倍数 (当日量 / 20日均量)
    "RETAIL_CONTRARY_BIAS_LIMIT": 0.15,  # 写死: |close/ma20 - 1| 上限 (乖离约束)
    # ── DEFERRED (数据/择时未就绪, 留 False 以备后用) ───────────────────────────────
    "USE_MARKET_POSITION": False,  # Layer-1 大盘仓位 (需 moneyflow_dc; 归到"晚点升 V9")
    "MF_INDUSTRY_ENABLE": False,   # Layer-2 行业加分 (需 moneyflow_ind_dc + 个股→行业映射)
    "MF_CONCEPT_ENABLE": False,    # Layer-3 概念加分 (默认关)
    "results_dir": "./results_v25_1_production",
```

---

## 2) `factors.py` — 构建 `mf_score`（插在 Cleanup 之前）

和 `mf_dipbuy` 一样，`mf_score` 需要 join 后的 moneyflow helper + 价格（`close/volume/pct_chg`），所以放在 moneyflow `join` 之后、`# ── Cleanup ──`（删 `pct_chg`）之前。**若你已应用上次的 `mf_dipbuy` 补丁，本块紧跟其后即可。**

**INSERT**：在 `# ── Cleanup ──` 那段**正上方**插入：
```python
            # ── NEW: Layer-4 资金流 overlay 打分列 mf_score (线性叠加, 不喂 LGBM) ────────
            # mf_score = w_s·z(mf_strength_8) + w_a·z(mf_accel) + w_r·retail_contrary。
            # 三分量全 point-in-time; 阈值/权重全写死。mf_score 作为【独立打分列】保留 (不入
            # self.factors), 在 backtest/run 里以 final = zscore(base) + MF_WEIGHT·mf_score 叠加。
            # 无 moneyflow 覆盖 -> mf_strength_8 为 NaN -> mf_score NaN -> 叠加时按 0 (无 tilt)。
            if CONFIG.get('USE_MONEYFLOW', False) and 'mf_strength_8' in df.columns:
                # retail_contrary (需价): A 当日上涨, B 放量(>VOL_RATIO×20日均量),
                # C 乖离 |close/ma20-1|<BIAS_LIMIT; 在满足 A&B&C 的票里取【小单净流出】当日横截面前 10%。
                if (CONFIG.get('RETAIL_CONTRARY_ENABLE', True)
                        and {'close', 'volume', 'pct_chg'}.issubset(df.columns)):
                    _vr  = CONFIG.get('RETAIL_CONTRARY_VOL_RATIO', 1.2)
                    _bl  = CONFIG.get('RETAIL_CONTRARY_BIAS_LIMIT', 0.15)
                    _pcc = CONFIG.get('RETAIL_CONTRARY_PERCENTILE', 0.90)
                    _ma20c = (df['close'].sort_index(level=['code', 'date']).groupby(level='code')
                              .transform(lambda s: s.rolling(20, min_periods=10).mean()).reindex(df.index))
                    _ma20v = (df['volume'].sort_index(level=['code', 'date']).groupby(level='code')
                              .transform(lambda s: s.rolling(20, min_periods=10).mean()).reindex(df.index))
                    _A = df['pct_chg'] > 0
                    _B = df['volume'] > (_vr * _ma20v)
                    _C = (df['close'] / _ma20c - 1.0).abs() < _bl
                    _cond = _A & _B & _C
                    _tmp = df['sm_outflow_rate'].where(_cond)         # 仅在 A&B&C 票内排名
                    _rk  = _tmp.groupby(level='date').rank(pct=True)
                    df['retail_contrary'] = ((_rk > _pcc) & _cond).astype('float32')
                else:
                    df['retail_contrary'] = np.float32(0.0)
                # 当日横截面 z-score (clip ±3); NaN(未覆盖) 保留
                def _csz(_col):
                    _g = df.groupby('date')[_col]
                    return ((df[_col] - _g.transform('mean')) / (_g.transform('std') + 1e-9)).clip(-3, 3)
                _ws = CONFIG.get('MF_W_STRENGTH', 0.6)
                _wa = CONFIG.get('MF_W_ACCEL', 0.2)
                _wr = CONFIG.get('MF_W_RETAIL', 0.2)
                df['mf_score'] = (_ws * _csz('mf_strength_8')
                                  + _wa * _csz('mf_accel')
                                  + _wr * df['retail_contrary']).astype('float32')
                print(f"   ...mf_score built (overlay w={_ws}/{_wa}/{_wr}); "
                      f"coverage {int(df['mf_score'].notna().sum()):,} rows")

```

**然后**把 Cleanup 删除清单**追加 4 个 Layer-4 中间列**（`mf_score` / `main_net_3d` / `sm_net_3d` 必须**保留**）。在 `# ── Cleanup ──` 的 `for tmp_col in [...]` 列表里加入：
```python
                            'mf_strength_8', 'mf_accel', 'sm_outflow_rate', 'retail_contrary',
```
> 例如，如果你现在的清单是 `['amplitude', 'pct_chg', 'illiq', 'is_limit_down', 'is_big_cap', 'elg_net_rate']`，改成在末尾追加上面四个。**不要删 `mf_score`、`main_net_3d`、`sm_net_3d`。**

---

## 3) `backtest.py` — 回测叠加 + check_ic 观察 mf_score

### 3a) 叠加（紧跟现有融合块之后、screen 之前）

**FIND**：
```python
            else:
                valid['fused_score'] = lgbm_scores
```
**REPLACE WITH**：
```python
            else:
                valid['fused_score'] = lgbm_scores
            # ── Layer-4 资金流 overlay: final = zscore(base) + MF_WEIGHT·mf_score (NOT in LGBM) ──
            if CONFIG.get('USE_MONEYFLOW', False) and 'mf_score' in valid.columns:
                valid['fused_score'] = (_zscore(valid['fused_score'])
                                        + CONFIG.get('MF_WEIGHT', 0.15) * valid['mf_score'].fillna(0.0))
```

### 3b) check_ic 末尾观察 mf_score 自身 IC + 分层多空（纯展示）

**FIND**（check_ic 里的这行；上次决策①重叠度块也加在它后面，本块再往后追加即可）：
```python
        print(pd.DataFrame(stats).sort_values('ICIR', ascending=False).to_string(index=False))
```
**REPLACE WITH**：
```python
        print(pd.DataFrame(stats).sort_values('ICIR', ascending=False).to_string(index=False))

        # ── Layer-4 观察: mf_score 自身 IC + 分层多空 (纯展示, 不参与选股/调参) ──────────
        # target = 残差化后的前向收益【排名】, 故 spread 是 rank 单位的单调性信号。
        if 'mf_score' in valid_y.columns:
            _mic, _msp = [], []
            for _d, _g in valid_y.groupby(level='date'):
                _gg = _g.dropna(subset=['mf_score', 'target'])
                if len(_gg) < 10:
                    continue
                _mic.append(_gg['mf_score'].corr(_gg['target'], method='spearman'))
                _q = _gg['mf_score'].rank(pct=True)
                _msp.append(_gg.loc[_q > 0.8, 'target'].mean() - _gg.loc[_q < 0.2, 'target'].mean())
            if _mic:
                _mic = np.array(_mic, dtype=float); _msp = np.array(_msp, dtype=float)
                print(f"\n💰 [Layer-4 mf_score] 自身 IC={np.nanmean(_mic):.4f} "
                      f"ICIR={np.nanmean(_mic) / (np.nanstd(_mic) + 1e-9):.2f} "
                      f"| 分层多空(Top20%-Bot20%)={np.nanmean(_msp):.4f} "
                      f"(IC/spread≈0 → overlay 无增量, 考虑 USE_MONEYFLOW=False)")
```

---

## 4) `run.py` — 生产叠加（与回测一致）

**FIND**：
```python
            today_df['score'] = fused_prod # 覆盖为融合分数
```
**REPLACE WITH**：
```python
            today_df['score'] = fused_prod # 覆盖为融合分数
            # ── Layer-4 资金流 overlay (与回测一致): final = zscore(base) + MF_WEIGHT·mf_score ──
            if CONFIG.get('USE_MONEYFLOW', False) and 'mf_score' in today_df.columns:
                today_df['score'] = (_zscore(today_df['score'])
                                     + CONFIG.get('MF_WEIGHT', 0.15) * today_df['mf_score'].fillna(0.0))
```

---

## A/B 怎么跑

| 想测 | config.py |
|---|---|
| 不叠加资金流 (baseline，承接你上次的冠军组合) | `"USE_MONEYFLOW": False` |
| 叠加 Layer-4 资金流 | `"USE_MONEYFLOW": True` |

跑回测后看新增的 **`💰 [Layer-4 mf_score]`** 行：
- **自身 IC / ICIR**：mf_score 单独对 target 的选股力（≈0 说明没信息）。
- **分层多空 Top20%-Bot20%**：>0 且单调才算有效（对应你 §2.6.2 的分层多空、§2.6.4 的独立验证）。
- 再对比 ON vs OFF 两次回测的 **CAGR / Sharpe / ExCAGR / 换手率**（diagnostics 那套指标），决定是否保留。**MF_WEIGHT 写死 0.15，不在回测上调**（§2.6.3 那个"按 p 值调权重"的循环按违规处理）。

## 要盯的风险旗标
1. **覆盖率**：`mf_score built ... coverage N rows`，N 太小说明很多票没 moneyflow，叠加退化成纯 LGBM。
2. **retail_contrary 太稀疏**：A&B&C 三条同时满足的票本就不多，再取前 10%，命中可能极少 → 该分量大部分时间是 0，贡献有限（这是设计本身的特性，不是 bug）。
3. **scale**：base 已做 z-score，所以 MF_WEIGHT=0.15 ≈ 给资金流约 15% 的相对话语权；想调强弱改这个数（但**别在回测上拟合它**）。

---

## ⚠ 一个需要你拍板的前后矛盾（重要）
你这次说「**moneyflow 因子喂进 LGBM 会拖累预测**」——但**上次**我们把 `mf_dipbuy` 加进了 `self.factors`（即喂进了 LGBM）。按你这次的新结论，`mf_dipbuy` 很可能也在拖后腿。

两个选择：
- **(A) 把 mf_dipbuy 也移出 LGBM**：从 `self.factors.append('mf_dipbuy')` 去掉（或直接 `dipbuy_enable=False`），保持「moneyflow 一律不进 LGBM」的一致性；
- **(B) 保留 mf_dipbuy 作为独立 A/B**：维持现状，单独对照它在/不在 LGBM 的回测差异。

我**倾向 (A)**（与你刚验证的结论一致，少一个拖累项）。你定，我跟着改。
