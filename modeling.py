"""
modeling.py — model construction and fitting, centralised so the backtest and the
production path stay identical.

Toggle CONFIG['use_ranker']:
  True  -> LGBMRanker (lambdarank), trained with one query group per trade date,
           which optimises the *ranking* of stocks within each day (what we use).
  False -> LGBMRegressor on the continuous rank target (original behaviour).
"""
import numpy as np
import lightgbm as lgb

from config import CONFIG

# Shared hyper-parameters (unchanged from the original tuning).
_LGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=5,
    num_leaves=31,
    min_data_in_leaf=50,
    reg_lambda=10,
    reg_alpha=2,
    colsample_bytree=0.7,
    subsample=0.8,
    verbose=-1,
    random_state=42,
    n_jobs=-1,
)

# The [0,1] rank target is bucketed into this many integer relevance grades for the
# ranker. Linear label_gain (0..N-1) avoids the exponential default and the
# "label out of range" error.
N_GRADES = 32


def build_model():
    """Return an LGBMRanker if CONFIG['use_ranker'] else an LGBMRegressor."""
    if CONFIG.get('use_ranker', True):
        return lgb.LGBMRanker(
            objective='lambdarank',
            label_gain=list(range(N_GRADES)),
            **_LGB_PARAMS,
        )
    return lgb.LGBMRegressor(**_LGB_PARAMS)


def build_models():
    """Seed-bagging ensemble (列表; USE_BAGGING=False 时退化为单模型, 行为与原先逐位一致).

    每个成员 = 完全相同的超参, 只换 random_state。多样性来自已有的
    colsample_bytree=0.7 (逐树随机抽特征列, 由 seed 驱动) — 不新增/不改任何超参,
    种子写死, 不在回测上调。目的: 压掉单模型的拟合方差 (variance reduction), 非寻找 alpha。
    """
    if not CONFIG.get('USE_BAGGING', False):
        return [build_model()]
    models = []
    for s in CONFIG.get('BAG_SEEDS', [42, 202, 777]):
        p = dict(_LGB_PARAMS, random_state=int(s))
        if CONFIG.get('use_ranker', True):
            models.append(lgb.LGBMRanker(objective='lambdarank',
                                         label_gain=list(range(N_GRADES)), **p))
        else:
            models.append(lgb.LGBMRegressor(**p))
    return models


def predict_ensemble(models, X):
    """成员各自预测 -> 各自转百分位 rank -> 取均值 (对分数尺度差异稳健).

    单成员时 rank(pct) 是分数的单调变换 — 下游只用排序/zscore, 结果不变。
    回测/生产里 X 都是单一信号日的横截面, 所以全局 rank == 当日 rank。
    """
    import pandas as pd
    ranks = [pd.Series(m.predict(X)).rank(pct=True).to_numpy() for m in models]
    return np.mean(ranks, axis=0)


def fit_model(model, X, y, sample_weight, dates):
    """
    Fit `model` on (X, y, sample_weight).

    For the ranker, `dates` is the per-row trade-date index aligned with X. It MUST
    be in ascending-date order (the panel is date-sorted, so X already is); group
    sizes are then one per date, in that same order. The continuous [0,1] rank
    target is bucketed into integer relevance grades.
    """
    if CONFIG.get('use_ranker', True):
        y_grade = np.floor(np.asarray(y, dtype=float) * (N_GRADES - 1e-6)).astype(int)
        np.clip(y_grade, 0, N_GRADES - 1, out=y_grade)
        # np.unique returns counts in ascending-date order, matching X's date blocks.
        _, group_sizes = np.unique(np.asarray(dates), return_counts=True)
        model.fit(X, y_grade, group=group_sizes, sample_weight=sample_weight)
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model
