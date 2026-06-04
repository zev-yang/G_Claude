"""
portfolio.py — correlation-cluster diversification for final pick selection.
"""
import pandas as pd


def build_sector_clusters(panel, ref_date, lookback_days=60, corr_threshold=0.55):
    """
    Build industry cluster IDs from rolling return correlations.
 
    Stocks with 60-day return correlation > corr_threshold are
    assigned the same cluster ID. This correctly groups PCB stocks
    (002463, 002916, 603228, 600183) into one cluster even though
    their 3-digit code prefixes are different.
 
    Args:
        panel          : the panel DataFrame (date, code) MultiIndex
        ref_date       : the date to compute clusters for
        lookback_days  : rolling window length
        corr_threshold : correlation threshold for same-cluster assignment
 
    Returns:
        dict {code: cluster_id}
    """
    try:
        # Get last N trading days of daily returns
        all_dates = panel.index.get_level_values('date').unique().sort_values()
 
        # BUG1 FIX: use -1 sentinel only for missing date; check explicitly
        # all_dates[-61:0] is an EMPTY slice in Python → must avoid t_loc=-1
        if ref_date not in all_dates:
            return {}
        t_loc = all_dates.get_loc(ref_date)
        if t_loc < lookback_days:
            return {}
 
        date_window = all_dates[max(0, t_loc - lookback_days): t_loc + 1]
        idx = pd.IndexSlice
        sub = panel.loc[idx[date_window, :], 'close'].unstack('code')
 
        # Need stocks with complete data in the window
        sub = sub.dropna(axis=1, thresh=int(lookback_days * 0.8))
        if sub.shape[1] < 10:
            return {}
 
        ret = sub.pct_change().fillna(0)
        corr_mat = ret.corr()
 
        # Greedy clustering: O(N²) but fast for <5000 stocks
        stocks     = list(corr_mat.columns)
        clusters   = {}
        cluster_id = 0
 
        for stock in stocks:
            if stock in clusters:
                continue
            clusters[stock] = cluster_id
            for other in stocks:
                if other not in clusters:
                    if corr_mat.loc[stock, other] >= corr_threshold:
                        clusters[other] = cluster_id
            cluster_id += 1
 
        return clusters
 
    except Exception:
        return {}
 


def diversify_picks(df, score_col='fused_score', top_k=30,
                    max_per_cluster=2, cluster_map=None):
    """
    Select top_k stocks, capping at max_per_cluster per correlation cluster.
 
    If cluster_map is None or empty, falls back to 6-digit code
    (each stock in its own cluster = no cap, standard nlargest).
 
    Args:
        df            : DataFrame sorted by score_col (desc)
        score_col     : column to rank by
        top_k         : max stocks to return
        max_per_cluster: max allowed from any one correlation cluster
        cluster_map   : dict {code: cluster_id} from build_sector_clusters()
 
    Returns:
        DataFrame of diversified top picks
    """
    df = df.copy().sort_values(score_col, ascending=False)
 
    if not cluster_map:
        # No cluster info → no cap, return top_k straight
        return df.head(top_k)
 
    selected, cluster_counts = [], {}
    for idx, _ in df.iterrows():
        code    = str(idx[-1] if isinstance(idx, tuple) else idx)
        cluster = cluster_map.get(code, code)  # fallback: code = own cluster
        count   = cluster_counts.get(cluster, 0)
 
        if count < max_per_cluster:
            selected.append(idx)
            cluster_counts[cluster] = count + 1
 
        if len(selected) >= top_k:
            break
 
    return df.loc[selected] if selected else df.head(top_k)
