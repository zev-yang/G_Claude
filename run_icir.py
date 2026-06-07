"""
run_icir.py — fast per-factor ICIR + redundancy screen. NO backtest.

It runs the exact same load + feature-engineering path as run_test.py
(AlphaLabV25_1().run, so the 4 new 主力-footprint factors and the
residual-label / regime settings from config.py are all applied), then:

  1. prints the GLOBAL per-factor IC / ICIR table — by calling the existing
     DailyAuditor.check_ic(), so the numbers match what the backtest's
     selection tournament uses, just whole-sample instead of rolling-60d; and
  2. prints, for the 4 new factors, their strongest cross-sectional rank
     correlation to every other factor — a redundancy screen.

Runtime ≈ one feature-engineering pass + two correlation sweeps (minutes).
None of the per-window LGBMRanker fitting that made the 4h backtest slow.

Drop this next to run_test.py and run:

    python run_icir.py
"""
from config import CONFIG, FAMILY_MAP
from data_loader import load_universe_audit
from factors import AlphaLabV25_1
from backtest import DailyAuditor

NEW_FACTORS = ['downside_rs', 'accum_trend', 'coil',
               'mf_cum20', 'mf_trend', 'elg_cum20']   # moneyflow factors now wired in


def main():
    print("Loading data lake + engineering features (once)...")
    panel = load_universe_audit(CONFIG['stock_data_path'])
    panel, feats = AlphaLabV25_1().run(panel)
    print(f"   panel rows={len(panel):,}  factors={len(feats)}  "
          f"residualize_label={CONFIG.get('residualize_label')}")

    auditor = DailyAuditor(panel, feats)

    # --- 1. Global IC / ICIR (their own proven function) -------------------
    # Table is sorted by ICIR desc. Reference bars from backtest.py:
    #   MIN_ICIR_FLOOR = 0.08  -> the rolling tournament discards factors below this
    #   icir_threshold = 0.15  -> the "stable" initial-screen standard (config.py)
    # NOTE: with residualize_label=True the target is the SIZE-NEUTRAL forward
    # return, so this ICIR is each factor's signal *after* size is stripped out
    # (a size-correlated factor will read lower here than its raw-return IC).
    auditor.check_ic()

    # --- 2. Redundancy screen for the 4 new factors -----------------------
    present_new = [f for f in NEW_FACTORS if f in feats]
    missing = [f for f in NEW_FACTORS if f not in feats]
    if missing:
        print(f"\n  not in feats (won't be scored): {missing}")
    if not present_new:
        return

    print("\n" + "=" * 64)
    print("REDUNDANCY SCREEN  —  new factors vs all others")
    print("avg cross-sectional rank corr (pooled within-date percentile ranks)")
    print("=" * 64)

    ranked = panel[feats].groupby(level='date').rank(pct=True)
    corr = ranked.corr()

    for f in present_new:
        others = corr[f].drop(labels=[f]).dropna()
        top = others.reindex(others.abs().sort_values(ascending=False).index).head(3)
        print(f"\n{f}  [{FAMILY_MAP.get(f, '?')}]")
        for g, c in top.items():
            print(f"    {c:+.2f}  {g:<18} [{FAMILY_MAP.get(g, '?')}]")

    print("\nRead: |corr| > ~0.7 with an existing factor = near-duplicate signal;")
    print("keep the higher-ICIR one. (The backtest culls correlated factors at")
    print("CORR_THRESHOLD = 0.62, but mainly within a group — each new factor is")
    print("its own family, so cross-family overlap can slip through. Watch it here.)")


if __name__ == "__main__":
    main()
