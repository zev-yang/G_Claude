"""
diagnostics.py — performance decomposition across hedge x logic-fusion combinations.

Runs the walk-forward backtest four times on the SAME in-memory panel
(load + feature engineering happen once), flipping CONFIG['enable_hedge'] and
CONFIG['enable_logic_fusion'] between runs, and prints a comparison table:

    (hedge OFF, logic OFF)  -> raw model alpha, the honest baseline
    (hedge OFF, logic ON )  -> does the LogicMatrix overlay add or subtract?
    (hedge ON , logic OFF)  -> does the safety filter add or subtract?
    (hedge ON , logic ON )  -> the full current system

Read the table this way: if the raw-alpha row is weak and only the hedged rows
look good, the edge is the (overfit) filter, not the model. A large 'Flat' count
means many periods were blocked/skipped and booked as 0.0 return, which mechanically
compresses volatility and inflates Sharpe — so compare Sharpe alongside Flat.

Run standalone:   python diagnostics.py
Or from a notebook, reusing an already-loaded panel:
    from diagnostics import run_diagnostics
    res = run_diagnostics(panel, feats)        # panel/feats from your earlier run
"""
import io
import sys
import contextlib

import numpy as np
import pandas as pd

from config import CONFIG
from data_loader import load_universe_audit
from factors import AlphaLabV25_1
from backtest import DailyAuditor


# ---------------------------------------------------------------------------
# Metrics (mirror DailyAuditor.analyze so the full-system row reconciles with
# your existing Step 5 output, plus traded/flat/hit breakdown for attribution).
# ---------------------------------------------------------------------------
def _ret_stats(s: pd.Series, h: int):
    """Annualized CAGR, Sharpe, and MaxDD for a per-period return series."""
    s = s.dropna().astype(float)
    if len(s) == 0:
        return float('nan'), float('nan'), float('nan')
    af = 252.0 / h
    cagr = s.mean() * af
    sharpe = cagr / (s.std() * np.sqrt(af) + 1e-9)
    eq = (1.0 + s).cumprod()
    maxdd = float((eq / eq.cummax() - 1.0).min())
    return float(cagr), float(sharpe), maxdd


def _metrics(logs: pd.DataFrame) -> dict:
    h = CONFIG['horizon']

    s = logs['Strat'].astype(float)
    n_periods = int(len(s))

    # A blocked/skipped period is booked as exactly 0.0; a real trade almost never is.
    traded = s[s != 0.0]
    n_traded = int(len(traded))
    n_flat = n_periods - n_traded

    s_cagr, s_sharpe, s_maxdd = _ret_stats(s, h)
    b = logs['Bench'] if 'Bench' in logs else pd.Series(dtype=float)
    b_cagr, b_sharpe, b_maxdd = _ret_stats(b, h)

    hit = float((traded > 0).mean()) if n_traded else float('nan')
    avg_trade = float(traded.mean()) if n_traded else float('nan')

    return dict(n_periods=n_periods, n_traded=n_traded, n_flat=n_flat,
                cagr=s_cagr, sharpe=s_sharpe, maxdd=s_maxdd,
                hit=hit, avg_trade=avg_trade,
                bench_cagr=b_cagr, bench_sharpe=b_sharpe, bench_maxdd=b_maxdd,
                excess_cagr=(s_cagr - b_cagr))


@contextlib.contextmanager
def _maybe_silence(enabled: bool):
    """Redirect stdout to a buffer so the per-period selection reports (which use
    print / tqdm.write -> stdout) are hidden during each run. The tqdm progress
    bar writes to stderr and stays visible."""
    if not enabled:
        yield
        return
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        yield
    finally:
        sys.stdout = old


def _print_table(rows: list):
    if not rows:
        print("No results to display.")
        return

    head = (f"{'Combo':<22}{'Periods':>9}{'Traded':>8}{'Flat':>6}"
            f"{'CAGR':>9}{'Sharpe':>8}{'MaxDD':>9}{'Hit':>7}{'Avg/Trd':>9}{'ExCAGR':>10}")
    bar = "=" * len(head)
    print("\n" + bar)
    print("PERFORMANCE DECOMPOSITION  (hedge x logic-fusion)")
    print(bar)
    print(head)
    print("-" * len(head))
    for r in rows:
        hit = "  n/a" if np.isnan(r['hit']) else f"{r['hit']:.0%}"
        avg = "    n/a" if np.isnan(r['avg_trade']) else f"{r['avg_trade']:.2%}"
        print(f"{r['label']:<22}{r['n_periods']:>9}{r['n_traded']:>8}{r['n_flat']:>6}"
              f"{r['cagr']:>9.2%}{r['sharpe']:>8.2f}{r['maxdd']:>9.2%}"
              f"{hit:>7}{avg:>9}{r['excess_cagr']:>10.2%}")
    print(bar)
    # Benchmark = filtered universe, equal-weight. It's ~identical across combos
    # (blocked periods still log Bench), so show it once from the fullest-sample row.
    ref = max(rows, key=lambda r: r['n_periods'])
    print(f"Benchmark (filtered universe, equal-weight): "
          f"CAGR={ref['bench_cagr']:.2%}  Sharpe={ref['bench_sharpe']:.2f}  MaxDD={ref['bench_maxdd']:.2%}")
    print("ExCAGR = strategy CAGR - benchmark CAGR  (selection alpha; negative = lagged the universe).")
    print("Baseline row = (hedge OFF, logic OFF) = raw model alpha.")
    print("High 'Flat' inflates Sharpe via zero-return periods — weigh Sharpe against Flat.\n")


def run_diagnostics(panel=None, feats=None, quiet=True, combos=None, csv_path=None):
    """
    panel, feats : reuse an already-loaded/engineered panel if provided; otherwise
                   load + engineer once here.
    quiet        : hide the verbose per-period reports during each run (default True).
    combos       : list of (enable_hedge, enable_logic_fusion) tuples to test.
    csv_path     : if given, write the results table to this CSV.
    """
    if panel is None or feats is None:
        print("Loading data lake + engineering features (once)...")
        panel = load_universe_audit(CONFIG['stock_data_path'])
        eng = AlphaLabV25_1()
        panel, feats = eng.run(panel)

    auditor = DailyAuditor(panel, feats)

    if combos is None:
        combos = [(False, False), (False, True), (True, False), (True, True)]

    saved = (CONFIG['enable_hedge'], CONFIG['enable_logic_fusion'])
    rows = []
    try:
        for hedge, fusion in combos:
            CONFIG['enable_hedge'] = hedge
            CONFIG['enable_logic_fusion'] = fusion
            label = f"hedge={'ON ' if hedge else 'OFF'} | logic={'ON ' if fusion else 'OFF'}"
            print(f"\n>> Running: {label}")
            try:
                with _maybe_silence(quiet):
                    logs, _ = auditor.run_simulation()
            except Exception as e:
                print(f"   ERROR in combo ({label}): {e!r}")
                continue
            if logs is None or len(logs) == 0:
                print(f"   (no periods produced — check data window / audit_start)")
                continue
            m = _metrics(logs)
            m['label'] = label
            rows.append(m)
            print(f"   periods={m['n_periods']} traded={m['n_traded']} flat={m['n_flat']} "
                  f"| CAGR={m['cagr']:.2%} Sharpe={m['sharpe']:.2f} "
                  f"MaxDD={m['maxdd']:.2%} hit={'n/a' if np.isnan(m['hit']) else format(m['hit'],'.0%')}")
    finally:
        # Always restore the user's original toggles, even on error.
        CONFIG['enable_hedge'], CONFIG['enable_logic_fusion'] = saved

    _print_table(rows)

    res_df = pd.DataFrame(rows,
                          columns=['label', 'n_periods', 'n_traded', 'n_flat',
                                   'cagr', 'sharpe', 'maxdd', 'hit', 'avg_trade',
                                   'excess_cagr', 'bench_cagr', 'bench_sharpe', 'bench_maxdd'])
    if csv_path:
        res_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"Saved decomposition table to {csv_path}")
    return res_df


if __name__ == "__main__":
    run_diagnostics()
