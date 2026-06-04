# strategy_v25_1 — modular layout

Same system as the single-file `strategy_v25_1.py`, split into focused modules so a
future change touches one small file instead of the whole script. **No logic was changed**
in this split — it's the exact code from the upgraded single file (all eight `UPGRADE Edit`
fixes included), just reorganized. Two constants that used to be duplicated were centralized.

## Files (import direction flows downward — nothing imports `run`)

| File | Contents | Imports from |
|------|----------|--------------|
| `config.py` | `CONFIG`, constants (`RED/GREEN/RESET`), `_zscore`, `Timer`, **`FAMILY_MAP`**, **`GROUPS_V3`** | — |
| `data_loader.py` | `load_single_robust`, `load_universe_audit` | config |
| `safety.py` | `check_market_safety_v9` | — |
| `factors.py` | `AlphaLabV25_1` (feature engineering) | config |
| `logic_matrix.py` | `LogicMatrixPredictorV5`, `IntegratedAuditorV5` | — |
| `portfolio.py` | `build_sector_clusters`, `diversify_picks` | — |
| `backtest.py` | `DailyAuditor` (walk-forward backtest) | config, safety, logic_matrix, portfolio |
| `run.py` | `deep_clean_memory` + the `__main__` orchestration (load → features → backtest → production) | all of the above |

## How to run (no IDE needed)

Put all eight files in the **same folder**, then either:

- terminal: `python run.py`
- Jupyter: `%run run.py`  (a single cell)

Both set `__name__ == "__main__"`, so the orchestration block executes.

## What changed vs. the single file (organization only)

1. **`FAMILY_MAP`** used to be defined inside `__main__` but was *used* inside
   `DailyAuditor.run_simulation`. That only worked by accident of global scope. It now lives
   in `config.py` and is imported where needed — the dependency is explicit and robust.
2. **`GROUPS_V3`** existed as two copies (one in the backtest, one in production). They were
   verified identical, so there is now a single definition in `config.py`.
3. The dead imports (`scipy.optimize.minimize`, `sklearn ... LedoitWolf`, unused `scipy.stats`
   names) were dropped. The Jupyter print-rate setting and `warnings.filterwarnings('ignore')`
   are preserved in `config.py`.

## Editing in future

To change, say, the hedge logic, you (or I) edit only `safety.py` and replace that one file
via GitHub's web editor. The toggles and thresholds you'll most often touch are all in
`config.py`.
