"""
run_test.py — quick single-combo diagnostic.

Runs ONLY (hedge OFF, logic ON) — your best config — instead of all four
combinations, so it finishes in roughly a quarter of the time. It loads the data
lake and engineers features once (including the new accumulation/distribution
factors and the residual-label / regime / ranker settings from config.py), then
prints the decomposition table for that one combo.

Run it from the folder that holds the other modules:

    python run_test.py
"""
from config import CONFIG
from diagnostics import run_diagnostics

if __name__ == "__main__":
    # OPTIONAL fast screen: uncomment the next line to use the regressor instead of
    # the ranker (~3x faster) just to check whether the new factors move things,
    # then re-comment it for the real number.
    # CONFIG["use_ranker"] = False

    # Both hedge-OFF combos (the hedge is dead, so we skip it):
    #   (False, False) = pure model  -> the CLEAN read on whether the new factors help
    #   (False, True)  = + logic tilt -> confirms the overlay still helps with them
    run_diagnostics(combos=[(False, False), (False, True)])
