"""Quick benchmark: parallel BnB at d=8 target=1.20 with vs without
the H_d cut, to verify the ~2x speedup from the proper sigma reduction.

Runs the cut-disabled version first (monkeypatching box_outside_hd to
return False), then the cut-enabled version. Compares closed_vol /
total_volume and node count.
"""
from __future__ import annotations

import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def run(label, *, disable_hd_cut: bool):
    """Run parallel BnB once. If disable_hd_cut, monkeypatch to no-op."""
    # Re-import freshly for clean state.
    import importlib
    import interval_bnb.symmetry as sym_mod
    import interval_bnb.parallel as par_mod
    import interval_bnb.bnb as bnb_mod
    importlib.reload(sym_mod)
    importlib.reload(par_mod)
    importlib.reload(bnb_mod)
    if disable_hd_cut:
        # Monkeypatch the H_d filter to a no-op so the BnB falls back to
        # the loose mu_0 <= 1/2 cut only (the previous behaviour).
        sym_mod.box_outside_hd = lambda B: False
        par_mod.box_outside_hd = lambda B: False
        bnb_mod.box_outside_hd = lambda B: False

    from interval_bnb.parallel import parallel_branch_and_bound
    t0 = time.time()
    result = parallel_branch_and_bound(
        d=8,
        target_c="1.20",
        workers=4,
        init_split_depth=10,
        donate_threshold_floor=2,
        time_budget_s=60.0,
        verbose=False,
    )
    dt = time.time() - t0
    success = result.get("success", result.get("certified", False)) if isinstance(result, dict) else getattr(result, "success", False)
    print(f"[{label}] success={success}  wall={dt:.1f}s  result_keys={list(result.keys()) if isinstance(result, dict) else 'obj'}")
    if isinstance(result, dict):
        for k in ("nodes", "node_count", "leaves_certified", "cert_count", "max_depth"):
            if k in result:
                print(f"  {k} = {result[k]}")
    return result, dt


if __name__ == "__main__":
    # Baseline: H_d cut disabled (loose mu_0 <= 1/2 only).
    print("=" * 60)
    print("BASELINE: H_d cut DISABLED (loose mu_0 <= 1/2 only)")
    print("=" * 60)
    r_off, t_off = run("OFF", disable_hd_cut=True)

    # H_d cut enabled (proper coordinate-coupled mu_0 <= mu_{d-1}).
    print()
    print("=" * 60)
    print("H_d CUT ENABLED (mu_0 <= mu_{d-1})")
    print("=" * 60)
    r_on, t_on = run("ON", disable_hd_cut=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"baseline:    wall={t_off:.1f}s")
    print(f"H_d cut on:  wall={t_on:.1f}s")
    if t_off > 0:
        print(f"speedup:     {t_off / t_on:.2f}x")
