"""Work-stealing parallel branch-and-bound.

Design (see plan file `now-lets-focus-on-iridescent-lollipop.md`):

* Each worker has a PRIVATE DFS stack (fast, in-process, preserves
  rank-1 cache locality between parent and child).
* A SHARED `mp.Queue` holds unclaimed boxes. Workers pull a BATCH when
  their stack is empty and donate the SHALLOW (oldest) HALF of their
  stack when `len(stack) >= donate_trigger()`. Batched IPC amortises
  pickle overhead.
* Termination is Safra-style: a shared `in_flight` counter tracks the
  number of boxes that have been created but NOT YET CLOSED (certified
  or declared infeasible). When a box splits into two children, the
  counter increments by +1 (one input removed, two created). When a box
  certifies or is infeasible, -1. `in_flight == 0` is a one-time
  monotone event observable without races.
* `mp.get_context("fork")` lets workers inherit the pre-built
  `A_tensor` via copy-on-write on Linux.

Correctness: identical boxes get processed (just on different cores),
rigor gate (Fraction replay) is per-box and process-local. The single
"silent failure" risk -- min_box_width hit without certification -- is
broadcast via an `mp.Event` that every worker polls at every loop
iteration.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
import signal
import sys
import time
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from interval_bnb.box import Box  # noqa: E402
from interval_bnb.symmetry import half_simplex_cuts  # noqa: E402


# ---------------------------------------------------------------------
# Starter partition
# ---------------------------------------------------------------------

def _split_initial(d: int, depth: int, sym_cuts) -> List[Box]:
    """Breadth-first midpoint split of the initial (half-)simplex box
    `depth` times. Drops sub-boxes that do not intersect the simplex.
    """
    root = Box.initial(d, sym_cuts)
    boxes: List[Box] = [root]
    for _ in range(depth):
        new_boxes: List[Box] = []
        for B in boxes:
            ax = B.widest_axis()
            left, right = B.split(ax)
            if left.intersects_simplex():
                new_boxes.append(left)
            if right.intersects_simplex():
                new_boxes.append(right)
        boxes = new_boxes
    return boxes


# ---------------------------------------------------------------------
# Worker-side globals (set once per process on fork/spawn)
# ---------------------------------------------------------------------

_W_STATE: dict = {}  # filled in _worker_main on first entry


def _worker_main(
    worker_id: int,
    d: int,
    target_c,
    min_box_width: float,
    max_nodes_per_worker: int,
    pull_batch_max: int,
    donate_threshold_floor: int,
    # Shared primitives -- pickled across fork/spawn.
    queue: mp.Queue,
    in_flight: "mp.sharedctypes.Synchronized",
    idle_count: "mp.sharedctypes.Synchronized",
    cert_count: "mp.sharedctypes.Synchronized",
    node_count: "mp.sharedctypes.Synchronized",
    closed_vol: "mp.sharedctypes.Synchronized",  # volume of closed boxes
    failed_event: "mp.synchronize.Event",
    done_event: "mp.synchronize.Event",
    stats_queue: mp.Queue,
):
    """Long-lived worker: pull from shared queue, process local DFS,
    donate when stack grows large, exit when done/failed signalled."""
    # Install SIGINT handler so Ctrl-C from the master cleanly
    # flips `failed_event` and exits.
    signal.signal(signal.SIGINT, lambda *_: (failed_event.set(), sys.exit(0)))

    # Lazy imports of the bnb-internal functions so the fork happens
    # with the module preloaded but per-process caches remain clean.
    from interval_bnb.bound_eval import (
        batch_bounds_full,
        batch_bounds_rank1_hi,
        batch_bounds_rank1_lo,
        gap_weighted_split_axis,
        window_tensor,
    )
    from interval_bnb.bnb import rigor_replay
    from interval_bnb.windows import build_windows

    windows = build_windows(d)
    A_tensor, scales = window_tensor(windows, d)
    target_q = target_c if isinstance(target_c, Fraction) else Fraction(str(target_c))
    target_f = float(target_q)

    # Depth threshold above which rigor_replay also tries the joint-face
    # McCormick dual certificate (closes boxes where max(SW, NE) < target
    # but the tighter joint LP crosses it). Below threshold, greedy bounds
    # + splitting is cheaper than invoking scipy.linprog per box.
    joint_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_JOINT_DEPTH", "20"
    ))
    # Depth threshold above which we tighten the box to its simplex
    # intersection before evaluating bounds. This tightens `hi` via the
    # simplex sum-constraint (new_hi[i] = min(hi[i], 1 - (lo_sum - lo[i])))
    # and `lo` symmetrically. The tightened box has the same simplex
    # intersection as the original, so any bound computed on it is still
    # valid, but `autoconv` (which uses (sum hi)^2 and hi^T A_W hi as
    # coefficients rather than just as constraints) gets strictly tighter.
    # At d=16 t=1.25 the BnB stalls because autoconv plateaus around 1.1
    # on deep boxes; tightening hi closes that gap.
    tighten_depth_threshold = int(os.environ.get(
        "INTERVAL_BNB_TIGHTEN_DEPTH", "15"
    ))

    local_stack: List[tuple] = []  # entries: (Box, depth, parent_cache, axis, which_end)
    local_nodes = 0
    local_cert = 0
    local_vol = 0.0
    local_max_depth = 0
    donate_period = random.randint(50, 150)
    since_last_donate_check = 0
    rng = random.Random(worker_id * 7919 + 1)

    def _publish_stats():
        nonlocal local_nodes, local_cert, local_vol
        with node_count.get_lock():
            node_count.value += local_nodes
        with cert_count.get_lock():
            cert_count.value += local_cert
        with closed_vol.get_lock():
            closed_vol.value += local_vol
        local_nodes = 0
        local_cert = 0
        local_vol = 0.0

    try:
        while not done_event.is_set() and not failed_event.is_set():
            # ----- Refill from shared queue if empty -----
            if not local_stack:
                with idle_count.get_lock():
                    idle_count.value += 1
                try:
                    batch = queue.get(timeout=0.1)
                except Exception:
                    batch = None
                with idle_count.get_lock():
                    idle_count.value -= 1
                if batch is None:
                    # Queue empty. Check termination.
                    with in_flight.get_lock():
                        if in_flight.value == 0:
                            done_event.set()
                            break
                    # Otherwise spin: another worker is still splitting.
                    continue
                # `batch` is a list of (Box, depth) tuples (no parent_cache -- we
                # lost it at donation; recompute via batch_bounds_full).
                for item in batch:
                    if isinstance(item, tuple) and len(item) >= 2:
                        B, depth = item[0], item[1]
                    else:
                        B = item
                        depth = 0
                    local_stack.append((B, depth, None, -1, None))
                continue

            # ----- Process one node -----
            B, depth, parent_cache, changed_k, which_end = local_stack.pop()
            local_nodes += 1
            if depth > local_max_depth:
                local_max_depth = depth
            # Periodic progress publish so the master sees counters advance.
            if local_nodes % 50_000 == 0:
                _publish_stats()
            if local_nodes >= max_nodes_per_worker:
                # Safety cap (shouldn't normally fire).
                failed_event.set()
                break

            if not B.intersects_simplex():
                local_vol += B.volume()
                with in_flight.get_lock():
                    in_flight.value -= 1
                continue

            # T3 (simplex-tightened box) re-enabled at depth threshold.
            # On stall boxes at d=16 the autoconv bound is binding (~1.1
            # vs target 1.25); tightening hi via sum-constraint is the
            # only cheap way to lift autoconv. Invalidates parent_cache
            # because lo/hi change, forcing a full batch_bounds_full
            # recompute on the tightened endpoints.
            if depth >= tighten_depth_threshold:
                if B.tighten_to_simplex():
                    parent_cache = None
                    if not B.intersects_simplex():
                        # Tightening collapsed the simplex intersection
                        # (should be rare; means box was barely feasible).
                        local_vol += B.volume()
                        with in_flight.get_lock():
                            in_flight.value -= 1
                        continue

            if parent_cache is None:
                lb_fast, w_idx, which, _, my_cache = batch_bounds_full(
                    B.lo, B.hi, A_tensor, scales, target_f,
                )
            elif which_end == "lo":
                lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_lo(
                    A_tensor, scales, parent_cache, B.lo, changed_k, target_f,
                )
            else:
                lb_fast, w_idx, which, _, my_cache = batch_bounds_rank1_hi(
                    A_tensor, scales, parent_cache, B.hi, changed_k, target_f,
                )

            if lb_fast >= target_f and w_idx >= 0:
                w = windows[w_idx]
                if rigor_replay(
                    B, w, d, which, target_q,
                    try_joint=(depth >= joint_depth_threshold),
                ):
                    local_cert += 1
                    local_vol += B.volume()
                    with in_flight.get_lock():
                        in_flight.value -= 1
                    continue
                # Fraction replay refused -- split further below.

            if B.max_width() < min_box_width:
                failed_event.set()
                break

            # Reverted to widest-axis (see bnb.py note on gap-weighted).
            axis = B.widest_axis()
            left, right = B.split(axis)
            delta = 1  # net change: +2 children - 1 closed parent
            if not left.intersects_simplex():
                delta -= 1
                local_vol += left.volume()
            if not right.intersects_simplex():
                delta -= 1
                local_vol += right.volume()
            with in_flight.get_lock():
                in_flight.value += delta
            if right.intersects_simplex():
                local_stack.append((right, depth + 1, my_cache, axis, "lo"))
            if left.intersects_simplex():
                local_stack.append((left, depth + 1, my_cache, axis, "hi"))

            # ----- Donate shallow half if stack grew large -----
            # trigger goes DOWN as more workers are idle (be more generous
            # donating when others have nothing; hoard only when everyone
            # is busy and donations would be wasted IPC).
            since_last_donate_check += 1
            if since_last_donate_check >= donate_period:
                since_last_donate_check = 0
                donate_period = rng.randint(10, 40)  # tight check period
                idle_now = max(0, idle_count.value)
                # Linear decay: 64 boxes when nobody idle, floor when all idle.
                trigger = max(donate_threshold_floor, 64 - 2 * idle_now)
                if len(local_stack) >= trigger:
                    # Share more when many idle: donate larger fraction.
                    share_frac = 0.5 + 0.01 * idle_now  # 0.5..~0.8
                    k = max(1, int(len(local_stack) * share_frac))
                    if k < len(local_stack):
                        donation = local_stack[:k]
                        local_stack = local_stack[k:]
                        donated_payload = [(B, depth) for (B, depth, _, _, _) in donation]
                        try:
                            queue.put(donated_payload, timeout=1.0)
                        except Exception:
                            local_stack = donation + local_stack

    finally:
        _publish_stats()
        if local_max_depth > 0:
            try:
                stats_queue.put_nowait({
                    "worker_id": worker_id,
                    "max_depth": local_max_depth,
                })
            except Exception:
                pass


# ---------------------------------------------------------------------
# Master-side driver
# ---------------------------------------------------------------------

def parallel_branch_and_bound(
    d: int,
    target_c,
    *,
    workers: Optional[int] = None,
    init_split_depth: int = 10,
    min_box_width: float = 1e-10,
    max_nodes: int = 10**10,
    pull_batch_max: int = 64,
    donate_threshold_floor: int = 16,
    time_budget_s: Optional[float] = None,
    use_symmetry: bool = True,
    verbose: bool = True,
    **_ignored,
):
    """Run work-stealing parallel BnB. Returns a result dict.

    Parameters
    ----------
    d : int                ambient dimension
    target_c : number      the c to certify (Fraction, str, or float)
    workers : int          number of worker processes (default cpu_count-1)
    init_split_depth : int seeds ~2**depth starter boxes
    min_box_width : float  failure threshold
    max_nodes : int        global node cap (each worker enforces a share)
    pull_batch_max : int   max boxes per shared-queue get
    donate_threshold_floor : int  min stack size before a worker donates
    time_budget_s : float  soft wall-clock cap
    """
    ctx = mp.get_context("fork") if sys.platform != "win32" else mp.get_context("spawn")

    sym = half_simplex_cuts(d) if use_symmetry else []
    t0 = time.time()
    starter_boxes = _split_initial(d, init_split_depth, sym)
    n_starter = len(starter_boxes)
    workers = workers or max(1, mp.cpu_count() - 1)
    if verbose:
        print(f"[par] d={d}  target={target_c}  init_boxes={n_starter}  "
              f"workers={workers}  init_split_depth={init_split_depth}")

    # Shared primitives
    queue = ctx.Queue(maxsize=0)
    in_flight = ctx.Value("i", n_starter)
    idle_count = ctx.Value("i", 0)
    cert_count = ctx.Value("i", 0)
    node_count = ctx.Value("i", 0)
    closed_vol = ctx.Value("d", 0.0)
    failed_event = ctx.Event()
    done_event = ctx.Event()
    stats_queue = ctx.Queue()

    # Total search volume = sum of starter box volumes. This is the
    # denominator of the ETA fraction.
    total_volume = float(sum(b.volume() for b in starter_boxes))

    # Seed the queue in CHUNKS so the queue machinery is primed. We send
    # each starter box as a singleton list so workers can batch-pull.
    # Larger chunks reduce queue IPC but may starve fast workers at start.
    seed_chunk = max(1, n_starter // (workers * 4) + 1)
    for i in range(0, n_starter, seed_chunk):
        chunk = [(b, 0) for b in starter_boxes[i:i + seed_chunk]]
        queue.put(chunk)

    # Max nodes per worker (approximate -- just a safety cap)
    max_nodes_per_worker = max(1, max_nodes // workers)

    procs: List[mp.Process] = []
    target_pass = target_c if isinstance(target_c, Fraction) else str(target_c)
    for wid in range(workers):
        p = ctx.Process(
            target=_worker_main,
            args=(
                wid, d, target_pass, min_box_width, max_nodes_per_worker,
                pull_batch_max, donate_threshold_floor,
                queue, in_flight, idle_count, cert_count, node_count,
                closed_vol, failed_event, done_event, stats_queue,
            ),
            daemon=False,
        )
        p.start()
        procs.append(p)

    last_log = time.time()
    last_nodes = 0
    try:
        while True:
            if done_event.wait(timeout=1.0):
                break
            if failed_event.is_set():
                break
            if time_budget_s is not None and time.time() - t0 > time_budget_s:
                failed_event.set()
                if verbose:
                    print(f"[par] time_budget_s={time_budget_s} exceeded; aborting")
                break
            # Progress log.
            t = time.time() - t0
            if verbose and t - (last_log - t0) > 5.0:
                n = node_count.value
                c = cert_count.value
                inf_ = in_flight.value
                cvol = closed_vol.value
                try:
                    qs = queue.qsize()
                except NotImplementedError:
                    qs = -1
                rate = (n - last_nodes) / max(1.0, time.time() - last_log)
                idle_n = idle_count.value
                # Volume-based progress + ETA.
                frac = min(1.0, cvol / total_volume) if total_volume > 0 else 0.0
                if frac > 1e-6:
                    eta_s = (1.0 - frac) * t / frac
                    eta_tag = f"eta={eta_s:7.0f}s" if eta_s < 99999 else "eta= >1day"
                else:
                    eta_tag = "eta=  --"
                print(f"[par] t={t:7.1f}s  nodes={n:>10d}  cert={c:>10d}  "
                      f"in_flight={inf_:>6d}  queue={qs:>4d}  "
                      f"active={workers - idle_n:>2d}/{workers}  "
                      f"rate={rate:>7.0f}/s  "
                      f"progress={100*frac:5.1f}%  {eta_tag}")
                last_log = time.time()
                last_nodes = n
    finally:
        # Signal workers to stop and join.
        done_event.set()
        for p in procs:
            p.join(timeout=5.0)
            if p.is_alive():
                p.terminate()
                p.join(timeout=2.0)

    ok = not failed_event.is_set()
    stats_items = []
    while True:
        try:
            stats_items.append(stats_queue.get_nowait())
        except Exception:
            break
    max_depth = max((s["max_depth"] for s in stats_items), default=0)

    elapsed = time.time() - t0
    cvol_final = closed_vol.value
    result = {
        "success": ok,
        "target_q": str(target_c) if isinstance(target_c, Fraction) else str(target_c),
        "d": d,
        "elapsed_s": elapsed,
        "total_nodes": node_count.value,
        "total_leaves_certified": cert_count.value,
        "closed_volume": cvol_final,
        "total_volume": total_volume,
        "coverage_fraction": cvol_final / total_volume if total_volume > 0 else 0.0,
        "in_flight_final": in_flight.value,
        "max_depth": max_depth,
        "workers": workers,
        "init_boxes": n_starter,
    }
    if verbose:
        print(f"[par] DONE success={ok} elapsed={elapsed:.1f}s  "
              f"nodes={result['total_nodes']}  cert={result['total_leaves_certified']}  "
              f"coverage={100*result['coverage_fraction']:.3f}%  "
              f"in_flight_final={result['in_flight_final']}")
    return result
