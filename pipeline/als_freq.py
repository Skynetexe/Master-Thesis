"""ALS with frequency-scaled regularization, for ALS phase D.

pipeline/als.py stays exactly as P6 ran it. P6's marker hashes it, and P11
to P13b2 hold P6's marker as an input, so a change there would make the
runner rebuild every stage from P11 on. This module imports als.py and
adds one option, the exponent nu of the regularizer.

Regularizer, after Rendle, Krichene, Zhang & Koren (2022), section 2.4:

  reg * (sum_u f_u^nu |x_u|^2 + sum_i f_i^nu |y_i|^2)

In their loss, f_u = |I(u)| + alpha_0 |I| is the total weight of user u's
loss terms: weight 1 on each observed pair and alpha_0 on every item. In
the parameterization of Hu, Koren & Volinsky (2008) that als.py uses
(weight c = 1 + alpha log1p(listens) on an observed pair, weight 1 on
every other pair), the total weight of row r is

  f_r = sum over r's observed pairs of (c - 1) + n_other,

where n_other counts the real slots on the other side (item slots for a
user row, user slots for an item row, slot 0 excluded). P6 stores
round(1000 log1p(listens)), so the sum is alpha * M_r / 1000 with M_r the
exact integer sum of row r's stored data. nu = 0 gives als.py's
regularizer. fit() then gives the same bits as als.fit() (test_als_d.py
checks this), because the row solves keep als.py's order of operations.

Objective:
  J = sum over observed pairs of (c (1 - s)^2 - s^2)
      + sum over all pairs of s^2
      + reg (sum_u w_u |x_u|^2 + sum_i w_i |y_i|^2),   w = f^nu.

Each half-step solves every row exactly for this J, so J cannot rise,
and the same check as in als.fit() stops a fit whose J rises.
"""

import json
import multiprocessing as mp
import os
import queue
import shutil
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import als
from .als import (
    ACCUM_DTYPE,
    BLOCK,
    DATA_SCALE,
    RISE_TOLERANCE,
    ObjectiveRose,
    Progress,
    threadpool_limits,
)

FREQ_VERSION = "freq-reg-v1"
MASS_CHUNK = 1 << 26  # stored entries per read in listen_mass()
MASS_SCALE = 1000.0  # P6 stores round(1000 * log1p(listens))
HEARTBEAT_SECONDS = 60.0  # a line at least this often while blocks run


@dataclass(frozen=True)
class FreqParams:
    factors: int
    reg: float
    alpha: float
    iterations: int
    seed: int
    nu: float


# Weights


def listen_mass(rows, log=None, label="als_freq: listen mass"):
    """M_r, the exact int64 sum of each row's stored data (round(1000
    log1p(listens))), for one CSR orientation (indptr, indices, data)."""
    indptr = np.asarray(rows[0], dtype=np.int64)
    data = rows[2]
    n = len(indptr) - 1
    out = np.zeros(n, dtype=np.int64)
    prog = Progress(label, int(indptr[-1]), log or (lambda msg: None), unit=" entries")
    lo = 0
    while lo < n:
        # Rows [lo, hi) with at most MASS_CHUNK entries, at least one row.
        hi = int(np.searchsorted(indptr, indptr[lo] + MASS_CHUNK, side="right")) - 1
        hi = min(max(hi, lo + 1), n)
        s, e = int(indptr[lo]), int(indptr[hi])
        if e > s:
            vals = np.asarray(data[s:e], dtype=np.int64)
            nonempty = lo + np.flatnonzero(np.diff(indptr[lo : hi + 1]) > 0)
            out[nonempty] = np.add.reduceat(vals, indptr[nonempty] - s)
        prog.advance(e - s)
        lo = hi
    prog.finish()
    return out


def row_weights(mass, alpha, n_other, nu):
    """w_r = f_r^nu with f_r = alpha * M_r / 1000 + n_other, float64. nu = 0
    gives exactly 1.0 for every row."""
    if nu == 0:
        return np.ones(len(mass), dtype=np.float64)
    f = float(alpha) * (np.asarray(mass, dtype=np.float64) / MASS_SCALE) + float(
        n_other
    )
    return f ** float(nu)


def mass_of(inter, cache_path, log=None):
    """(M_users, M_items) for inter, from cache_path when it belongs to the
    same data, else computed and cached."""
    cache_path = Path(cache_path)
    fp = inter.fingerprint()
    if cache_path.exists():
        z = np.load(cache_path)
        if (
            str(z["fingerprint"]) == fp
            and len(z["users"]) == inter.n_users
            and len(z["items"]) == inter.n_items
        ):
            return (
                np.asarray(z["users"], dtype=np.int64),
                np.asarray(z["items"], dtype=np.int64),
            )
    m_u = listen_mass(inter.u, log, "als_freq: listen mass of the users")
    m_i = listen_mass(inter.i, log, "als_freq: listen mass of the items")
    if int(m_u.sum()) != int(m_i.sum()):
        raise ValueError(
            "the two CSR orientations disagree on the total " "listen mass"
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez(tmp, users=m_u, items=m_i, fingerprint=np.array(fp))
    os.replace(tmp, cache_path)
    return m_u, m_i


def median_frequency(mass, indptr, alpha, n_other):
    """The median of f over the rows that hold data."""
    nonempty = np.diff(np.asarray(indptr, dtype=np.int64)) > 0
    f = row_weights(np.asarray(mass)[nonempty], alpha, n_other, 1.0)
    return float(np.median(f))


def frequency_quantiles(mass, indptr, alpha, n_other, qs=(0.0, 0.01, 0.5, 0.99, 1.0)):
    """Quantiles of f over the rows that hold data, as {"q0.01": ...}. They
    show how much the weight f^nu varies within one side."""
    nonempty = np.diff(np.asarray(indptr, dtype=np.int64)) > 0
    f = row_weights(np.asarray(mass)[nonempty], alpha, n_other, 1.0)
    return {f"q{q:g}": float(np.quantile(f, q)) for q in qs}


# Solver
_CTX = {}


def _solve_rows(lo, hi):
    """als._solve_rows with the weight w_r in the diagonal term. With
    w_r = 1 every operation equals als.py's, in the same order."""
    indptr, indices, data = _CTX["rows"]
    target, fixed = _CTX["target"], _CTX["fixed"]
    G, eye, reg = _CTX["gram"], _CTX["eye"], _CTX["reg"]
    w, alpha = _CTX["w_target"], _CTX["alpha"]
    scale = ACCUM_DTYPE(DATA_SCALE)
    observed = 0.0
    for r in range(lo, hi):
        s, e = int(indptr[r]), int(indptr[r + 1])
        if e <= s:
            target[r] = 0.0
            continue
        cols = np.asarray(indices[s:e])
        c = 1.0 + alpha * (np.asarray(data[s:e], dtype=ACCUM_DTYPE) * scale)
        F = np.asarray(fixed[cols], dtype=ACCUM_DTYPE)
        gram_reg = G + (reg * float(w[r])) * eye
        A = gram_reg + ((F * (c - 1.0)[:, None]).T @ F).astype(np.float64)
        b = (F.T @ c).astype(np.float64)
        x = np.linalg.solve(A, b).astype(np.float32)
        target[r] = x
        sc = F.astype(np.float64) @ x.astype(np.float64)
        c64 = c.astype(np.float64)
        observed += float(np.sum(c64 * (1.0 - sc) ** 2 - sc * sc))
    return observed


def _solve_block(task):
    with threadpool_limits(1):
        return _solve_rows(*task)


def _weighted_sum_squares(M, w):
    """sum_r w_r |m_r|^2 in float64, in als._sum_squares's chunks."""
    total = 0.0
    step = 1 << 20
    for lo in range(0, M.shape[0], step):
        blk = np.asarray(M[lo : lo + step], dtype=np.float64)
        total += float(np.sum(np.sum(blk * blk, axis=1) * w[lo : lo + step]))
    return total


def _reg_term(target, fixed, w_target, w_fixed, nu):
    if nu == 0:
        return als._sum_squares(target) + als._sum_squares(fixed)
    return _weighted_sum_squares(target, w_target) + _weighted_sum_squares(
        fixed, w_fixed
    )


def _worker_init():
    """Workers ignore Ctrl+C. The main process handles it and ends them."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _stop_pool(pool):
    """End a pool at once: cancel the blocks not yet started and terminate
    the workers. Workers write only to the iteration's work files, which
    the next run copies afresh, so no committed file is touched."""
    try:
        pool.terminate_workers()  # Python 3.14 and later
    except Exception:  # older Python, or already gone
        for p in list((getattr(pool, "_processes", None) or {}).values()):
            try:
                p.terminate()
            except Exception:  # already ended
                pass
    pool.shutdown(wait=False, cancel_futures=True)


def _run_blocks(tasks, indptr, workers, prog, label, log):
    """_solve_block over every task. Returns the parts in task order.

    Each block is its own task, and its entries count toward the progress
    as soon as it finishes, in any order. So the first progress line
    comes about one block's time after the start. While no block finishes,
    a line appears every HEARTBEAT_SECONDS. The caller
    sums the parts in task order, so the order in which blocks finish
    cannot change a result."""
    log = log or (lambda msg: None)
    parts = [None] * len(tasks)
    if workers <= 1:
        for k, (lo, hi) in enumerate(tasks):
            parts[k] = _solve_block((lo, hi))
            prog.advance(int(indptr[hi] - indptr[lo]))
        return parts
    t0 = time.time()
    finished = queue.SimpleQueue()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("fork"), initializer=_worker_init
    )
    try:
        futures = []
        for k, task in enumerate(tasks):
            f = pool.submit(_solve_block, task)
            f.add_done_callback(lambda _f, k=k: finished.put(k))
            futures.append(f)
        left = len(tasks)
        while left:
            try:
                k = finished.get(timeout=HEARTBEAT_SECONDS)
            except queue.Empty:
                log(
                    f"{label}: working, {len(tasks) - left:,} of "
                    f"{len(tasks):,} blocks done, "
                    f"{als._clock(time.time() - t0)} elapsed"
                )
                continue
            parts[k] = futures[k].result()  # raises a worker's error
            lo, hi = tasks[k]
            prog.advance(int(indptr[hi] - indptr[lo]))
            left -= 1
    except BaseException:
        _stop_pool(pool)
        raise
    pool.shutdown(wait=True)
    return parts


def _half(
    rows,
    target,
    fixed,
    reg,
    alpha,
    w_target,
    w_fixed,
    nu,
    workers,
    label="als_freq: half-step",
    log=None,
):
    """Solve every row of target exactly with fixed held constant. Returns
    the exact objective J at the new (target, fixed)."""
    indptr = rows[0]
    n = len(indptr) - 1
    tasks = [(lo, min(lo + BLOCK, n)) for lo in range(0, n, BLOCK)]
    if log:
        log(
            f"{label}: started, {n:,} rows in {len(tasks):,} blocks on "
            f"{max(1, workers)} worker(s). First the Gram matrix of the "
            f"fixed side, then a progress line as blocks finish."
        )
    G = als._gram(fixed)
    _CTX.update(
        rows=rows,
        target=target,
        fixed=fixed,
        alpha=float(alpha),
        gram=G,
        eye=np.eye(G.shape[0]),
        reg=float(reg),
        w_target=w_target,
    )
    prog = Progress(
        label, int(indptr[n] - indptr[0]), log or (lambda msg: None), unit=" entries"
    )
    try:
        parts = _run_blocks(tasks, indptr, workers, prog, label, log)
    finally:
        _CTX.clear()
    prog.finish()
    observed = float(np.sum(parts))  # task order, so the sum is deterministic
    return (
        observed
        + als._sum_quadratic(target, G)
        + reg * _reg_term(target, fixed, w_target, w_fixed, nu)
    )


def objective(inter, X, Y, params, mass_users, mass_items):
    """J at (X, Y), computed from scratch without the solver (for checks)."""
    w_u = row_weights(mass_users, params.alpha, inter.n_items - 1, params.nu)
    w_i = row_weights(mass_items, params.alpha, inter.n_users - 1, params.nu)
    indptr, indices, data = inter.u
    observed = 0.0
    for u in range(inter.n_users):
        s, e = int(indptr[u]), int(indptr[u + 1])
        if e <= s:
            continue
        cols = np.asarray(indices[s:e])
        c = 1.0 + params.alpha * (np.asarray(data[s:e], dtype=np.float64) * DATA_SCALE)
        sc = np.asarray(Y[cols], dtype=np.float64) @ np.asarray(X[u], dtype=np.float64)
        observed += float(np.sum(c * (1.0 - sc) ** 2 - sc * sc))
    Xd = np.asarray(X, dtype=np.float64)
    Yd = np.asarray(Y, dtype=np.float64)
    all_pairs = float(np.sum((Xd @ Yd.T) ** 2))
    reg = params.reg * (
        float(np.sum(np.sum(Xd * Xd, axis=1) * w_u))
        + float(np.sum(np.sum(Yd * Yd, axis=1) * w_i))
    )
    return observed + all_pairs + reg


# Fit


def fit_key(inter, params):
    """als.fit_key with the exponent and this module's version."""
    return {
        "params": asdict(params),
        "solver": als.SOLVER_VERSION,
        "freq_solver": FREQ_VERSION,
        "accum": np.dtype(ACCUM_DTYPE).name,
        "data": inter.fingerprint(),
        "shape": [inter.n_users, inter.n_items, inter.nnz],
    }


def fit(
    inter,
    params,
    x_path,
    y_path,
    state_path,
    mass_users,
    mass_items,
    workers=None,
    on_iteration=None,
    log=None,
):
    """als.fit() with the regularizer above. The same resume state, factor
    files, callback contract and objective check. mass_users and
    mass_items: listen_mass() of each orientation (mass_of())."""
    log = log or (lambda msg: print(msg, flush=True))
    real = als.orientation_real_fraction(inter, rows=100, per_row=100)
    if real < 1.0:
        raise ValueError(
            f"the item-major CSR is not the transpose of the user-major CSR: "
            f"only {real:.1%} of sampled item-major entries are real pairs. "
            f"Build it with Interactions.with_transposed_item_major()."
        )
    mass_users = np.asarray(mass_users, dtype=np.int64)
    mass_items = np.asarray(mass_items, dtype=np.int64)
    if len(mass_users) != inter.n_users or len(mass_items) != inter.n_items:
        raise ValueError("the listen masses do not match the data's slots")
    w_u = row_weights(mass_users, params.alpha, inter.n_items - 1, params.nu)
    w_i = row_weights(mass_items, params.alpha, inter.n_users - 1, params.nu)
    x_path, y_path, state_path = Path(x_path), Path(y_path), Path(state_path)
    for p in (x_path, y_path, state_path):
        p.parent.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, min(16, os.cpu_count() or 1))
    key = fit_key(inter, params)

    state = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    resumable = (
        state is not None
        and state.get("key") == key
        and x_path.exists()
        and y_path.exists()
        and state.get("x_sha") == als._file_sha(x_path)
        and state.get("y_sha") == als._file_sha(y_path)
    )
    if resumable:
        history = state["history"]
        if history:
            log(f"als_freq: resuming after iteration {len(history)}")
    else:
        if state is not None:
            log(
                "als_freq: the saved fit does not match these parameters, "
                "this solver or this data. Starting fresh."
            )
        rng = np.random.default_rng(params.seed)
        for path, rows in ((x_path, inter.n_users), (y_path, inter.n_items)):
            work = als._work_path(path)
            arr = np.lib.format.open_memmap(
                work, mode="w+", dtype=np.float32, shape=(rows, params.factors)
            )
            step = 2_000_000
            for lo in range(0, rows, step):
                hi = min(lo + step, rows)
                arr[lo:hi] = (
                    rng.standard_normal((hi - lo, params.factors)) * 0.01
                ).astype(np.float32)
            arr.flush()
            del arr
            os.replace(work, path)  # never rewrite a committed file in place
        history = []
        als._write_json(
            state_path,
            {
                "key": key,
                "history": history,
                "x_sha": als._file_sha(x_path),
                "y_sha": als._file_sha(y_path),
            },
        )

    def save(x_sha, y_sha):
        als._write_json(
            state_path, {"key": key, "history": history, "x_sha": x_sha, "y_sha": y_sha}
        )

    def run_callback(iteration):
        extra = {}
        if on_iteration is not None:
            extra = (
                on_iteration(
                    iteration,
                    np.load(x_path, mmap_mode="r"),
                    np.load(y_path, mmap_mode="r"),
                    history,
                )
                or {}
            )
        history[-1].update(extra)
        history[-1]["callback_done"] = True

    if history and not history[-1].get("callback_done"):
        run_callback(len(history))
        save(als._file_sha(x_path), als._file_sha(y_path))

    prev = history[-1]["objective_after_items"] if history else None
    with threadpool_limits(1):
        for it in range(len(history), params.iterations):
            if history and history[-1].get("stop"):
                break
            t0 = time.time()
            step = f"als_freq: iteration {it + 1}/{params.iterations}"
            log(f"{step} started")
            xw, yw = als._work_path(x_path), als._work_path(y_path)
            shutil.copyfile(x_path, xw)
            shutil.copyfile(y_path, yw)
            X = np.lib.format.open_memmap(xw, mode="r+")
            Y = np.lib.format.open_memmap(yw, mode="r+")
            j_users = _half(
                inter.u,
                X,
                Y,
                params.reg,
                params.alpha,
                w_u,
                w_i,
                params.nu,
                workers,
                label=f"{step}, users half",
                log=log,
            )
            j_items = _half(
                inter.i,
                Y,
                X,
                params.reg,
                params.alpha,
                w_i,
                w_u,
                params.nu,
                workers,
                label=f"{step}, items half",
                log=log,
            )
            X.flush()
            Y.flush()
            del X, Y
            rises = []
            for j in (j_users, j_items):
                if not np.isfinite(j):
                    raise ObjectiveRose(
                        f"iteration {it + 1}: the objective is " f"not finite"
                    )
                if prev is not None:
                    rel = (j - prev) / abs(prev)
                    rises.append(rel)
                    if rel > RISE_TOLERANCE:
                        raise ObjectiveRose(
                            f"iteration {it + 1}: the objective rose by a "
                            f"relative {rel:.2e}. Exact row solves on "
                            f"consistent data cannot do that."
                        )
                prev = j
            x_sha, y_sha = als._file_sha(xw), als._file_sha(yw)
            os.replace(xw, x_path)
            os.replace(yw, y_path)
            history.append(
                {
                    "iteration": it + 1,
                    "seconds": round(time.time() - t0, 1),
                    "objective_after_users": j_users,
                    "objective_after_items": j_items,
                    "max_relative_rise": max(rises) if rises else None,
                }
            )
            save(x_sha, y_sha)
            change = (
                f", largest relative objective change {max(rises):+.1e}"
                if rises
                else ""
            )
            log(
                f"{step}: fit done in {history[-1]['seconds']:.0f} s, "
                f"objective {j_items:.6e}{change}"
            )
            run_callback(it + 1)
            save(x_sha, y_sha)
    return (np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r"), history)
