"""Checks for pipeline/als_freq.py and als_tune_d.py (ALS phase D) on small
synthetic data. No real data and no DuckDB needed.

1. listen_mass() equals the dense row sums of the stored data, also when
   rows straddle its read chunks, and row_weights() follows the formula.
2. mass_of() caches the masses, reuses the cache for the same data,
   recomputes it for other data, and refuses orientations that disagree.
3. With nu = 1, one half-step equals the dense closed-form solve of every
   row, the reported objective equals the dense objective, and the
   dense objective never rises over ten iterations.
4. With nu = 0, als_freq.fit() gives the same bits as als.fit(): the
   factors and every recorded objective. So phase D's nu = 0 cell runs
   the solver of phases A to C.
5. Two worker processes give the same bits as one. A stop inside a
   half-step or inside the callback resumes to the same bits as an
   uninterrupted fit. A changed nu starts fresh.
6. The search rule (search_cell, run_cells) with stand-in scores: the k
   sequence in each direction, the step limits, the start of D2 from
   D1's best reg, the gate, the nu and reg of D3, and ties.
7. run_phase() end to end with real fits: the winner rule, best.json and
   its files (equal to a fresh fit stopped at the winner's best
   iteration), the report, a second call that fits nothing, recovery
   from a stop inside keep_best(), the disk check, and CURATED is never
   written.
8. Ties and early stops: with every score tied, no cell goes past
   k = -1, every config stops at iteration 3, the first config's first
   iteration wins and is kept, and phase D fails the gate. A stop in the
   first ranking keeps no factors, and the next call reaches the same
   winner. The declared constants hold.

Usage, from the repository root:
    python3 test_als_d.py
"""

import ast
import json
import math
import multiprocessing as mp
import os
import shutil
import signal
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import als_tune
from pipeline import als_tune_d as td
from pipeline import als, als_freq

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)
        if os.environ.get("FAIL_FAST"):  # FAIL_FAST=1: stop at the first failed check
            raise SystemExit(1)


def quiet(msg):
    pass


def build(P, R):
    """Interactions in P6's layout from dense 0/1 P and listen counts R
    (as in test_als.py). Row 0 and column 0 are the phantom slots."""
    data_dense = np.where(P > 0, np.round(1000.0 * np.log1p(R)), 0)

    def csr(M, D):
        rows, cols = np.nonzero(M)
        indptr = np.zeros(M.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        return (indptr, cols.astype(np.int32), D[rows, cols].astype(np.uint16))

    u = csr(P, data_dense)
    i = csr(P.T, data_dense.T)
    return als.Interactions(*u, *i), data_dense


def dense_conf(inter, alpha, dtype=np.float32):
    """c as the solver computes it (in ACCUM_DTYPE), as a dense matrix."""
    C = np.ones((inter.n_users, inter.n_items), dtype=np.float64)
    Pm = np.zeros_like(C)
    indptr, indices, data = inter.u
    scale = dtype(als.DATA_SCALE)
    for u in range(inter.n_users):
        s, e = indptr[u], indptr[u + 1]
        c = 1.0 + alpha * (np.asarray(data[s:e], dtype=dtype) * scale)
        C[u, indices[s:e]] = c.astype(np.float64)
        Pm[u, indices[s:e]] = 1.0
    return C, Pm


def dense_objective(X, Y, C, Pm, reg, w_u, w_i):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    S = X @ Y.T
    return float(
        (C * (Pm - S) ** 2).sum()
        + reg * (np.sum((X**2).sum(axis=1) * w_u) + np.sum((Y**2).sum(axis=1) * w_i))
    )


rng = np.random.default_rng(29)
NU, NI = 120, 95
P = (rng.random((NU, NI)) < 0.14).astype(np.float64)
P[0, :] = 0
P[:, 0] = 0
P[6, :] = 0  # a user without data
P[:, 9] = 0  # an item without data
R = np.where(P > 0, rng.pareto(1.2, (NU, NI)) * 20 + 1, 0.0)
inter, DATA = build(P, R)
TMP = Path(tempfile.mkdtemp(prefix="test_als_d_"))

# 1: masses and weights
m_u_dense = DATA.sum(axis=1).astype(np.int64)
m_i_dense = DATA.sum(axis=0).astype(np.int64)
real_chunk = als_freq.MASS_CHUNK
ok = True
for chunk in (real_chunk, 7, 1, 50):
    als_freq.MASS_CHUNK = chunk
    ok &= np.array_equal(als_freq.listen_mass(inter.u), m_u_dense)
    ok &= np.array_equal(als_freq.listen_mass(inter.i), m_i_dense)
als_freq.MASS_CHUNK = real_chunk
check(
    ok,
    "listen_mass() equals the dense row sums for both orientations, "
    "with read chunks of 1, 7, 50 and 2^26 entries",
)
alpha = 3.0
w1 = als_freq.row_weights(m_u_dense, alpha, NI - 1, 1.0)
w_half = als_freq.row_weights(m_u_dense, alpha, NI - 1, 0.5)
w0 = als_freq.row_weights(m_u_dense, alpha, NI - 1, 0.0)
f_hand = alpha * m_u_dense / 1000.0 + (NI - 1)
check(
    np.allclose(w1, f_hand, rtol=1e-15, atol=0)
    and np.allclose(w_half, np.sqrt(f_hand), rtol=1e-15, atol=0)
    and np.array_equal(w0, np.ones(NU))
    and w0.dtype == np.float64,
    "row_weights(): f = alpha M / 1000 + n_other, f^nu, and exactly 1 " "for nu = 0",
)
f_nonempty = f_hand[np.diff(inter.u[0]) > 0]
check(
    abs(
        als_freq.median_frequency(m_u_dense, inter.u[0], alpha, NI - 1)
        - float(np.median(f_nonempty))
    )
    < 1e-9,
    "median_frequency() is the median of f over the rows with data",
)

# 2: mass cache
cache = TMP / "mass.npz"
m_u, m_i = als_freq.mass_of(inter, cache)
real_lm = als_freq.listen_mass
calls = []


def counting_lm(*a, **k):
    calls.append(1)
    return real_lm(*a, **k)


als_freq.listen_mass = counting_lm
try:
    m_u2, m_i2 = als_freq.mass_of(inter, cache)
    reuse = not calls
    P2 = P.copy()
    P2[3, 4] = 1.0 - P2[3, 4]
    inter2, _ = build(P2, np.where(P2 > 0, 5.0, 0.0))
    m_u3, _ = als_freq.mass_of(inter2, cache)
    recomputed = len(calls) == 2
finally:
    als_freq.listen_mass = real_lm
check(
    np.array_equal(m_u, m_u_dense)
    and np.array_equal(m_i, m_i_dense)
    and reuse
    and np.array_equal(m_u2, m_u)
    and recomputed
    and not np.array_equal(m_u3, m_u),
    "mass_of() caches the masses, reuses them for the same data and "
    "recomputes them for other data",
)
bad_i = (inter.i[0], inter.i[1], inter.i[2].copy())
bad_i[2][0] += 1
inter_bad = als.Interactions(*inter.u, *bad_i)
try:
    als_freq.mass_of(inter_bad, TMP / "mass_bad.npz")
    refused = False
except ValueError as e:
    refused = "total listen mass" in str(e)
check(refused, "mass_of() refuses two orientations with different totals")

# 3: nu = 1, dense checks
params1 = als_freq.FreqParams(
    factors=6, reg=0.004, alpha=alpha, iterations=10, seed=5, nu=1.0
)
w_u = als_freq.row_weights(m_u, alpha, NI - 1, 1.0)
w_i = als_freq.row_weights(m_i, alpha, NU - 1, 1.0)
C, Pm = dense_conf(inter, alpha)
Y0 = (np.random.default_rng(1).standard_normal((NI, 6)) * 0.1).astype(np.float32)
X0 = np.zeros((NU, 6), dtype=np.float32)
Y0[9] = 0.0
j_half = als_freq._half(inter.u, X0, Y0, params1.reg, alpha, w_u, w_i, 1.0, 1)
Yd = Y0.astype(np.float64)
close = True
for u in range(NU):
    A = (Yd.T * C[u]) @ Yd + params1.reg * w_u[u] * np.eye(6)
    b = (Yd.T * C[u]) @ Pm[u]
    x = np.linalg.solve(A, b) if Pm[u].any() else np.zeros(6)
    close &= np.allclose(X0[u], x, rtol=2e-4, atol=1e-6)
check(
    close and not X0[6].any(),
    "with nu = 1 a users half-step equals the dense closed-form solve of "
    "every row (with reg w_u), and a user without data solves to 0",
)
j_dense = dense_objective(X0, Y0, C, Pm, params1.reg, w_u, w_i)
j_obj = als_freq.objective(inter, X0, Y0, params1, m_u, m_i)
check(
    abs(j_half - j_dense) / j_dense < 1e-6 and abs(j_obj - j_dense) / j_dense < 1e-6,
    f"the objective of _half() and objective() equal the dense objective "
    f"(relative gaps {abs(j_half - j_dense) / j_dense:.1e}, "
    f"{abs(j_obj - j_dense) / j_dense:.1e})",
)
fit_dir = TMP / "fit_nu1"
X, Y, hist = als_freq.fit(
    inter,
    params1,
    fit_dir / "x.npy",
    fit_dir / "y.npy",
    fit_dir / "s.json",
    m_u,
    m_i,
    workers=1,
    log=quiet,
)
objs = [
    v for h in hist for v in (h["objective_after_users"], h["objective_after_items"])
]
rises = max(np.diff(objs) / np.abs(objs[:-1]))
j_end = dense_objective(X, Y, C, Pm, params1.reg, w_u, w_i)
check(
    len(hist) == 10 and rises <= 0 and abs(objs[-1] - j_end) / j_end < 1e-6,
    f"over ten iterations with nu = 1 the objective never rises (largest "
    f"change {rises:+.1e}) and the last recorded value equals the dense "
    f"objective",
)
# The items half-step runs last, so each item row minimizes J for the final
# X. (The user rows minimize J for the Y before that half-step.)
Xd, Ydd = np.asarray(X, np.float64), np.asarray(Y, np.float64)
grad_i = np.stack(
    [
        2 * ((Xd.T * C[:, i]) @ (Xd @ Ydd[i] - Pm[:, i]))
        + 2 * params1.reg * w_i[i] * Ydd[i]
        for i in range(NI)
    ]
)
grad_wrong = np.stack(
    [
        2 * ((Xd.T * C[:, i]) @ (Xd @ Ydd[i] - Pm[:, i])) + 2 * params1.reg * Ydd[i]
        for i in range(NI)
    ]
)
check(
    np.max(np.abs(grad_i)) < 1e-3 and np.max(np.abs(grad_wrong)) > 1e-1,
    f"after a fit, every item row is a stationary point of J with its own "
    f"weight w_i (largest gradient {np.max(np.abs(grad_i)):.1e}), and not "
    f"of J with weight 1 ({np.max(np.abs(grad_wrong)):.1e})",
)
fq = als_freq.frequency_quantiles(m_u_dense, inter.u[0], alpha, NI - 1)
check(
    set(fq) == {"q0", "q0.01", "q0.5", "q0.99", "q1"}
    and fq["q0"] == float(f_nonempty.min())
    and fq["q1"] == float(f_nonempty.max())
    and abs(fq["q0.5"] - float(np.median(f_nonempty))) < 1e-9,
    "frequency_quantiles(): minimum, median and maximum of f over the rows "
    "with data",
)

# 4: nu = 0 equals als.py
params_a = als.ALSParams(factors=6, reg=0.7, alpha=alpha, iterations=6, seed=9)
params_f = als_freq.FreqParams(
    factors=6, reg=0.7, alpha=alpha, iterations=6, seed=9, nu=0.0
)
d_a, d_f = TMP / "fit_als", TMP / "fit_freq0"
Xa, Ya, ha = als.fit(
    inter, params_a, d_a / "x.npy", d_a / "y.npy", d_a / "s.json", workers=1, log=quiet
)
Xf, Yf, hf = als_freq.fit(
    inter,
    params_f,
    d_f / "x.npy",
    d_f / "y.npy",
    d_f / "s.json",
    m_u,
    m_i,
    workers=1,
    log=quiet,
)
check(
    np.array_equal(np.asarray(Xa).view(np.uint32), np.asarray(Xf).view(np.uint32))
    and np.array_equal(np.asarray(Ya).view(np.uint32), np.asarray(Yf).view(np.uint32))
    and [h["objective_after_users"] for h in ha]
    == [h["objective_after_users"] for h in hf]
    and [h["objective_after_items"] for h in ha]
    == [h["objective_after_items"] for h in hf],
    "with nu = 0, als_freq.fit() gives the same bits as als.fit(): factors "
    "and every recorded objective",
)
check(
    not np.array_equal(np.asarray(X), np.asarray(Xf)),
    "nu = 1 gives other factors than nu = 0",
)

# 5: workers and resume
d_w = TMP / "fit_w2"
Xw, Yw, _ = als_freq.fit(
    inter,
    params1,
    d_w / "x.npy",
    d_w / "y.npy",
    d_w / "s.json",
    m_u,
    m_i,
    workers=2,
    log=quiet,
)
check(
    np.array_equal(np.asarray(Xw), np.asarray(X))
    and np.array_equal(np.asarray(Yw), np.asarray(Y)),
    "two worker processes give the same bits as one (nu = 1)",
)

# The pool, with small blocks so that a half-step has many.
real_block, real_beat = als_freq.BLOCK, als_freq.HEARTBEAT_SECONDS
real_rows_fn = als_freq._solve_rows


def half_once(workers, log=None):
    """One users half-step into a memmap, as in fit(): forked workers write
    their rows to the shared file."""
    path = TMP / f"half_{workers}_{time.time_ns()}.npy"
    Xh = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(NU, 6))
    j = als_freq._half(
        inter.u,
        Xh,
        Y0.copy(),
        0.004,
        alpha,
        w_u,
        w_i,
        1.0,
        workers,
        label="test half",
        log=log,
    )
    return j, np.array(Xh)


def slow_first(lo, hi):
    if lo == 0:
        import time as _t

        _t.sleep(0.6)  # block 0 finishes last
    return real_rows_fn(lo, hi)


als_freq.BLOCK = 16
try:
    j1, X1 = half_once(1)
    als_freq._solve_rows = slow_first
    als_freq.HEARTBEAT_SECONDS = 0.15
    heard = []
    j3, X3 = half_once(3, heard.append)
finally:
    als_freq._solve_rows = real_rows_fn
    als_freq.HEARTBEAT_SECONDS = real_beat
n_blocks = -(-NU // 16)
check(
    n_blocks == 8
    and j3 == j1
    and np.array_equal(X3, X1)
    and any("test half: started, 120 rows in 8 blocks" in x for x in heard)
    and any("test half: working, 7 of 8 blocks done" in x for x in heard),
    "with 8 blocks on 3 workers and the first block finishing last, a "
    "half-step gives the same bits as one worker, logs its start, and "
    "logs a heartbeat while it waits",
)


def failing_block(lo, hi):
    import time as _t

    if lo == 0:
        raise ValueError("planted failure in block 0")
    _t.sleep(3.0)
    return real_rows_fn(lo, hi)


als_freq._solve_rows = failing_block
t_fail = time.time()
try:
    half_once(2)
    failed = False
except ValueError as e:
    failed = "planted failure" in str(e)
finally:
    als_freq._solve_rows = real_rows_fn
    als_freq.BLOCK = real_block
waited = time.time() - t_fail
time.sleep(0.3)
check(
    failed and waited < 2.5 and not mp.active_children(),
    f"a failing block stops the half-step after {waited:.1f} s (the other "
    f"blocks need 3 s each), and no worker process stays behind",
)
old_handler = signal.getsignal(signal.SIGINT)
try:
    als_freq._worker_init()
    ignores = signal.getsignal(signal.SIGINT) == signal.SIG_IGN
finally:
    signal.signal(signal.SIGINT, old_handler)


def sigint_state(lo, hi):
    return 1.0 if signal.getsignal(signal.SIGINT) == signal.SIG_IGN else 0.0


als_freq._solve_rows = sigint_state
try:
    tasks8 = [(k, k + 1) for k in range(8)]
    parts8 = als_freq._run_blocks(
        tasks8, np.arange(9), 2, als.Progress("x", 8, quiet), "x", quiet
    )
finally:
    als_freq._solve_rows = real_rows_fn
check(
    ignores and parts8 == [1.0] * 8 and signal.getsignal(signal.SIGINT) == old_handler,
    "the worker processes ignore Ctrl+C (all 8 blocks see it ignored), so "
    "only the main process handles it and ends them",
)


class Counter:
    def __init__(self):
        self.steps = []

    def advance(self, n):
        self.steps.append(int(n))


def block_id(lo, hi):
    if lo == 0:
        import time as _t

        _t.sleep(0.4)  # block 0 finishes last
    return float(lo)


ptr8 = np.array([0, 3, 3, 10, 12, 20, 21, 30, 44])
als_freq._solve_rows = block_id
try:
    cnt = Counter()
    ids8 = als_freq._run_blocks(
        [(k, k + 1) for k in range(8)], ptr8, 3, cnt, "x", quiet
    )
finally:
    als_freq._solve_rows = real_rows_fn
check(
    ids8 == [float(k) for k in range(8)]
    and sorted(cnt.steps) == sorted(np.diff(ptr8).tolist())
    and sum(cnt.steps) == 44
    and cnt.steps[-1] == 3,
    "_run_blocks() returns the parts in task order while blocks finish in "
    "another order, and counts each block's entries once, when it finishes",
)


class Stop(Exception):
    pass


d_r = TMP / "fit_resume"


def stop_after(n):
    def cb(iteration, X_, Y_, history):
        if iteration == n and not getattr(cb, "done", False):
            cb.done = True
            raise Stop()
        return {}

    return cb


try:
    als_freq.fit(
        inter,
        params1,
        d_r / "x.npy",
        d_r / "y.npy",
        d_r / "s.json",
        m_u,
        m_i,
        workers=1,
        log=quiet,
        on_iteration=stop_after(3),
    )
    stopped = False
except Stop:
    stopped = True
real_solve = als_freq._solve_rows
count = {"rows": 0}


def failing_solve(lo, hi):
    count["rows"] += 1
    if count["rows"] == 5:
        raise Stop()
    return real_solve(lo, hi)


als_freq._solve_rows = failing_solve
try:
    als_freq.fit(
        inter,
        params1,
        d_r / "x.npy",
        d_r / "y.npy",
        d_r / "s.json",
        m_u,
        m_i,
        workers=1,
        log=quiet,
    )
    stopped2 = False
except Stop:
    stopped2 = True
finally:
    als_freq._solve_rows = real_solve
Xr, Yr, hr = als_freq.fit(
    inter,
    params1,
    d_r / "x.npy",
    d_r / "y.npy",
    d_r / "s.json",
    m_u,
    m_i,
    workers=1,
    log=quiet,
)
check(
    stopped
    and stopped2
    and np.array_equal(np.asarray(Xr), np.asarray(X))
    and np.array_equal(np.asarray(Yr), np.asarray(Y))
    and len(hr) == 10
    and all(h.get("callback_done") for h in hr),
    "a stop inside the callback and a stop inside a half-step both resume "
    "to the same bits as an uninterrupted fit",
)
# A committed factor file that no longer matches the fit state.
del Xr, Yr
np.save(d_r / "x.npy", np.zeros((NU, 6), dtype=np.float32))
lines_c = []
Xc, Yc, hc = als_freq.fit(
    inter,
    params1,
    d_r / "x.npy",
    d_r / "y.npy",
    d_r / "s.json",
    m_u,
    m_i,
    workers=1,
    log=lines_c.append,
)
check(
    any("Starting fresh" in x for x in lines_c)
    and len(hc) == 10
    and np.array_equal(np.asarray(Xc), np.asarray(X))
    and np.array_equal(np.asarray(Yc), np.asarray(Y)),
    "a factor file changed outside fit() makes it start fresh, to the same " "bits",
)
del Xc, Yc
lines = []
params_nu = als_freq.FreqParams(
    factors=6, reg=0.004, alpha=alpha, iterations=10, seed=5, nu=0.5
)
als_freq.fit(
    inter,
    params_nu,
    d_r / "x.npy",
    d_r / "y.npy",
    d_r / "s.json",
    m_u,
    m_i,
    workers=1,
    log=lines.append,
)
check(
    any("Starting fresh" in x for x in lines)
    and als_freq.fit_key(inter, params_nu)["params"]["nu"] == 0.5
    and als_freq.fit_key(inter, params_nu)["freq_solver"] == als_freq.FREQ_VERSION,
    "a changed nu starts a fresh fit, because the fit key holds nu and the "
    "solver version",
)
try:
    als_freq.fit(
        inter,
        params1,
        d_r / "x.npy",
        d_r / "y.npy",
        d_r / "s.json",
        m_u[:-1],
        m_i,
        workers=1,
        log=quiet,
    )
    refused = False
except ValueError as e:
    refused = "do not match" in str(e)
check(refused, "fit() refuses masses of other data")
real_half = als_freq._half


def rising_half(values):
    it = iter(values)

    def half(*a, **k):
        return next(it)

    return half


rose = []
for values, text in (
    ([10.0, 9.0, 9.5, 8.0], "rose by"),
    ([10.0, float("nan")], "not finite"),
):
    als_freq._half = rising_half(values)
    d_x = TMP / f"fit_rise_{len(rose)}"
    try:
        als_freq.fit(
            inter,
            params1,
            d_x / "x.npy",
            d_x / "y.npy",
            d_x / "s.json",
            m_u,
            m_i,
            workers=1,
            log=quiet,
        )
        rose.append(False)
    except als.ObjectiveRose as e:
        rose.append(text in str(e))
    finally:
        als_freq._half = real_half
als_freq._half = rising_half([10.0, 9.0, 9.0 * (1 + 0.5e-4), 8.0, 7.0, 6.0])
try:
    d_x = TMP / "fit_rise_small"
    _, _, h_small = als_freq.fit(
        inter,
        als_freq.FreqParams(
            factors=6, reg=0.004, alpha=alpha, iterations=3, seed=5, nu=1.0
        ),
        d_x / "x.npy",
        d_x / "y.npy",
        d_x / "s.json",
        m_u,
        m_i,
        workers=1,
        log=quiet,
    )
    small_ok = len(h_small) == 3
except als.ObjectiveRose:
    small_ok = False
finally:
    als_freq._half = real_half
check(
    rose == [True, True] and small_ok,
    "fit() stops when the objective rises by more than RISE_TOLERANCE or "
    "is not finite, and goes on after a rise below it",
)

# 6: the search rule


def fake_runner(score_of):
    """run_config stand-in: score_of(cell, factors, nu, k) gives the
    score. Records every call."""
    calls = []

    def run_config(cell, factors, nu, reg, k):
        calls.append((cell, factors, nu, k, reg))
        return {
            "name": f"{cell}{factors}{nu}{k}",
            "cell": cell,
            "factors": factors,
            "nu": nu,
            "reg": reg,
            "k": k,
            "score": score_of(cell, factors, nu, k),
        }

    return run_config, calls


peak = {"D1": 2, "D2": -1, "D3": 0}


def hill(cell, factors, nu, k):
    return 0.02 - 0.001 * abs(k - peak[cell])


rc, calls = fake_runner(hill)
entries = td.search_cell(rc, "D1", 128, 0.0, 100.0, 3)
check(
    [c[3] for c in calls] == [0, 1, 2, 3]
    and abs(calls[1][4] - 100 * 10**0.5) < 1e-9
    and abs(calls[2][4] - 1000.0) < 1e-9
    and td.best_of(entries)["k"] == 2,
    "search_cell(): upward while each step beats the best (k = 0, 1, 2, "
    "3), reg on the grid start * 10^(k/2)",
)
peak["D1"] = -2
rc, calls = fake_runner(hill)
td.search_cell(rc, "D1", 128, 0.0, 100.0, 3)
check(
    [c[3] for c in calls] == [0, 1, -1, -2, -3],
    "search_cell(): when k = +1 does not beat k = 0, downward from -1 "
    "until a step does not beat the best",
)
peak["D1"] = -9
rc, calls = fake_runner(hill)
td.search_cell(rc, "D1", 128, 0.0, 100.0, 3)
peak["D1"] = 9
rc2, calls2 = fake_runner(hill)
td.search_cell(rc2, "D1", 128, 0.0, 100.0, 2)
check(
    [c[3] for c in calls] == [0, 1, -1, -2, -3] and [c[3] for c in calls2] == [0, 1, 2],
    "search_cell(): at most 3 steps in a direction, or 2 when asked",
)
flat = lambda cell, factors, nu, k: 0.02
rc, calls = fake_runner(flat)
entries = td.search_cell(rc, "D1", 128, 0.0, 100.0, 3)
check(
    [c[3] for c in calls] == [0, 1, -1] and td.best_of(entries)["k"] == 0,
    "search_cell(): a tie does not continue, and best_of() keeps the "
    "earliest of equal scores",
)

peak.update(D1=1, D2=0, D3=0)
starts = []


def d2_start(reg):
    starts.append(reg)
    return reg / 7.0


def gate_scores(d2_gain):
    def score(cell, factors, nu, k):
        base = {"D1": 0.0175, "D2": 0.0175 + d2_gain, "D3": 0.019}[cell]
        return base - 0.001 * abs(k - peak[cell])

    return score


rc, calls = fake_runner(gate_scores(0.0005))
ac = {"reg": 100.0, "score": 0.01701}
entries, gate = td.run_cells(rc, ac, d2_start)
d3 = [c for c in calls if c[0] == "D3"]
check(
    abs(starts[0] - 100 * 10**0.5) < 1e-9
    and abs([c for c in calls if c[0] == "D2"][0][4] - 100 * 10**0.5 / 7) < 1e-9
    and gate["passed"]
    and gate["best_128"] == "D21281.00"
    and gate["best_128_score"] == 0.0175 + 0.0005
    and d3
    and all(c[1] == 256 and c[2] == 1.0 for c in d3)
    and abs(d3[0][4] - 100 * 10**0.5 / 7) < 1e-9
    and abs(gate["threshold"] - 0.01701 * 1.02) < 1e-15,
    "run_cells(): D2 starts from D1's best reg through d2_start(), the gate "
    "passes above 1.02 times the phase A to C score, and D3 runs with 256 "
    "factors, the nu of the better 128-factor cell and its best reg",
)
rc, calls = fake_runner(gate_scores(-0.0005))
entries, gate = td.run_cells(
    rc, {"reg": 100.0, "score": 0.0175}, lambda reg: reg, sesoi=0.0
)
check(
    not gate["passed"]
    and not [c for c in calls if c[0] == "D3"]
    and gate["best_128"] == "D11280.01"
    and gate["threshold"] == 0.0175
    and gate["best_128_score"] == 0.0175,
    "run_cells(): a best 128-factor score equal to the threshold does not "
    "pass the gate, and D3 does not run",
)
rc, calls = fake_runner(gate_scores(-0.0005))
entries, gate = td.run_cells(
    rc, {"reg": 100.0, "score": 0.0175 / 1.0199}, lambda reg: reg
)
check(
    not gate["passed"] and not [c for c in calls if c[0] == "D3"],
    "run_cells(): with SESOI 2 percent, a gain of 1.99 percent does not "
    "pass the gate",
)
rc, calls = fake_runner(gate_scores(-0.0005))
entries, gate = td.run_cells(rc, {"reg": 100.0, "score": 0.01}, lambda reg: reg / 3.0)
d3 = [c for c in calls if c[0] == "D3"]
check(
    gate["passed"]
    and gate["best_128"] == "D11280.01"
    and d3
    and all(c[1] == 256 and c[2] == 0.0 for c in d3)
    and abs(d3[0][4] - 100 * 10**0.5) < 1e-9
    and [c[3] for c in d3] == [0, 1, -1],
    "run_cells(): when the nu = 0 cell wins at 128 factors, D3 runs with "
    "nu = 0 from D1's best reg",
)
check(
    td._need_bytes(inter, 5) == int(3 * (NU + NI) * 5 * 4 * 1.2)
    and td._need_bytes(inter, 10) == int(3 * (NU + NI) * 10 * 4 * 1.2),
    "the disk need of a config is 3 float32 factor sets with a 20 percent " "margin",
)

# 7: run_phase end to end
model = TMP / "model"
work = TMP / "scratch" / "als_tune_d"
best_dir = model / "als_d"
model.mkdir(parents=True)
users = np.arange(1, 60)
picks = np.stack(
    [rng.integers(1, NI, len(users)), rng.integers(1, NI, len(users))], axis=1
)
ac_results = {
    "winner": {
        "alpha": 2.0,
        "reg": 0.3,
        "score": 0.0,
        "seed": 4,
        "factors": 64,
        "iterations": 2,
    },
    "reference": {"score": 0.001},
}
masses = (m_u, m_i)
real_f128, real_f256 = td.FACTORS_128, td.FACTORS_256
real_steps = dict(td.MAX_STEPS)
td.FACTORS_128, td.FACTORS_256 = 5, 7
td.MAX_STEPS.clear()
td.MAX_STEPS.update({5: 2, 7: 1})
fits = []
real_fit = als_freq.fit


def counting_fit(*a, **k):
    fits.append(a[1])
    return real_fit(*a, **k)


als_freq.fit = counting_fit
try:
    res = td.run_phase(
        inter,
        users,
        picks,
        masses,
        ac_results,
        model / "als_tuning_d.json",
        work,
        best_dir,
        workers=1,
        say=quiet,
        max_iter=5,
        patience=2,
    )
    n_fits = len(fits)
    left_first = sorted(p.relative_to(work).as_posix() for p in work.rglob("*.npy"))
    fits.clear()
    res2 = td.run_phase(
        inter,
        users,
        picks,
        masses,
        ac_results,
        model / "als_tuning_d.json",
        work,
        best_dir,
        workers=1,
        say=quiet,
        max_iter=5,
        patience=2,
    )
    refit = len(fits)
finally:
    als_freq.fit = real_fit
stored = json.loads((model / "als_tuning_d.json").read_text())
order = [stored["configs"][n] for n in stored["order"]]
win = td.best_of(order)
check(
    n_fits == len(order)
    and refit == 0
    and stored["gate"]["passed"]
    and res2["winner"] == res["winner"]
    and stored["winner"]["config"] == win["name"]
    and all(e["finished"] for e in order)
    and {e["cell"] for e in order} == {"D1", "D2", "D3"},
    f"run_phase() runs {n_fits} configs over D1, D2 and D3, the winner is "
    f"the best of them, and a second call fits nothing",
)
best = json.loads((best_dir / "best.json").read_text())
files_ok = all(
    (best_dir / best["files"][t]["file"]).exists()
    and als._file_sha(best_dir / best["files"][t]["file"]) == best["files"][t]["sha256"]
    for t in ("user", "item")
)
leftover = sorted(p.name for p in best_dir.iterdir() if p.name != "best.json")
check(
    best["config"] == win["name"]
    and best["iteration"] == win["best_iteration"]
    and abs(best["score"] - win["score"]) == 0
    and files_ok
    and leftover == sorted(best["files"][t]["file"] for t in ("user", "item")),
    "best.json names the winner's best iteration, its files exist with "
    "the stated SHA-256, and no older factor files remain",
)
wp = als_freq.FreqParams(
    factors=win["factors"],
    reg=win["reg"],
    alpha=2.0,
    iterations=win["best_iteration"],
    seed=4,
    nu=win["nu"],
)
d_fresh = TMP / "fresh"
Xw, Yw, _ = als_freq.fit(
    inter,
    wp,
    d_fresh / "x.npy",
    d_fresh / "y.npy",
    d_fresh / "s.json",
    m_u,
    m_i,
    workers=1,
    log=quiet,
)
check(
    np.array_equal(np.load(best_dir / best["files"]["user"]["file"]), np.asarray(Xw))
    and np.array_equal(
        np.load(best_dir / best["files"]["item"]["file"]), np.asarray(Yw)
    ),
    "the kept factors equal a fresh fit of the winner stopped at its best "
    "iteration, bit for bit",
)
d1 = [e for e in order if e["cell"] == "D1"]
d2 = [e for e in order if e["cell"] == "D2"]
F_u = res["median_frequency"]["users"]
F_i = res["median_frequency"]["items"]
start_d2 = td.best_of(d1)["reg"] / math.sqrt(F_u * F_i)
check(
    abs(d1[0]["reg"] - 0.3) < 1e-15
    and abs(d2[0]["reg"] - start_d2) < 1e-15 * start_d2
    and d2[0]["nu"] == 1.0
    and d1[0]["nu"] == 0.0,
    "the cells start where the protocol says: D1 at the phase A to C reg, "
    "D2 at D1's best reg over sqrt(F_u F_i)",
)
has_u = np.diff(inter.u[0]) > 0
has_i = np.diff(inter.i[0]) > 0
f_u = 2.0 * m_u_dense[has_u] / 1000.0 + (NI - 1)
f_i = 2.0 * m_i_dense[has_i] / 1000.0 + (NU - 1)
q = res["frequency_quantiles"]
check(
    abs(F_u - float(np.median(f_u))) < 1e-9
    and abs(F_i - float(np.median(f_i))) < 1e-9
    and q["users"]["q0"] == float(f_u.min())
    and q["items"]["q1"] == float(f_i.max()),
    "the median frequencies and their quantiles follow from the dense "
    "listen masses with the phase A to C alpha (users over n_items - 1, "
    "items over n_users - 1)",
)
check(
    all(e["iterations_run"] == min(5, e["best_iteration"] + 2) for e in order),
    "each config stops 2 iterations after its best one, or at the " "iteration limit",
)
check(
    all(
        e["score"] == max(e["scores"])
        and e["best_iteration"] == e["scores"].index(max(e["scores"])) + 1
        for e in order
    )
    and all(len(e["scores"]) <= 5 for e in order),
    "each config's score is its best iteration's, at most 5 iterations " "here",
)
td.write_report(res, model / "reports" / "als_tuning_d.md")
rep = (model / "reports" / "als_tuning_d.md").read_text()
check(
    all(f"| {e['cell']} |" in rep for e in order)
    and ";" not in rep
    and f"Phase D winner: {win['name']}" in rep
    and "For P14: the phase D winner." in rep
    and "The 256-factor cell ran." in rep
    and "- f of the users with data: minimum" in rep
    and "- f of the items with data: minimum" in rep,
    "the report lists every config, the frequency spread, the gate, the "
    "winner and the choice for P14, without a semicolon",
)
tree = ast.parse(Path(td.__file__).read_text())
names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
imported = {
    a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names
}
check(
    "CURATED" not in names | attrs | imported
    and "write_als_params" not in names | attrs
    and not hasattr(td, "CURATED"),
    "als_tune_d.py never uses CURATED or als_tune.write_als_params(), so it "
    "cannot write CURATED/als_params.json",
)
ok_names = all(
    e["name"] == td.config_name(e["factors"], e["nu"], 2.0, e["reg"]) for e in order
)
check(
    ok_names
    and left_first == []
    and not any(
        (work / e["name"] / f).exists()
        for e in order
        for f in ("user_factors.npy", "item_factors.npy")
    ),
    "no finished config leaves a factor file in the work directory",
)
# A stop between the last save and the removal of a config's factor files.
first = order[0]
for f in ("user_factors.npy", "item_factors.npy"):
    (work / first["name"] / f).write_bytes(b"stale")
td.run_phase(
    inter,
    users,
    picks,
    masses,
    ac_results,
    model / "als_tuning_d.json",
    work,
    best_dir,
    workers=1,
    say=quiet,
    max_iter=5,
    patience=2,
)
check(
    not (work / first["name"] / "user_factors.npy").exists()
    and not (work / first["name"] / "item_factors.npy").exists(),
    "a finished config removes factor files that a stop left behind",
)

# A stop inside keep_best(), between the file copies and best.json.
model2 = TMP / "model2"
work2 = TMP / "scratch2"
best2 = model2 / "als_d"
model2.mkdir()
real_write = td.atomic_json_write
state = {"n": 0}


def failing_write(doc, path):
    if Path(path).name == "best.json":
        state["n"] += 1
        if state["n"] == 2:
            raise Stop()
    return real_write(doc, path)


td.atomic_json_write = failing_write
try:
    td.run_phase(
        inter,
        users,
        picks,
        masses,
        ac_results,
        model2 / "als_tuning_d.json",
        work2,
        best2,
        workers=1,
        say=quiet,
        max_iter=5,
        patience=2,
    )
    stopped = False
except Stop:
    stopped = True
finally:
    td.atomic_json_write = real_write
at_stop = json.loads((best2 / "best.json").read_text())
n_at_stop = len([p for p in best2.iterdir() if p.name.endswith("_factors.npy")])
at_stop_ok = all(
    (best2 / at_stop["files"][t]["file"]).exists() for t in ("user", "item")
)
# Leftovers of other stops, and a file that is not a factor file.
(best2 / "f5_nu0_alpha2_reg9_it3_user_factors.npy.tmp").write_bytes(b"x")
(best2 / "f5_nu0_alpha2_reg9_it3_item_factors.npy").write_bytes(b"x")
(best2 / "notes.txt").write_text("keep me")
res3 = td.run_phase(
    inter,
    users,
    picks,
    masses,
    ac_results,
    model2 / "als_tuning_d.json",
    work2,
    best2,
    workers=1,
    say=quiet,
    max_iter=5,
    patience=2,
)
best3 = json.loads((best2 / "best.json").read_text())
left3 = sorted(
    p.name for p in best2.iterdir() if p.name not in ("best.json", "notes.txt")
)
check(
    stopped
    and n_at_stop == 4
    and at_stop_ok
    and res3["winner"] == res["winner"]
    and best3["config"] == best["config"]
    and best3["iteration"] == best["iteration"]
    and left3 == sorted(best3["files"][t]["file"] for t in ("user", "item"))
    and (best2 / "notes.txt").read_text() == "keep me",
    "a stop inside keep_best() leaves best.json naming files that exist. "
    "The resumed run reaches the same winner, removes every other factor "
    "file and keeps other files",
)
real_usage = shutil.disk_usage


class Usage:
    def __init__(self, free):
        self.free = free


td.shutil.disk_usage = lambda p: Usage(10)
try:
    try:
        td.run_phase(
            inter,
            users,
            picks,
            masses,
            ac_results,
            TMP / "model3.json",
            TMP / "scratch3",
            TMP / "best3",
            workers=1,
            say=quiet,
            max_iter=5,
            patience=2,
        )
        disk_refused = False
    except OSError as e:
        disk_refused = "of free disk" in str(e)
finally:
    td.shutil.disk_usage = real_usage
check(
    disk_refused,
    "a config stops before its fit when the disk is too " "full for its files",
)


# 8: ties and early stops
# Every pick at rank 7 in every iteration: all scores tie.
def rank7(X, Y, inter_, users_, picks_, **kw):
    return np.full(np.asarray(picks_).shape, 7, dtype=np.int64)


s7 = als_tune.score_picks(np.full(picks.shape, 7, dtype=np.int64))[0]
ac_tie = {"winner": dict(ac_results["winner"], score=s7)}
res_t = td.run_phase(
    inter,
    users,
    picks,
    masses,
    ac_tie,
    TMP / "model_t" / "als_tuning_d.json",
    TMP / "work_t",
    TMP / "model_t" / "als_d",
    workers=1,
    say=quiet,
    max_iter=5,
    patience=2,
    rank=rank7,
)
order_t = [res_t["configs"][n] for n in res_t["order"]]
best_t = json.loads((TMP / "model_t" / "als_d" / "best.json").read_text())
check(
    abs(s7 - 1 / 3) < 1e-15
    and [(e["cell"], e["k"]) for e in order_t]
    == [("D1", 0), ("D1", 1), ("D1", -1), ("D2", 0), ("D2", 1), ("D2", -1)]
    and all(e["iterations_run"] == 3 and e["best_iteration"] == 1 for e in order_t)
    and res_t["winner"]["config"] == order_t[0]["name"]
    and best_t["config"] == order_t[0]["name"]
    and best_t["iteration"] == 1
    and not res_t["gate"]["passed"]
    and res_t["for_p14"] == "phase A to C winner (P6)",
    "with all scores tied: no cell goes past k = -1, every config stops "
    "at iteration 3, the first config's first iteration wins and is kept, "
    "and a phase D score equal to the phase A to C score fails the gate "
    "and leaves P14 with P6",
)
# A stop inside the first ranking, before any factors are kept.
calls_r = {"n": 0}


def rank_stop_once(X, Y, inter_, users_, picks_, **kw):
    calls_r["n"] += 1
    if calls_r["n"] == 1:
        raise Stop()
    return als.rank_items(X, Y, inter_, users_, picks_, **kw)


try:
    td.run_phase(
        inter,
        users,
        picks,
        masses,
        ac_results,
        TMP / "model_s" / "als_tuning_d.json",
        TMP / "work_s",
        TMP / "model_s" / "als_d",
        workers=1,
        say=quiet,
        max_iter=5,
        patience=2,
        rank=rank_stop_once,
    )
    stopped_s = False
except Stop:
    stopped_s = True
saved_s = json.loads((TMP / "model_s" / "als_tuning_d.json").read_text())
res_s = td.run_phase(
    inter,
    users,
    picks,
    masses,
    ac_results,
    TMP / "model_s" / "als_tuning_d.json",
    TMP / "work_s",
    TMP / "model_s" / "als_d",
    workers=1,
    say=quiet,
    max_iter=5,
    patience=2,
    rank=rank_stop_once,
)
check(
    stopped_s
    and saved_s["best"] == {"score": None}
    and res_s["winner"] == res["winner"],
    "a stop in the first ranking leaves no kept factors (best score "
    "stored as null), and the next call reaches the same winner",
)
td.FACTORS_128, td.FACTORS_256 = real_f128, real_f256
td.MAX_STEPS.clear()
td.MAX_STEPS.update(real_steps)
check(
    td.MAX_STEPS == {128: 3, 256: 2}
    and td.MAX_CONFIGS == 14
    and td.PATIENCE == als_tune.PATIENCE
    and td.MAX_ITER == als_tune.MAX_ITER
    and td.SESOI == 0.02,
    "the declared constants: 3 steps with 128 factors, 2 with 256, at most "
    "14 configs, the iterations and patience of phases A to C, SESOI 2 "
    "percent",
)
shutil.rmtree(TMP, ignore_errors=True)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("all checks passed")
