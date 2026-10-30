"""Checks for pipeline/als.py, pipeline/selection.py and als_tune.py on
small synthetic data. No real data and no DuckDB needed.

1. One half-step equals the dense closed-form solve for every row.
2. The objective that _half() reports equals the dense objective.
3. The dense objective never rises over ten iterations.
4. Rows without data solve to exactly zero.
5. Two worker processes give bit-identical factors to one.
6. A fit stopped after iteration 3 equals a fresh 3-iteration fit, bit
   for bit. This is what lets P6 reproduce the tuned model exactly.
7. Interrupted runs resume to the same factors as an uninterrupted run,
   whether the stop happens inside a half-step or inside the callback.
8. A changed train pair, a changed parameter or a tampered factor file
   each force a fresh start.
9. rank_items() equals a brute-force stable argsort rank, with ties.
10. selection.select() ignores input order, never returns an excluded
    user and returns exactly SIZE users.
11. run_search() follows the protocol: phase A, then phase B at the best
    alpha, early stopping, the winner rule, and a second call that fits
    nothing new.
12. An item-major CSR built the way P6 built it (per train file, then
    concatenated) scores far below 1.0, and fit() refuses it. Correct
    pairs with wrong data values also score below 1.0.
13. transpose_user_major() equals the exact transpose, bit for bit, with
    one bucket, many buckets, and buckets smaller than the largest row.
14. The transposed cache is reused on a second call and rebuilt when the
    user-major fingerprint changes.
15. Progress lines appear for the fingerprint, both half-steps, the
    ranking and the transpose, and they change no result: a fit with
    progress on equals a silent fit, bit for bit. The solver and
    transpose versions are unchanged, so a run that the previous version
    of this file started still resumes.
16. threadpoolctl is installed. rank_items() lifts fit()'s one-thread
    BLAS limit for its matmul, and the ranks and scores are the same bits
    with one thread and with all cores.
17. Fork workers started right after a multi-threaded BLAS call finish
    (no hang) and give the same factors.
18. In a fresh process, als_tune.setup_logging() sends INFO lines to the
    console and the log file exactly once each, with UTC time stamps,
    also when called twice or after another console handler exists.
19. als_tune.main() prints its first lines before it reads any data, and
    an error goes to the log file with its traceback.
20. als_status.py reports an empty state and a finished search without
    errors.
21. Phase C, with a stand-in fit and stand-in scores so that the rule
    itself is tested: the configs it runs, their order, where each axis
    stops (a better value past the stop point is never tried), resume,
    no extension for an interior winner, and the mirror case (largest
    alpha, smallest reg). The first measurement of P6's old item-major
    CSR survives later runs.
22. write_als_params() writes the winner and logs what it replaced.
23. The P6 helpers in als.py: user_major_fingerprint() equals
    Interactions.fingerprint(), from_user_major() builds or reuses the
    same cache that als_tune.py uses, and fit_state_problems() passes a
    finished fit and catches other parameters, too few iterations, other
    data, a swapped factor file, a nonzero empty row and a missing state.

Usage, from the repository root:
    python3 test_als.py
"""

import calendar
import contextlib
import io
import json
import os
import shutil
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from pipeline import als, selection

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


def build(P, R):
    """Interactions in P6's layout from dense 0/1 P and listen counts R.
    Row 0 and column 0 are the phantom slots and must be empty."""
    data_dense = np.where(P > 0, np.round(1000.0 * np.log1p(R)), 0)

    def csr(M, D):
        rows, cols = np.nonzero(M)
        indptr = np.zeros(M.shape[0] + 1, dtype=np.int64)
        np.add.at(indptr, rows + 1, 1)
        indptr = np.cumsum(indptr)
        return (indptr, cols.astype(np.int32), D[rows, cols].astype(np.uint16))

    u = csr(P, data_dense)
    i = csr(P.T, data_dense.T)
    return als.Interactions(*u, *i)


def dense_conf(inter, alpha, dtype=np.float32):
    """c exactly as the solver computes it (in ACCUM_DTYPE), as a dense
    matrix."""
    C = np.ones((inter.n_users, inter.n_items), dtype=np.float64)
    P = np.zeros_like(C)
    indptr, indices, data = inter.u
    scale = dtype(als.DATA_SCALE)
    for u in range(inter.n_users):
        s, e = indptr[u], indptr[u + 1]
        c = 1.0 + alpha * (np.asarray(data[s:e], dtype=dtype) * scale)
        C[u, indices[s:e]] = c.astype(np.float64)
        P[u, indices[s:e]] = 1.0
    return C, P


def dense_objective(X, Y, C, P, reg):
    X, Y = np.asarray(X, np.float64), np.asarray(Y, np.float64)
    S = X @ Y.T
    return float((C * (P - S) ** 2).sum() + reg * ((X**2).sum() + (Y**2).sum()))


rng = np.random.default_rng(11)
NU, NI = 140, 110
P = (rng.random((NU, NI)) < 0.12).astype(np.float64)
P[0, :] = 0
P[:, 0] = 0
P[5, :] = 0  # a user without data
P[:, 7] = 0  # an item without data
R = np.where(P > 0, rng.pareto(1.2, (NU, NI)) * 20 + 1, 0.0)
inter = build(P, R)
params = als.ALSParams(factors=8, reg=0.5, alpha=20.0, iterations=10, seed=3)
C, Pd = dense_conf(inter, params.alpha)
tmp = Path(tempfile.mkdtemp(prefix="test_als_"))


def paths(tag):
    d = tmp / tag
    return d / "x.npy", d / "y.npy", d / "state.json"


# 1 and 2: one half-step against the dense closed form, with float64
# accumulation so the comparison is tight, then with the default float32.
for accum, tol in ((np.float64, 1e-6), (np.float32, 1e-3)):
    als.ACCUM_DTYPE = accum
    C, Pd = dense_conf(inter, params.alpha, accum)
    r2 = np.random.default_rng(0)
    X = (r2.standard_normal((NU, 8)) * 0.1).astype(np.float32)
    Y = (r2.standard_normal((NI, 8)) * 0.1).astype(np.float32)
    J = als._half(inter.u, X, Y, params.reg, params.alpha, workers=1)
    Y64 = Y.astype(np.float64)
    worst = 0.0
    for u in range(NU):
        Cu = C[u]
        A = (Y64 * Cu[:, None]).T @ Y64 + params.reg * np.eye(8)
        b = (Y64 * Cu[:, None]).T @ Pd[u]
        x = np.linalg.solve(A, b) if Pd[u].any() else np.zeros(8)
        worst = max(worst, float(np.abs(x - X[u]).max()))
    Jd = dense_objective(X, Y, C, Pd, params.reg)
    check(
        worst < tol,
        f"[{np.dtype(accum).name}] users half-step equals "
        f"the dense closed form (max gap {worst:.1e})",
    )
    check(
        abs(J - Jd) <= 1e-9 * abs(Jd),
        f"[{np.dtype(accum).name}] _half() objective equals the dense "
        f"objective (relative gap {abs(J - Jd) / abs(Jd):.1e})",
    )
als.ACCUM_DTYPE = np.float32
C, Pd = dense_conf(inter, params.alpha, np.float32)

# 3 and 4: monotone over ten iterations, empty rows exactly zero.
dense_js = []
X, Y, hist = als.fit(
    inter,
    params,
    *paths("mono"),
    workers=1,
    log=quiet,
    on_iteration=lambda it, X, Y, h: dense_js.append(
        dense_objective(X, Y, C, Pd, params.reg)
    ),
)
rises = np.diff(dense_js) / np.abs(np.array(dense_js[:-1]))
check(
    len(dense_js) == 10 and bool(np.all(rises <= 1e-12)),
    f"dense objective never rises over 10 iterations (largest relative "
    f"change {rises.max():.1e})",
)
check(
    all(
        abs(h["objective_after_items"] - d) <= 1e-9 * abs(d)
        for h, d in zip(hist, dense_js)
    ),
    "reported objective equals the dense objective at every iteration",
)
check(
    not np.any(X[0]) and not np.any(X[5]) and not np.any(Y[0]) and not np.any(Y[7]),
    "rows without data (phantom slots, user 5, item 7) are exactly zero",
)

# 5: fork workers give bit-identical results.
X1, Y1, _ = als.fit(inter, params, *paths("w1"), workers=1, log=quiet)
X2, Y2, _ = als.fit(inter, params, *paths("w2"), workers=2, log=quiet)
check(
    np.array_equal(X1, X2) and np.array_equal(Y1, Y2),
    "2 worker processes give bit-identical factors to 1",
)

# 6: stopping at iteration 3 equals a fresh 3-iteration fit.
p3 = als.ALSParams(factors=8, reg=0.5, alpha=20.0, iterations=3, seed=3)
Xa, Ya, _ = als.fit(inter, p3, *paths("three"), workers=1, log=quiet)
Xb, Yb, hb = als.fit(
    inter,
    params,
    *paths("stop3"),
    workers=1,
    log=quiet,
    on_iteration=lambda it, X, Y, h: {"stop": it == 3},
)
check(
    len(hb) == 3 and np.array_equal(Xa, Xb) and np.array_equal(Ya, Yb),
    "a fit stopped after iteration 3 equals a fresh 3-iteration fit",
)

# 7: interruptions resume to the uninterrupted result.
real_half = als._half
calls = {"n": 0}


def failing_half(*a, **k):
    calls["n"] += 1
    if calls["n"] == 5:  # inside iteration 3
        raise KeyboardInterrupt
    return real_half(*a, **k)


als._half = failing_half
try:
    als.fit(inter, params, *paths("int_half"), workers=1, log=quiet)
    check(False, "the injected interruption did not happen")
except KeyboardInterrupt:
    pass
als._half = real_half
Xr, Yr, hr = als.fit(inter, params, *paths("int_half"), workers=1, log=quiet)
check(
    len(hr) == 10 and np.array_equal(Xr, X1) and np.array_equal(Yr, Y1),
    "a stop inside a half-step resumes to the uninterrupted factors",
)

seen = []


def failing_callback(it, X, Y, h):
    if it == 4 and not seen:
        seen.append(it)
        raise KeyboardInterrupt
    return {"marker": it}


try:
    als.fit(
        inter,
        params,
        *paths("int_cb"),
        workers=1,
        log=quiet,
        on_iteration=failing_callback,
    )
except KeyboardInterrupt:
    pass
Xc, Yc, hc = als.fit(
    inter, params, *paths("int_cb"), workers=1, log=quiet, on_iteration=failing_callback
)
check(
    len(hc) == 10
    and all(h.get("marker") == h["iteration"] for h in hc)
    and np.array_equal(Xc, X1),
    "a stop inside the callback reruns it for that iteration, then "
    "continues to the same factors",
)

# 8: fresh starts when anything the fit depends on changes.
messages = []
P_changed = P.copy()
R_changed = R.copy()
k = tuple(np.argwhere(P > 0)[0])
R_changed[k] += 50.0
inter_changed = build(P_changed, R_changed)
als.fit(inter_changed, params, *paths("w1"), workers=1, log=messages.append)
check(
    any("Starting fresh" in m for m in messages),
    "a changed train pair forces a fresh start",
)
messages.clear()
p_other = als.ALSParams(factors=8, reg=0.6, alpha=20.0, iterations=10, seed=3)
als.fit(inter, p_other, *paths("w2"), workers=1, log=messages.append)
check(
    any("Starting fresh" in m for m in messages),
    "a changed parameter forces a fresh start",
)
messages.clear()
xp, yp, sp = paths("mono")
arr = np.load(xp, mmap_mode="r+")
arr[1, 0] += 1.0
arr.flush()
del arr
als.fit(inter, params, xp, yp, sp, workers=1, log=messages.append)
check(
    any("Starting fresh" in m for m in messages),
    "a tampered factor file forces a fresh start",
)

# 9: rank_items() against a brute-force stable argsort, with ties.
Xi = rng.integers(-2, 3, size=(NU, 4)).astype(np.float32)
Yi = rng.integers(-2, 3, size=(NI, 4)).astype(np.float32)
users = np.array([u for u in range(1, NU) if P[u].any()][:40])
picks = np.stack(
    [[rng.choice(np.flatnonzero(P[u] == 0)[1:]) for _ in range(2)] for u in users]
)
got = als.rank_items(Xi, Yi, inter, users, picks, chunk=7)
bad = 0
for k, u in enumerate(users):
    s = (Xi[u] @ Yi.T).astype(np.float64)
    s[0] = -np.inf
    s[np.flatnonzero(P[u])] = -np.inf
    order = np.argsort(-s, kind="stable")
    for j in range(2):
        bad += int(int(np.flatnonzero(order == picks[k, j])[0]) + 1 != got[k, j])
check(
    bad == 0,
    f"rank_items() equals a stable argsort rank on integer "
    f"scores with many ties ({bad} mismatches)",
)

# 10: the selection rule.
cand = np.arange(1, 5001)
excl = rng.choice(cand, size=700, replace=False)
a = selection.select(cand, excl, seed="t", size=1000)
b = selection.select(rng.permutation(cand), rng.permutation(excl), seed="t", size=1000)
check(
    np.array_equal(a, b) and len(a) == 1000 and not np.intersect1d(a, excl).size,
    "select() ignores input order, excludes the eval users, returns SIZE",
)

# 11: the search protocol on a synthetic catalog with held-out picks.
from pipeline import als_tune

P_train = P.copy()
sel_users, sel_picks = [], []
for u in range(1, NU):
    have = np.flatnonzero(P[u])
    if len(have) >= 6:
        held = rng.choice(have, size=2, replace=False)
        P_train[u, held] = 0
        sel_users.append(u)
        sel_picks.append(held)
sel_users = np.array(sel_users)
sel_picks = np.array(sel_picks)
inter_t = build(P_train, R)
log_lines = []
res = als_tune.run_search(
    inter_t,
    sel_users,
    sel_picks,
    tmp / "tuning.json",
    tmp / "tune",
    alphas=(1.0, 20.0),
    reg_a=1.0,
    regs_b=(0.1,),
    factors=6,
    seed=1,
    max_iter=6,
    patience=2,
    workers=1,
    say=log_lines.append,
    phase_c=None,
)
cfgs = list(res["configs"].values())
names = list(res["configs"])
phase_a = [c for c in cfgs if c["phase"] == "A"]
best_alpha = phase_a[als_tune._best_index([c["score"] for c in phase_a])]["alpha"]
ok_order = (
    names[:2] == ["alpha1_reg1_f6", "alpha20_reg1_f6"]
    and cfgs[2]["phase"] == "B"
    and cfgs[2]["alpha"] == best_alpha
    and cfgs[2]["reg"] == 0.1
)
check(ok_order, "run_search() runs phase A, then phase B at the best alpha")
ok_stop = all(
    c["best_iteration"] == als_tune._best_index(c["scores"]) + 1
    and (c["iterations_run"] == 6 or c["iterations_run"] - c["best_iteration"] == 2)
    for c in cfgs
)
check(ok_stop, "each config stops at max_iter or 2 iterations after its best")
win_idx = als_tune._best_index([c["score"] for c in cfgs])
check(
    res["winner"]["alpha"] == cfgs[win_idx]["alpha"]
    and res["winner"]["reg"] == cfgs[win_idx]["reg"]
    and res["winner"]["iterations"] == cfgs[win_idx]["best_iteration"],
    "the winner is the best score over all configs",
)
log_lines.clear()
als_tune.run_search(
    inter_t,
    sel_users,
    sel_picks,
    tmp / "tuning.json",
    tmp / "tune",
    alphas=(1.0, 20.0),
    reg_a=1.0,
    regs_b=(0.1,),
    factors=6,
    seed=1,
    max_iter=6,
    patience=2,
    workers=1,
    say=log_lines.append,
    phase_c=None,
)
check(
    all("already finished" in m for m in log_lines) and len(log_lines) == 3,
    "a second run_search() call fits nothing new",
)
als_tune.write_report(res, tmp / "report.md")
check("Winner:" in (tmp / "report.md").read_text(), "the report is written")


# 12: P6's per-file item-major build is detected and refused.
def p6_style_item_major(inter_u, n_items, n_files=4):
    """P6's _write_orientation for the item-major side: the train files
    hold contiguous user ranges, each file is sorted by (item_id,
    user_id), the files are concatenated, and indptr comes from the
    global item counts."""
    indptr, indices, data = inter_u
    n_users = len(indptr) - 1
    rows = np.repeat(np.arange(n_users), np.diff(indptr))
    edges = np.linspace(0, n_users, n_files + 1).astype(np.int64)
    idx, dat = [], []
    counts = np.zeros(n_items, dtype=np.int64)
    for f in range(n_files):
        m = (rows >= edges[f]) & (rows < edges[f + 1])
        uf, itf, df = rows[m], np.asarray(indices)[m], np.asarray(data)[m]
        o = np.lexsort((uf, itf))
        idx.append(uf[o])
        dat.append(df[o])
        counts += np.bincount(itf, minlength=n_items)
    return (
        np.concatenate(([0], np.cumsum(counts))),
        np.concatenate(idx).astype(np.int32),
        np.concatenate(dat),
    )


bad = als.Interactions(*inter.u, *p6_style_item_major(inter.u, inter.n_items))
frac_bad = als.orientation_real_fraction(bad)
frac_good = als.orientation_real_fraction(inter)
check(
    frac_good == 1.0 and frac_bad < 0.5,
    f"orientation check scores the exact transpose 1.0 and the P6-style "
    f"build {frac_bad:.1%}",
)
try:
    als.fit(bad, params, *paths("refuse"), workers=1, log=quiet)
    check(False, "fit() accepted the P6-style item-major CSR")
except ValueError as exc:
    check("not the transpose" in str(exc), "fit() refuses the P6-style item-major CSR")
except Exception as exc:  # any other failure is a FAIL
    check(
        False,
        f"fit() did not refuse the P6-style item-major CSR up "
        f"front, it failed later with {type(exc).__name__}",
    )
i_ptr, i_idx, i_dat = inter.i
shuffled = als.Interactions(*inter.u, i_ptr, i_idx, rng.permutation(np.asarray(i_dat)))
check(
    als.orientation_real_fraction(shuffled) < 1.0,
    "right pairs with wrong data values score below 1.0",
)

# 13: the out-of-core transpose is exact.
ref_ptr, ref_idx, ref_dat = (np.asarray(a) for a in inter.i)
largest = int(np.diff(ref_ptr).max())
for bucket, chunk in ((10**9, 1 << 20), (200, 97), (max(1, largest // 3), 31)):
    d = tmp / f"transpose_{bucket}_{chunk}"
    als.transpose_user_major(
        inter.u, inter.n_items, d, bucket_entries=bucket, chunk=chunk
    )
    got = [
        np.load(d / n)
        for n in ("csr_item_indptr.npy", "csr_item_indices.npy", "csr_item_data.npy")
    ]
    check(
        np.array_equal(got[0], ref_ptr)
        and np.array_equal(got[1], ref_idx)
        and np.array_equal(got[2], ref_dat),
        f"transpose_user_major() is exact (bucket {bucket}, chunk {chunk})",
    )
fixed = als.Interactions(
    *bad.u,
    *als.load_or_build_item_major(bad.u, bad.n_items, tmp / "cache", bad.fingerprint()),
)
check(
    als.orientation_real_fraction(fixed) == 1.0,
    "the transpose repairs the P6-style build",
)

# 14: the cache is reused, and rebuilt for other data.
messages = []
als.load_or_build_item_major(
    bad.u, bad.n_items, tmp / "cache", bad.fingerprint(), log=messages.append
)
check(
    any("reusing" in m for m in messages), "a second call reuses the cached transpose"
)
messages.clear()
als.load_or_build_item_major(
    inter_changed.u,
    inter_changed.n_items,
    tmp / "cache",
    inter_changed.fingerprint(),
    log=messages.append,
)
check(
    any("building" in m for m in messages),
    "other user-major data rebuilds the transpose",
)

# 15: progress lines, and no effect on results or resume keys.
lines = []
als.PROGRESS_SECONDS = 0.0
Xp, Yp, _ = als.fit(inter, params, *paths("progress"), workers=2, log=lines.append)
fresh = als.Interactions(*inter.u, *inter.i)
fresh.fingerprint(log=lines.append)
als.rank_items(Xp, Yp, inter, users, picks, chunk=7, log=lines.append)
als.transpose_user_major(
    inter.u,
    inter.n_items,
    tmp / "transpose_progress",
    bucket_entries=200,
    chunk=97,
    log=lines.append,
)
als.PROGRESS_SECONDS = 30.0
wanted = [
    "users half: ",
    "items half: ",
    "fingerprint",
    "ranking picks: ",
    "transpose bucket 1/",
    "counting item degrees",
    "left",
    "done in",
    "started",
    "fit done in",
]
missing = [w for w in wanted if not any(w in line for line in lines)]
check(
    not missing,
    f"progress lines appear for every long step " f"(missing: {missing or 'none'})",
)
Xs, Ys, _ = als.fit(inter, params, *paths("silent"), workers=2, log=quiet)
check(
    np.array_equal(Xp, Xs) and np.array_equal(Yp, Ys),
    "a fit with progress lines equals a silent fit, bit for bit",
)
check(
    np.array_equal(Xs, X1) and np.array_equal(Ys, Y1),
    "the committed factors of an earlier fit stay intact after a fresh "
    "start writes into the same directory",
)
check(
    als.SOLVER_VERSION == "exact-rows-v3"
    and als.TRANSPOSE_VERSION == "bucket-transpose-v1",
    "solver and transpose versions are unchanged, so running searches " "still resume",
)
print("      sample progress lines:")
for line in [l for l in lines if "users half: " in l][:2] + [
    l for l in lines if "ranking picks: " in l
][:1]:
    print("        " + line)

# 16: the ranking uses all cores and gives the same bits as one core.
try:
    import threadpoolctl
except ImportError:
    threadpoolctl = None
check(
    threadpoolctl is not None,
    "threadpoolctl is installed (als_tune.py stops without it: pip "
    "install threadpoolctl)",
)
n_cpu = os.cpu_count() or 1
if threadpoolctl is not None:
    real_tl = als.threadpool_limits
    r16 = np.random.default_rng(16)
    NI16 = 400_003
    Y16 = (r16.standard_normal((NI16, 64)) * 0.1).astype(np.float32)
    X16 = (r16.standard_normal((65, 64)) * 0.1).astype(np.float32)

    class _Train:
        def train_items(self, u):
            return (np.arange(1, 5000, dtype=np.int64) * (37 + u)) % NI16

    users16 = np.arange(1, 65)
    picks16 = r16.integers(1, NI16, size=(64, 2))
    seen_limits = []

    def recording_limits(limits=None, *a, **k):
        seen_limits.append(limits)
        return real_tl(limits, *a, **k)

    als.threadpool_limits = recording_limits
    try:
        with real_tl(1):  # the limit that fit() holds around its callback
            r_one = als.rank_items(
                X16, Y16, _Train(), users16, picks16, chunk=16, threads=1
            )
            r_all = als.rank_items(X16, Y16, _Train(), users16, picks16, chunk=16)
    finally:
        als.threadpool_limits = real_tl
    check(
        n_cpu in seen_limits and 1 in seen_limits,
        f"rank_items() sets its own BLAS thread count inside fit()'s "
        f"one-thread limit ({n_cpu} by default)",
    )
    with real_tl(1):
        with real_tl(n_cpu):
            inner = {
                d["num_threads"]
                for d in threadpoolctl.threadpool_info()
                if d["user_api"] == "blas"
            }
            S_all = X16 @ Y16.T
        S_one = X16 @ Y16.T
    check(
        inner == {n_cpu},
        f"the inner limit takes effect: BLAS runs {sorted(inner)} threads "
        f"inside fit()'s one-thread limit, {n_cpu} wanted",
    )
    check(
        np.array_equal(S_one, S_all),
        f"the score matmul gives the same bits with 1 and {n_cpu} threads "
        f"(max gap {float(np.abs(S_one - S_all).max()):.1e})",
    )
    check(
        np.array_equal(r_one, r_all),
        f"rank_items() gives the same ranks with 1 and {n_cpu} threads",
    )

# 17: fork workers right after a threaded BLAS call, as in each iteration
#     that follows a ranking.
np.savez(
    tmp / "inter.npz",
    u0=inter.u[0],
    u1=inter.u[1],
    u2=inter.u[2],
    i0=inter.i[0],
    i1=inter.i[1],
    i2=inter.i[2],
)
fork_code = """
import sys
from pathlib import Path
import numpy as np
from pipeline import als
d = np.load(sys.argv[1])
inter = als.Interactions(d["u0"], d["u1"], d["u2"], d["i0"], d["i1"], d["i2"])
out = Path(sys.argv[2])
a = np.random.default_rng(0).standard_normal((1500, 1500)).astype(np.float32)
def threaded_matmul(it, X, Y, h):
    with als.threadpool_limits(max(2, int(sys.argv[3]))):
        float((a @ a).sum())
threaded_matmul(0, None, None, None)
p = als.ALSParams(factors=8, reg=0.5, alpha=20.0, iterations=10, seed=3)
als.fit(inter, p, out / "x.npy", out / "y.npy", out / "state.json", workers=2,
        on_iteration=threaded_matmul, log=lambda m: None)
"""
try:
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            fork_code,
            str(tmp / "inter.npz"),
            str(tmp / "forked"),
            str(n_cpu),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=os.getcwd(),
    )
    same = (
        r.returncode == 0
        and np.array_equal(np.load(tmp / "forked" / "x.npy"), X1)
        and np.array_equal(np.load(tmp / "forked" / "y.npy"), Y1)
    )
    check(
        same,
        "fork workers after a threaded BLAS call finish and give the "
        "same factors" + ("" if r.returncode == 0 else f" (error: {r.stderr[-300:]})"),
    )
except subprocess.TimeoutExpired:
    check(False, "fork workers after a threaded BLAS call hung for 10 minutes")

# 18: logging in a fresh process, where nothing has configured it.
log_code = """
import logging, sys
from pathlib import Path
if sys.argv[2] == "preconfigured":
    logging.basicConfig()
from pipeline import als_tune
from pipeline import als
from pipeline.common import log
path = Path(sys.argv[1])
als_tune.setup_logging(path)
als_tune.setup_logging(path)
log.info("marker-info")
als.Progress("marker-progress", 10, log.info).finish()
"""
for mode in ("fresh", "preconfigured"):
    lp = tmp / f"log_{mode}" / "als_tune.log"
    # A zone 5.5 hours off UTC, so a local-time stamp fails the UTC check
    # on any machine.
    r = subprocess.run(
        [sys.executable, "-c", log_code, str(lp), mode],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=os.getcwd(),
        env={**os.environ, "TZ": "IST-5:30"},
    )
    text = lp.read_text(encoding="utf-8") if lp.exists() else ""
    stamp = re.search(
        r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d{3} INFO " r"marker-info$", text, re.M
    )
    utc = (
        bool(stamp)
        and abs(
            calendar.timegm(time.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S"))
            - time.time()
        )
        < 300
    )
    check(
        r.returncode == 0
        and r.stderr.count("marker-info") == 1
        and text.count("marker-info") == 1
        and "marker-progress: done in" in r.stderr
        and "marker-progress: done in" in text
        and utc,
        f"[{mode}] log lines reach the console and the log file once each, "
        f"with UTC time stamps",
    )

# 19: main() speaks before it reads data, and logs errors to the file.
main_code = """
import sys
from pathlib import Path
from pipeline import als_tune
als_tune.LOG_PATH = Path(sys.argv[1])
def stop(*a, **k):
    raise RuntimeError("stop-before-data")
als_tune.als.Interactions.from_p6 = stop
als_tune.main([])
"""
lp = tmp / "log_main" / "als_tune.log"
r = subprocess.run(
    [sys.executable, "-c", main_code, str(lp)],
    capture_output=True,
    text=True,
    timeout=120,
    cwd=os.getcwd(),
)
text = lp.read_text(encoding="utf-8") if lp.exists() else ""
check(
    r.returncode == 1
    and "als_tune: started" in r.stderr
    and "als_tune: BLAS" in r.stderr
    and "stopped by an error" in text
    and "RuntimeError: stop-before-data" in text,
    "main() prints its first lines before any data step, and an error "
    "goes to the log file with its traceback",
)

# 20: als_status.py on an empty state and on the finished search from 11.
import als_status

out = {}
for tag, results, work, cache, report in (
    (
        "empty",
        tmp / "none.json",
        tmp / "no_work",
        tmp / "no_cache",
        tmp / "no_report.md",
    ),
    ("finished", tmp / "tuning.json", tmp / "tune", tmp / "cache", tmp / "report.md"),
):
    als_status.RESULTS, als_status.WORK = results, work
    als_status.CACHE, als_status.REPORT = cache, report
    als_status.LOG = tmp / "log_main" / "als_tune.log"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        als_status.main()
    out[tag] = buf.getvalue()
check(
    "none started yet" in out["empty"]
    and "Transposed item-major CSR: not built yet" in out["empty"]
    and "not written yet" in out["empty"]
    and out["finished"].count(": finished.") == 3
    and "Winner:" in out["finished"]
    and "als_params.json" in out["finished"]
    and "Transposed item-major CSR: built" in out["finished"],
    "als_status.py reports an empty state and a finished search",
)
print("      als_status.py on the finished synthetic search:")
for line in out["finished"].splitlines():
    print("        " + line)

# 21: phase C with a stand-in fit. Each config's score peaks at iteration
#     2 at the value in the table, so every config stops at iteration 4.
real_fit, real_rank, real_score = als.fit, als.rank_items, als_tune.score_picks
now, fit_calls = {}, []
blk = {"recall_at_100": 0.0, "ndcg_at_100": 0.0}


def fake_fit(
    inter_, p, x_path, y_path, state_path, workers=None, on_iteration=None, log=None
):
    fit_calls.append((p.alpha, p.reg))
    history = []
    for it in range(1, p.iterations + 1):
        now.update(cfg=(p.alpha, p.reg), it=it)
        history.append({"iteration": it, "seconds": 0.0, "max_relative_rise": None})
        history[-1].update(on_iteration(it, None, None, history) or {})
        if history[-1].get("stop"):
            break
    return None, None, history


def run_c(table, tag, fraction):
    fit_calls.clear()
    als.fit = fake_fit
    als.rank_items = lambda *a, **k: np.zeros((1, 2), dtype=np.int64)
    als_tune.score_picks = lambda ranks: (  # -1 for a config the rule
        table.get(now["cfg"], -1.0) - 0.001 * abs(now["it"] - 2),  # should never reach
        blk,
        blk,
    )
    ref = {
        "score": 0.003,
        "val_last": blk,
        "val_first": blk,
        "p6_item_major_real_fraction": fraction,
    }
    try:
        return als_tune.run_search(
            None,
            np.array([1]),
            np.zeros((1, 2)),
            tmp / f"{tag}.json",
            tmp / tag,
            reference=ref,
            workers=1,
            say=quiet,
        )
    finally:
        als.fit, als.rank_items, als_tune.score_picks = (
            real_fit,
            real_rank,
            real_score,
        )


ab_calls = [(1.0, 1.0), (5.0, 1.0), (20.0, 1.0), (40.0, 1.0)]
table1 = {
    (1.0, 1.0): 0.0167,
    (5.0, 1.0): 0.0160,
    (20.0, 1.0): 0.0131,
    (40.0, 1.0): 0.0111,
    (1.0, 0.1): 0.0166,
    (1.0, 10.0): 0.0169,
    (1.0, 100.0): 0.0175,
    (1.0, 1000.0): 0.0170,
    (1.0, 10000.0): 0.0190,  # better, but past the stop point
    (0.25, 100.0): 0.0178,
    (0.0, 100.0): 0.0172,
}
res1 = run_c(table1, "phase_c1", 0.013)
want1 = ab_calls + [
    (1.0, 0.1),
    (1.0, 10.0),
    (1.0, 100.0),
    (1.0, 1000.0),
    (0.25, 100.0),
    (0.0, 100.0),
]
check(
    fit_calls == want1,
    "phase C walks reg up at alpha 1, stops at the first value without a "
    "gain (10000 never runs), then walks alpha down at the best reg",
)
w1 = res1["winner"]
check(
    (w1["alpha"], w1["reg"], w1["iterations"], w1["phase"]) == (0.25, 100.0, 2, "C")
    and w1["als_params_json"]
    == {
        "factors": 64,
        "iterations": 2,
        "regularization": 100.0,
        "alpha": 0.25,
        "seed": als_tune.SEED,
    }
    and (res1["winner_phases_ab"]["alpha"], res1["winner_phases_ab"]["reg"])
    == (1.0, 10.0),
    "the winner is the best over phases A, B and C, and the A-B winner "
    "stays on record",
)
res1b = run_c(table1, "phase_c1", 1.0)
check(
    fit_calls == []
    and res1b["winner"] == w1
    and res1b["reference"]["p6_item_major_real_fraction"] == 0.013,
    "a second run fits nothing new, keeps the winner, and keeps the first "
    "measurement of P6's old item-major CSR",
)
als_tune.write_report(res1b, tmp / "phase_c_report.md")
rep = (tmp / "phase_c_report.md").read_text()
check(
    "phase C, declared" in rep
    and "Winner of phases A and B: alpha 1, reg 10" in rep
    and "Winner: alpha 0.25, reg 100" in rep
    and rep.index("| C | 1 | 100 |") < rep.index("| C | 0.25 | 100 |"),
    "the report shows the phase C rule, both winners, and phase C rows in " "run order",
)
table3 = {
    (1.0, 1.0): 0.010,
    (5.0, 1.0): 0.020,
    (20.0, 1.0): 0.015,
    (40.0, 1.0): 0.012,
    (5.0, 0.1): 0.018,
    (5.0, 10.0): 0.019,
}
res3 = run_c(table3, "phase_c3", 0.013)
check(
    fit_calls == ab_calls + [(5.0, 0.1), (5.0, 10.0)]
    and (res3["winner"]["alpha"], res3["winner"]["reg"]) == (5.0, 1.0),
    "an interior A-B winner runs no phase C config",
)
table4 = {
    (1.0, 1.0): 0.010,
    (5.0, 1.0): 0.011,
    (20.0, 1.0): 0.012,
    (40.0, 1.0): 0.013,
    (40.0, 0.1): 0.014,
    (40.0, 10.0): 0.012,
    (40.0, 0.01): 0.013,
    (80.0, 0.1): 0.0135,
}
res4 = run_c(table4, "phase_c4", 0.013)
check(
    fit_calls == ab_calls + [(40.0, 0.1), (40.0, 10.0), (40.0, 0.01), (80.0, 0.1)]
    and (res4["winner"]["alpha"], res4["winner"]["reg"]) == (40.0, 0.1),
    "the mirror case walks reg down and alpha up",
)

# 22: writing the winner to als_params.json.
target = tmp / "curated" / "als_params.json"
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(
    '{"factors": 64, "iterations": 6, "regularization": 10.0, ' '"alpha": 40.0}'
)
msgs = []
als_tune.write_als_params(w1, target, msgs.append)
check(
    json.loads(target.read_text()) == w1["als_params_json"]
    and '"alpha": 40.0' in msgs[0]
    and "Before:" in msgs[0],
    "write_als_params() writes the winner and logs the file it replaced",
)

# 23: the P6 helpers in als.py.
fresh23 = als.Interactions(*inter.u, *inter.i)
check(
    als.user_major_fingerprint(inter.u) == fresh23.fingerprint(),
    "user_major_fingerprint() equals Interactions.fingerprint()",
)
msgs = []
shared = als.Interactions(*inter.u, *inter.i)
shared.with_transposed_item_major(tmp / "cache_shared")  # als_tune's path
fu = als.Interactions.from_user_major(
    inter.u, inter.n_items, tmp / "cache_shared", log=msgs.append
)
check(
    any("reusing" in m for m in msgs)
    and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(fu.i, inter.i))
    and fu.fingerprint() == fresh23.fingerprint(),
    "from_user_major() reuses the cache that als_tune.py built, and it "
    "holds the exact transpose",
)
msgs.clear()
als.Interactions.from_user_major(
    inter.u, inter.n_items, tmp / "cache_new", log=msgs.append
)
check(
    any("building" in m for m in msgs),
    "from_user_major() builds the cache when it is missing",
)
pv = als.ALSParams(factors=8, reg=0.5, alpha=20.0, iterations=4, seed=3)
xv, yv, sv = paths("verify")
als.fit(inter, pv, xv, yv, sv, workers=1, log=quiet)
check(
    als.fit_state_problems(inter, pv, xv, yv, sv) == [],
    "fit_state_problems() passes a finished fit",
)
more = als.ALSParams(factors=8, reg=0.5, alpha=20.0, iterations=6, seed=3)
got = als.fit_state_problems(inter, more, xv, yv, sv)
check(
    any("finished iterations" in g for g in got),
    "fit_state_problems() catches a fit with fewer iterations than asked",
)
other = als.ALSParams(factors=8, reg=0.6, alpha=20.0, iterations=4, seed=3)
check(
    any(
        "other parameters" in g
        for g in als.fit_state_problems(inter, other, xv, yv, sv)
    ),
    "fit_state_problems() catches other parameters",
)
check(
    any(
        "other train data" in g
        for g in als.fit_state_problems(inter_changed, pv, xv, yv, sv)
    ),
    "fit_state_problems() catches other train data",
)
xs, ys, ss = paths("verify_swap")
als.fit(inter, other, xs, ys, ss, workers=1, log=quiet)
check(
    any(
        "committed factor file" in g
        for g in als.fit_state_problems(inter, pv, xv, ys, sv)
    ),
    "fit_state_problems() catches a factor file from another fit",
)
xz, yz, sz = paths("verify_zero")
xz.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(xv, xz)
Yz = np.load(yv)
Yz[7] = 1.0  # item 7 has no train data
np.save(yz, Yz)
state = json.loads(sv.read_text())
state["x_sha"], state["y_sha"] = als._file_sha(xz), als._file_sha(yz)
sz.write_text(json.dumps(state))
check(
    any(
        "without train data" in g for g in als.fit_state_problems(inter, pv, xz, yz, sz)
    ),
    "fit_state_problems() catches a nonzero row without train data, even "
    "when the state's hashes match the files",
)
check(
    any(
        "does not exist" in g
        for g in als.fit_state_problems(inter, pv, xv, yv, tmp / "no_state.json")
    ),
    "fit_state_problems() catches a missing fit state",
)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL ALS CHECKS PASSED")
