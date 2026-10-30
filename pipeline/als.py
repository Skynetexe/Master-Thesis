"""Implicit-feedback ALS (Hu, Koren & Volinsky 2008) with exact row
solves, an exact objective check, and a resume state bound to the data.

als_tune.py (hyperparameter selection) and stage_p6.py (the final fit)
both use this module. It replaces the solver that lived in stage_p6.py.

Changes from the P6 solver:

1. The item-major CSR is the exact transpose of the user-major CSR.
   P6's _write_orientation built the item-major arrays one train file at
   a time, sorted each file by item, and concatenated the files. The
   files split users, not items, so the result was 16 item-sorted runs
   under an indptr that assumes one global item order. Every item row
   then held other items' listeners: in a rebuild of that logic, 0.7
   percent of item-major entries were real pairs. P6's item half-step
   therefore fit each item to the wrong users, which is why ALS lost to
   popularity. On the real data the objective check below caught it at
   iteration 2 (a relative rise of 7.66e-3). transpose_user_major()
   builds the correct orientation, and fit() refuses any pair of
   orientations that orientation_real_fraction() does not score 1.0.
2. Resume is bound to the data. P6 resumed a finished fit whenever
   als_params.json matched, even after it rebuilt the CSR from new train
   files. All three P6 runs on record rebuilt the CSR and then ran zero
   iterations. Here the resume key holds the
   parameters, SOLVER_VERSION, ACCUM_DTYPE and a SHA-256 of the CSR.
3. Empty rows solve to zero. A row without data has b = 0, so its exact
   minimizer is 0. P6 skipped those rows and kept their random start
   values, which then scored items that have no train data.
4. The exact objective is computed after every half-step, from terms the
   row solves already produce. An exact block solve on consistent data
   cannot raise the objective, so a relative rise above RISE_TOLERANCE
   stops the fit.
5. The Gram matrix is accumulated in float64. It is computed once per
   half-step, so this is cheap. The per-row sums keep P6's float32
   (ACCUM_DTYPE). A synthetic run with rows of up to 197K entries (the
   size of the most popular real items) gave the same objective path in
   float32 and float64, and float64 costs 2.3 to 2.6 times as much on
   heavy rows.
6. Interruption is safe. Each iteration works on copies. The committed
   factor files change only by an atomic rename after a finished
   iteration, and their SHA-256 is in the state. A stopped run resumes
   from the last finished iteration, so results stay bit-reproducible.
7. Ranking uses all cores. fit() holds BLAS to one thread
   per process, and the per-iteration callback ran inside that limit, so
   the val ranking in als_tune.py used one core. A benchmark with 6.6M
   items measured 49 ms per user on one core, which is 16 minutes per
   iteration for 20,000 users. rank_items() now lifts the limit for its
   matmul only. The scores are the same bits for any thread count.
8. P6 support. Interactions.from_user_major() gives P6 the
   same transpose cache that als_tune.py built. fit_state_problems()
   lets P6's verify() prove that its factors are a finished fit of its
   parameters on its data. Neither changes a result, so SOLVER_VERSION
   and TRANSPOSE_VERSION stay, and existing caches and fits stay valid.

Objective (Hu et al. 2008). p = 1 on observed pairs and 0 elsewhere.
c = 1 + alpha * log1p(listens) on observed pairs and 1 elsewhere.

  J = sum over observed pairs of (c (1 - s)^2 - s^2)
      + sum over all pairs of s^2 + reg (|X|^2 + |Y|^2),   s = x_u . y_i

The second sum equals sum_u x_u^T (Y^T Y) x_u, so J needs no pass over
all (user, item) pairs.
"""

import hashlib
import json
import multiprocessing as mp
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # same fallback as stage_p6.py
    import contextlib

    @contextlib.contextmanager
    def threadpool_limits(limits=1):
        yield


SOLVER_VERSION = "exact-rows-v3"
TRANSPOSE_VERSION = "bucket-transpose-v1"
ACCUM_DTYPE = np.float32
RISE_TOLERANCE = 1e-4
DATA_SCALE = 1.0 / 1000.0  # P6 stores round(1000 * log1p(listens)) as uint16
BLOCK = 2048


class ObjectiveRose(RuntimeError):
    pass


PROGRESS_SECONDS = 30.0  # at most one progress line per task in this time


def _clock(seconds):
    seconds = int(round(max(seconds, 0.0)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    return f"{m}m{s:02d}s" if m else f"{s}s"


class Progress:
    """Progress lines for a long loop: percent done, counts, elapsed time
    and an estimate of the time left from the rate so far. It prints at
    most one line per PROGRESS_SECONDS, plus one closing line. It only
    reads counters, so it cannot change any result."""

    def __init__(self, label, total, log, unit="", scale=1):
        self.label, self.log, self.unit, self.scale = label, log, unit, scale
        self.total = max(int(total), 1)
        self.done = 0
        self.t0 = self.last = time.time()

    def _count(self, n):
        return f"{n:,}" if self.scale == 1 else f"{n / self.scale:,.1f}"

    def advance(self, n):
        self.done += int(n)
        now = time.time()
        if now - self.last >= PROGRESS_SECONDS and self.done < self.total:
            self.last = now
            frac = self.done / self.total
            elapsed = now - self.t0
            left = elapsed * (1.0 - frac) / frac if frac > 0 else 0.0
            self.log(
                f"{self.label}: {frac:.0%} ({self._count(self.done)} of "
                f"{self._count(self.total)}{self.unit}), "
                f"{_clock(elapsed)} elapsed, about {_clock(left)} left"
            )

    def finish(self, extra=""):
        self.log(f"{self.label}: done in {_clock(time.time() - self.t0)}" f"{extra}")


@dataclass(frozen=True)
class ALSParams:
    factors: int
    reg: float
    alpha: float
    iterations: int
    seed: int


class Interactions:
    """The train pairs in both orientations, in P6's CSR layout.
    indptr int64, indices int32, data uint16 = round(1000 * log1p(listens)).
    Row 0 is the phantom slot in both orientations."""

    def __init__(
        self, u_indptr, u_indices, u_data, i_indptr, i_indices, i_data, fingerprint=None
    ):
        self.u = (np.asarray(u_indptr, dtype=np.int64), u_indices, u_data)
        self.i = (np.asarray(i_indptr, dtype=np.int64), i_indices, i_data)
        self.n_users = len(self.u[0]) - 1
        self.n_items = len(self.i[0]) - 1
        self.nnz = int(self.u[0][-1])
        if int(self.i[0][-1]) != self.nnz:
            raise ValueError("the two CSR orientations disagree on nnz")
        self._fingerprint = fingerprint

    @classmethod
    def from_p6(cls, model_dir):
        """Both orientations exactly as P6 wrote them. P6's item-major
        arrays are wrong (see change 1 above), so use this only to
        measure them. with_transposed_item_major() is the usable form."""
        d = Path(model_dir)

        def mm(name):
            return np.load(d / name, mmap_mode="r")

        return cls(
            np.load(d / "csr_user_indptr.npy"),
            mm("csr_user_indices.npy"),
            mm("csr_user_data.npy"),
            np.load(d / "csr_item_indptr.npy"),
            mm("csr_item_indices.npy"),
            mm("csr_item_data.npy"),
        )

    @classmethod
    def from_user_major(cls, u_arrays, n_items, cache_dir, log=None):
        """The user-major arrays with their exact transpose, read from the
        cache in cache_dir. The cache is built first if it is missing or
        belongs to other data. P6 and als_tune.py share this cache."""
        u = (np.asarray(u_arrays[0], dtype=np.int64), u_arrays[1], u_arrays[2])
        fp = user_major_fingerprint(u, log)
        i_arrays = load_or_build_item_major(u, n_items, cache_dir, fp, log)
        return cls(*u, *i_arrays, fingerprint=fp)

    def with_transposed_item_major(self, cache_dir, log=None):
        """The same user-major arrays, with an item-major orientation built
        as their exact transpose (cached in cache_dir, keyed by the
        user-major SHA-256)."""
        i_arrays = load_or_build_item_major(
            self.u, self.n_items, cache_dir, self.fingerprint(), log
        )
        return Interactions(*self.u, *i_arrays, fingerprint=self.fingerprint())

    def fingerprint(self, log=None):
        """SHA-256 over all three user-major arrays. fit() only accepts an
        item-major orientation that is the transpose of these arrays, so
        they identify the data."""
        if self._fingerprint is None:
            self._fingerprint = user_major_fingerprint(self.u, log)
        return self._fingerprint

    def train_items(self, user):
        indptr, indices, _ = self.u
        return np.asarray(indices[int(indptr[user]) : int(indptr[user + 1])])


def user_major_fingerprint(u_arrays, log=None):
    """SHA-256 over the user-major arrays (indptr as int64, indices, data),
    fed in blocks of 2^26 elements. It keys the transpose cache and every
    fit state."""
    total = sum(len(a) * np.dtype(a.dtype).itemsize for a in u_arrays)
    prog = Progress(
        "als: fingerprint of the user-major CSR",
        total,
        log or (lambda msg: None),
        unit=" GiB",
        scale=2**30,
    )
    h = hashlib.sha256()
    step = 1 << 26
    for arr in u_arrays:
        for lo in range(0, len(arr), step):
            blk = np.ascontiguousarray(arr[lo : lo + step])
            h.update(blk.tobytes())
            prog.advance(blk.nbytes)
    prog.finish()
    return h.hexdigest()


# Orientation


def orientation_real_fraction(inter, rows=200, per_row=200, seed=0):
    """Fraction of sampled item-major entries that are real pairs of the
    user-major CSR, with the same data value. The exact transpose scores
    1.0. A rebuild of P6's per-file construction scored 0.007."""
    i_indptr, i_indices, i_data = inter.i
    u_indptr, u_indices, u_data = inter.u
    rng = np.random.default_rng(seed)
    nonempty = np.flatnonzero(np.diff(i_indptr) > 0)
    if len(nonempty) == 0:
        return 1.0
    ok = n = 0
    for it in rng.choice(nonempty, size=min(rows, len(nonempty)), replace=False):
        s, e = int(i_indptr[it]), int(i_indptr[it + 1])
        pos = (
            np.arange(s, e)
            if e - s <= per_row
            else np.sort(rng.choice(np.arange(s, e), per_row, replace=False))
        )
        for p in pos:
            n += 1
            usr = int(i_indices[p])
            if not 0 <= usr < inter.n_users:
                continue
            us, ue = int(u_indptr[usr]), int(u_indptr[usr + 1])
            k = int(np.searchsorted(u_indices[us:ue], it))
            ok += int(
                k < ue - us
                and int(u_indices[us + k]) == it
                and int(u_data[us + k]) == int(i_data[p])
            )
    return ok / n


def transpose_user_major(
    u_arrays, n_items, out_dir, bucket_entries=400_000_000, chunk=1 << 26, log=None
):
    """Write the item-major CSR as the exact transpose of the user-major
    CSR: csr_item_indptr.npy, csr_item_indices.npy, csr_item_data.npy in
    out_dir. Within each item row, users come out in ascending order.

    Out of core: items are split into buckets of at most bucket_entries
    entries (a single larger item gets a bucket of its own). Each bucket
    reads the user-major arrays once, sorts its own entries by item
    (stably, so user order holds) and writes one contiguous output range.
    Peak memory is about 18 bytes per bucket entry."""
    log = log or (lambda msg: None)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    u_indptr, u_indices, u_data = u_arrays
    u_indptr = np.asarray(u_indptr, dtype=np.int64)
    nnz = int(u_indptr[-1])
    counts = np.zeros(n_items, dtype=np.int64)
    prog = Progress("als: transpose, counting item degrees", nnz, log, unit=" entries")
    for lo in range(0, nnz, chunk):
        c = np.bincount(np.asarray(u_indices[lo : lo + chunk]), minlength=n_items)
        if len(c) != n_items:
            raise ValueError("a user-major index exceeds the item slots")
        counts += c
        prog.advance(min(chunk, nnz - lo))
    prog.finish()
    i_indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    edges = [0]
    while edges[-1] < n_items:
        start = edges[-1]
        end = (
            int(
                np.searchsorted(
                    i_indptr, i_indptr[start] + bucket_entries, side="right"
                )
            )
            - 1
        )
        edges.append(min(max(end, start + 1), n_items))

    names = ("csr_item_indptr.npy", "csr_item_indices.npy", "csr_item_data.npy")
    tmp = {n: out / (n[:-4] + ".tmp.npy") for n in names}
    idx_out = np.lib.format.open_memmap(
        tmp[names[1]], mode="w+", dtype=np.int32, shape=(nnz,)
    )
    dat_out = np.lib.format.open_memmap(
        tmp[names[2]], mode="w+", dtype=np.uint16, shape=(nnz,)
    )
    n_buckets = len(edges) - 1
    for b in range(n_buckets):
        lo_it, hi_it = edges[b], edges[b + 1]
        s, e = int(i_indptr[lo_it]), int(i_indptr[hi_it])
        if e == s:
            continue
        prog = Progress(
            f"als: transpose bucket {b + 1}/{n_buckets}, scan",
            nnz,
            log,
            unit=" entries",
        )
        users, items, data = [], [], []
        for lo in range(0, nnz, chunk):
            hi = min(lo + chunk, nnz)
            it = np.asarray(u_indices[lo:hi])
            m = (it >= lo_it) & (it < hi_it)
            prog.advance(hi - lo)
            if not m.any():
                continue
            pos = np.flatnonzero(m).astype(np.int64) + lo
            users.append(
                (np.searchsorted(u_indptr, pos, side="right") - 1).astype(np.int32)
            )
            items.append(it[m])
            data.append(np.asarray(u_data[lo:hi])[m])
        users = np.concatenate(users)
        items = np.concatenate(items)
        data = np.concatenate(data)
        if len(users) != e - s:
            raise RuntimeError(
                f"transpose bucket {b}: found {len(users)} "
                f"entries, the counts say {e - s}"
            )
        order = np.argsort(items, kind="stable")
        idx_out[s:e] = users[order]
        dat_out[s:e] = data[order]
        prog.finish(f", {e - s:,} entries written")
    idx_out.flush()
    dat_out.flush()
    del idx_out, dat_out
    np.save(tmp[names[0]], i_indptr)
    for n in names:
        os.replace(tmp[n], out / n)


def load_or_build_item_major(u_arrays, n_items, cache_dir, fingerprint, log=None):
    """The transposed item-major arrays from cache_dir, built first if the
    cache is missing or belongs to other data."""
    log = log or (lambda msg: None)
    cache = Path(cache_dir)
    meta_path = cache / "meta.json"
    want = {
        "version": TRANSPOSE_VERSION,
        "user_major_sha256": fingerprint,
        "n_items": int(n_items),
    }
    names = ("csr_item_indptr.npy", "csr_item_indices.npy", "csr_item_data.npy")
    have = (
        json.loads(meta_path.read_text(encoding="utf-8"))
        if meta_path.exists()
        else None
    )
    if have != want or not all((cache / n).exists() for n in names):
        cache.mkdir(parents=True, exist_ok=True)
        meta_path.unlink(missing_ok=True)
        need = int(u_arrays[0][-1]) * 6 * 1.05
        free = shutil.disk_usage(cache).free
        if free < need:
            raise OSError(
                f"the transpose needs {need / 2**30:.1f} GiB in "
                f"{cache}, {free / 2**30:.1f} GiB is free"
            )
        log(
            "als: building the item-major CSR as the transpose of the " "user-major CSR"
        )
        transpose_user_major(u_arrays, n_items, cache, log=log)
        _write_json(meta_path, want)
    else:
        log("als: reusing the transposed item-major CSR")
    return (
        np.load(cache / names[0]),
        np.load(cache / names[1], mmap_mode="r"),
        np.load(cache / names[2], mmap_mode="r"),
    )


# Solver
_CTX = {}


def _solve_rows(lo, hi):
    indptr, indices, data = _CTX["rows"]
    target, fixed = _CTX["target"], _CTX["fixed"]
    gram_reg, alpha = _CTX["gram_reg"], _CTX["alpha"]
    scale = ACCUM_DTYPE(DATA_SCALE)
    observed = 0.0  # sum of c (1 - s)^2 - s^2 over this block's pairs
    for r in range(lo, hi):
        s, e = int(indptr[r]), int(indptr[r + 1])
        if e <= s:
            target[r] = 0.0
            continue
        cols = np.asarray(indices[s:e])
        c = 1.0 + alpha * (np.asarray(data[s:e], dtype=ACCUM_DTYPE) * scale)
        F = np.asarray(fixed[cols], dtype=ACCUM_DTYPE)
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


def _gram(M):
    """M^T M accumulated in float64, in a fixed chunk order."""
    G = np.zeros((M.shape[1], M.shape[1]), dtype=np.float64)
    step = 1 << 20
    for lo in range(0, M.shape[0], step):
        blk = np.asarray(M[lo : lo + step], dtype=np.float64)
        G += blk.T @ blk
    return G


def _sum_quadratic(M, G):
    """sum_r m_r^T G m_r in float64, chunked."""
    total = 0.0
    step = 1 << 20
    for lo in range(0, M.shape[0], step):
        blk = np.asarray(M[lo : lo + step], dtype=np.float64)
        total += float(np.sum((blk @ G) * blk))
    return total


def _sum_squares(M):
    total = 0.0
    step = 1 << 20
    for lo in range(0, M.shape[0], step):
        blk = np.asarray(M[lo : lo + step], dtype=np.float64)
        total += float(np.sum(blk * blk))
    return total


def _half(rows, target, fixed, reg, alpha, workers, label="als: half-step", log=None):
    """Solve every row of target exactly with fixed held constant. Returns
    the exact objective J at the new (target, fixed). Progress counts
    train entries, because a row's cost grows with its entries."""
    G = _gram(fixed)
    _CTX.update(
        rows=rows,
        target=target,
        fixed=fixed,
        alpha=float(alpha),
        gram_reg=G + reg * np.eye(G.shape[0]),
    )
    indptr = rows[0]
    n = len(indptr) - 1
    tasks = [(lo, min(lo + BLOCK, n)) for lo in range(0, n, BLOCK)]
    prog = Progress(
        label, int(indptr[n] - indptr[0]), log or (lambda msg: None), unit=" entries"
    )
    parts = []
    try:
        if workers <= 1:
            results = map(_solve_block, tasks)
            for (lo, hi), part in zip(tasks, results):
                parts.append(part)
                prog.advance(int(indptr[hi] - indptr[lo]))
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("fork")
            ) as pool:
                results = pool.map(_solve_block, tasks, chunksize=8)
                for (lo, hi), part in zip(tasks, results):
                    parts.append(part)
                    prog.advance(int(indptr[hi] - indptr[lo]))
    finally:
        _CTX.clear()
    prog.finish()
    observed = float(np.sum(parts))  # task order, so the sum is deterministic
    return (
        observed
        + _sum_quadratic(target, G)
        + reg * (_sum_squares(target) + _sum_squares(fixed))
    )


# Fit


def _file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path, doc):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True), encoding="utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _work_path(path):
    return path.with_name(path.stem + ".work.npy")


def fit_key(inter, params):
    """Everything a fit depends on: the parameters, the solver version,
    the accumulation type, the data fingerprint and the shape. fit()
    resumes only on an equal key."""
    return {
        "params": asdict(params),
        "solver": SOLVER_VERSION,
        "accum": np.dtype(ACCUM_DTYPE).name,
        "data": inter.fingerprint(),
        "shape": [inter.n_users, inter.n_items, inter.nnz],
    }


def fit_state_problems(inter, params, x_path, y_path, state_path):
    """The reasons why the committed factor files are not a finished fit of
    these parameters on this data, as sentences. An empty list means they
    are one. P6's verify() uses this, so stale factors cannot pass
    again."""
    x_path, y_path, state_path = Path(x_path), Path(y_path), Path(state_path)
    if not state_path.exists():
        return [f"{state_path} does not exist"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    problems = []
    if state.get("key") != fit_key(inter, params):
        problems.append(
            "the fit state belongs to other parameters, another "
            "solver version or other train data"
        )
    n_done = len(state.get("history", []))
    if n_done != params.iterations:
        problems.append(
            f"the fit state holds {n_done} finished iterations, "
            f"the parameters ask for {params.iterations}"
        )
    for path, name in ((x_path, "x_sha"), (y_path, "y_sha")):
        if not path.exists() or state.get(name) != _file_sha(path):
            problems.append(
                f"{path.name} is not the committed factor file " f"of this fit"
            )
    if problems:
        return problems
    X = np.load(x_path, mmap_mode="r")
    Y = np.load(y_path, mmap_mode="r")
    if X.shape != (inter.n_users, params.factors) or Y.shape != (
        inter.n_items,
        params.factors,
    ):
        return [f"factor shapes {X.shape} and {Y.shape} do not match the data"]
    for M, indptr, what in ((X, inter.u[0], "user"), (Y, inter.i[0], "item")):
        empty = np.flatnonzero(np.diff(indptr) == 0)
        if len(empty) and np.any(np.asarray(M[empty]) != 0):
            problems.append(
                f"a row without train data ({what} side) is not " f"exactly zero"
            )
    return problems


def fit(
    inter, params, x_path, y_path, state_path, workers=None, on_iteration=None, log=None
):
    """Fit ALS, or resume a matching fit. Returns (X, Y, history) with X and
    Y as read-only memmaps of the committed factor files.

    on_iteration(iteration, X, Y, history) runs after each committed
    iteration. history already ends with that iteration's record. The
    callback may return a dict. Its keys go into that record, and
    {"stop": True} ends the fit there. If a run stops inside the
    callback, the next call runs the callback again for that iteration
    before it fits anything new."""
    log = log or (lambda msg: print(msg, flush=True))
    real = orientation_real_fraction(inter, rows=100, per_row=100)
    if real < 1.0:
        raise ValueError(
            f"the item-major CSR is not the transpose of the user-major CSR: "
            f"only {real:.1%} of sampled item-major entries are real pairs. "
            f"Build it with Interactions.with_transposed_item_major()."
        )
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
        and state.get("x_sha") == _file_sha(x_path)
        and state.get("y_sha") == _file_sha(y_path)
    )
    if resumable:
        history = state["history"]
        if history:
            log(f"als: resuming after iteration {len(history)}")
    else:
        if state is not None:
            log(
                "als: the saved fit does not match these parameters, this "
                "solver or this data. Starting fresh."
            )
        rng = np.random.default_rng(params.seed)
        for path, rows in ((x_path, inter.n_users), (y_path, inter.n_items)):
            work = _work_path(path)
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
        _write_json(
            state_path,
            {
                "key": key,
                "history": history,
                "x_sha": _file_sha(x_path),
                "y_sha": _file_sha(y_path),
            },
        )

    def save(x_sha, y_sha):
        _write_json(
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

    # A callback that was interrupted for the last committed iteration
    # runs again before any new iteration starts.
    if history and not history[-1].get("callback_done"):
        run_callback(len(history))
        save(_file_sha(x_path), _file_sha(y_path))

    prev = history[-1]["objective_after_items"] if history else None
    with threadpool_limits(1):
        for it in range(len(history), params.iterations):
            if history and history[-1].get("stop"):
                break
            t0 = time.time()
            step = f"als: iteration {it + 1}/{params.iterations}"
            log(f"{step} started")
            xw, yw = _work_path(x_path), _work_path(y_path)
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
                            f"consistent data cannot do that. The two CSR "
                            f"orientations passed the transpose check, so "
                            f"suspect precision next: set ACCUM_DTYPE = "
                            f"np.float64 in pipeline/als.py and rerun."
                        )
                prev = j
            x_sha, y_sha = _file_sha(xw), _file_sha(yw)
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


# Ranking


def rank_items(
    X,
    Y,
    inter,
    users,
    picks,
    chunk=64,
    log=None,
    label="als: ranking picks",
    threads=None,
):
    """Rank of each pick among its user's candidates, under the P5 harness
    rules: every item slot except slot 0 and the user's train items. Ties
    rank by item_id ascending. Same arithmetic as StageP6._rank.

    The score matmul runs with `threads` BLAS threads (default: all
    cores). fit() holds BLAS to one thread for the whole iteration loop,
    and its callback runs inside that limit, so without this the ranking
    ran on one core. OpenBLAS does not split the 64-long dot products
    across threads, so the scores are the same bits for any thread count
    (test_als.py checks this on the machine that runs it).

    users: (n,) user ids. picks: (n, k) item ids. Returns (n, k) int64."""
    threads = int(threads or os.cpu_count() or 1)
    Yd = np.asarray(Y, dtype=np.float32)
    users = np.asarray(users, dtype=np.int64)
    picks = np.asarray(picks, dtype=np.int64)
    out = np.empty(picks.shape, dtype=np.int64)
    prog = Progress(label, len(users), log or (lambda msg: None), unit=" users")
    for c0 in range(0, len(users), chunk):
        uc = users[c0 : c0 + chunk]
        with threadpool_limits(threads):
            S = np.asarray(X[uc], dtype=np.float32) @ Yd.T
        S[:, 0] = -np.inf
        for k in range(len(uc)):
            row = S[k]
            row[inter.train_items(int(uc[k]))] = -np.inf
            for j in range(picks.shape[1]):
                t = int(picks[c0 + k, j])
                st = row[t]
                if not np.isfinite(st):
                    out[c0 + k, j] = row.shape[0]
                else:
                    out[c0 + k, j] = (
                        1
                        + int(np.count_nonzero(row > st))
                        + int(np.count_nonzero(row[:t] == st))
                    )
        prog.advance(len(uc))
    prog.finish()
    return out
