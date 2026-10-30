"""The cold-start curve: the GAM, the cold-track GAM, ItemKNN and
popularity when each eval listener's history is cut to its first 5, 10,
25 and 100 plays.

RULES, declared in the docstring of pipeline/p14.py before P14 read any
eval listener. Nothing here changes after a result.
- listeners: the eval listeners with at least 100 training pairs, the same
  set at every point.
- history: the listener's first N training pairs in time order (first_ts,
  ties by item id), N in 5, 10, 25 and 100. Each kept pair counts as one
  play (weight round(1000 log1p(1)) / 1000), because the pair table cannot
  tell which later plays fell before the cut.
- candidates: P5's candidates of the full history (the whole training set
  excluded), so every point ranks the same candidates and only the
  evidence changes. Picks: the same test picks.
- arms: the GAM, the cold-track GAM, ItemKNN both lists and popularity
  (flat), with the frozen tables and lists. ALS needs a fold-in and stays
  out.
- metrics: NDCG@100 and R@100 per N with 95 percent listener intervals,
  and P14's full-history value as the last point. Descriptive, no test.

How the computation reads the rules (written before any run on real
data):
- "the eval listeners" are P14's listeners (P5's frame: the eval listeners
  with both test picks), read from p14.npz with their picks and their P14
  ranks, so the last point is P14's own value on the same listeners.
- a listener's metric is the mean over its two picks, as in P14. The
  interval is the mean over the listeners plus or minus 1.96 standard
  errors.
- popularity is flat by construction: its scores ignore the history, and
  the candidates stay those of the full history. Its value at every N is
  P14's popularity value on these listeners.

COMPUTATION.
- the first pairs: one DuckDB query per train file gives each listener's
  first FIRST_KEEP pairs by (first_ts, item_id) and its pair count, saved
  in SCRATCH_DIR/cold_start_curve and keyed by the train manifest and the
  listeners.
- the GAM arms: the final model's and the cold-track GAM's tables of
  p14.load_models(), scored by p14.term_sums() on the concepts of each slot
  range (CatalogScorer.concept_range), so equal term values give equal
  scores and equal scores rank by item id (P14's change
  note). Ranks are exact up to RANK_CUT = 100 and RANK_CUT + 1
  beyond, as in P13's selection pass: the ranges that hold a pick go
  first, and a listener stops when every pick of both tables has RANK_CUT
  or more better candidates. NDCG@100 and R@100 read no rank beyond 100.
- ItemKNN: itemknn.user_scores() on the first N pairs with the item side's
  transposed lists (the matrices of itemknn.transposed()), the full history
  excluded, P5's tie rule (itemknn.pick_ranks).
- one pass over the listeners in blocks of BLOCK, saved as they finish.

Checks, each one stops the run:
- the listeners' pair counts equal the CSR rows, every kept pair is a
  train item of its listener, and the kept pairs run in (first_ts, item)
  order.
- in every slot range scored, the scores of every CHECK_STRIDE-th slot and
  of the picks equal catalog.reference_scores() on those slots exactly.
- the pool gives the probe listeners the results of this process.
- VERIFY_USERS listeners at every N: the ranks of both tables equal the
  dense reference over the full catalog (concepts.full_catalog,
  catalog.reference_scores, the P5 rule) cut at RANK_CUT, and ItemKNN's
  ranks equal a full sort and lie in the range of a second route
  (itemknn.brute_scores).
- the same listeners with their full history (the CSR weights): the pass's
  ranks equal P14's stored ranks of both tables cut at RANK_CUT, and
  ItemKNN's equal P14's ItemKNN ranks exactly.
- the read-back: every result follows from the stored ranks, and the
  stored P14 ranks equal p14.npz.

Outputs: MODEL_DIR/cold_start_curve.json (the rules, the results, the
checks), MODEL_DIR/cold_start_curve.npz (per listener: each arm's ranks at
each N, P14's ranks, the first pairs) and
MODEL_DIR/reports/cold_start_curve.md. Every line also goes to
MODEL_DIR/logs/cold_start_curve.log. Long steps print a line at most every
30 seconds with the share done and the time left.

Resume: after Ctrl+C the same command continues: the first pairs and the
pass blocks stay.

Expected run on the real data: the DuckDB query a few minutes (16 train
files), the pass about 1 to 3 hours (four cut points per listener, most
listeners stop after a few ranges, the probe prints an estimate), the
checks a few minutes.

Usage, from the repository root, after the stage p14_eval:
    python3 -m pipeline.runner --only cold_start_curve
The pass uses P13's worker setting. To use fewer workers:
    P13A_WORKERS=8 python3 -m pipeline.runner --only cold_start_curve
"""

import glob as globlib
import math
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import gam_knn_diag as gkd
from pipeline import gam_refit as gr
from pipeline import itemknn as knn
from pipeline import p14
from pipeline import p14_cold as cold
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import Progress, _clock
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.script_stage import ScriptStage, log_file

VERSION = "cold-curve-v1"
WORK_NAME = "cold_start_curve"
DECLARED = "before P14 read any eval listener"
OUT_FILES = ("cold_start_curve.json", "cold_start_curve.npz")
REPORT_NAME = "cold_start_curve.md"
LOG_NAME = "cold_start_curve.log"
COMMAND = "python3 -m pipeline.runner --only cold_start_curve"
POINTS = (5, 10, 25, 100)
MIN_PAIRS = 100
RANK_CUT = 100
CHUNK = 1 << 16
BLOCK = 25
PROBE_USERS = 4
CHECK_STRIDE = 64
VERIFY_USERS = 6
VERIFY_SEED = 2026100501
W_ONE = round(1000.0 * math.log1p(1.0)) / 1000.0
TABLE_ROWS = (0, 2)  # the GAM and the cold-track GAM in load_models
CURVE_ARMS = ("gam", "cold_gam", "knn_both")
ARMS = CURVE_ARMS + ("popularity",)
ARM_LABELS = {
    "gam": "GAM (final model)",
    "cold_gam": "cold-track GAM",
    "knn_both": "ItemKNN, both lists",
    "popularity": "popularity",
}
METRICS = ("ndcg_at_100", "recall_at_100")
Z95 = p14.Z95


def protocol():
    return {
        "version": VERSION,
        "declared": DECLARED,
        "listeners": f"P14's eval listeners with at least {MIN_PAIRS} "
        f"training pairs",
        "history": "the first N training pairs by (first_ts, item_id), one "
        "play each",
        "points": [int(N) for N in POINTS],
        "candidates": "P5's candidates of the full history",
        "arms": {a: ARM_LABELS[a] for a in ARMS},
        "metrics": "NDCG@100 and R@100, listener means over the two test "
        "picks, 95 percent normal intervals, descriptive",
    }


def keep_n():
    return int(max(POINTS))


class Inputs:
    """The DuckDB part, which reads the real data. The smoke test
    replaces this class."""

    def __init__(self, model_dir):
        self.d = Path(model_dir)

    def key(self):
        return {"train_manifest.json": file_sha256(self.d / "train_manifest.json")}

    def first_pairs(self, users, keep, say):
        """(user, item, first_ts) rows of each listener's first keep train
        pairs by (first_ts, item_id) in each train file, and {user: pair
        count} summed over the files."""
        import pandas as pd

        files = sorted(globlib.glob(str(self.d / "interactions_train_*.parquet")))
        if not files:
            raise ValueError("no interactions_train_*.parquet files")
        rows_u, rows_i, rows_t = [], [], []
        counts = {}
        prog = Progress(
            f"{WORK_NAME}: the first {keep} pairs per listener",
            len(files),
            say,
            unit=" files",
        )
        con = connect(8)
        try:
            con.register(
                "cc_users", pd.DataFrame({"user_id": np.asarray(users, dtype=np.int64)})
            )
            for path in files:
                z = con.execute(f"""
                    SELECT user_id, item_id, first_ts, n FROM (
                        SELECT t.user_id, t.item_id, t.first_ts,
                               row_number() OVER (
                                   PARTITION BY t.user_id
                                   ORDER BY t.first_ts, t.item_id) AS rn,
                               count(*) OVER (PARTITION BY t.user_id) AS n
                        FROM read_parquet('{path}') t
                        WHERE t.user_id IN (SELECT user_id FROM cc_users))
                    WHERE rn <= {int(keep)}
                    ORDER BY user_id, first_ts, item_id""").fetchnumpy()
                if any(np.ma.is_masked(z[k]) for k in z):
                    raise ValueError(
                        f"{path}: a train pair has no user, " f"item or first_ts"
                    )
                u = np.asarray(z["user_id"], dtype=np.int64)
                n = np.asarray(z["n"], dtype=np.int64)
                rows_u.append(u)
                rows_i.append(np.asarray(z["item_id"], dtype=np.int64))
                rows_t.append(np.asarray(z["first_ts"], dtype=np.int64))
                first = (
                    np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
                    if len(u)
                    else np.zeros(0, dtype=np.int64)
                )
                for k in first:
                    counts[int(u[k])] = counts.get(int(u[k]), 0) + int(n[k])
                prog.advance(1)
        finally:
            con.close()
        prog.finish()
        return (
            (np.concatenate(rows_u), np.concatenate(rows_i), np.concatenate(rows_t)),
            counts,
        )


def first_arrays(rows, counts, users, keep):
    """(indptr over users, items, first_ts, pair counts): each listener's
    first keep pairs by (first_ts, item_id) over the rows of every file."""
    u, i, t = (np.asarray(x, dtype=np.int64) for x in rows)
    order = np.lexsort((i, t, u))
    u, i, t = u[order], i[order], t[order]
    users = np.asarray(users, dtype=np.int64)
    lo = np.searchsorted(u, users, side="left")
    hi = np.searchsorted(u, users, side="right")
    take = np.minimum(hi - lo, keep)
    indptr = np.concatenate(([0], np.cumsum(take))).astype(np.int64)
    idx = (
        np.concatenate([np.arange(a, a + n) for a, n in zip(lo, take)])
        if len(users)
        else np.zeros(0, dtype=np.int64)
    )
    cnt = np.array([int(counts.get(int(x), 0)) for x in users], dtype=np.int64)
    return indptr, i[idx], t[idx], cnt


def check_first(users, indptr, items, ts, counts, u_indptr, u_indices, keep):
    """The checks of the first pairs (module docstring)."""
    own = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
        u_indptr[users], dtype=np.int64
    )
    if not np.array_equal(own, counts):
        k = int(np.flatnonzero(own != counts)[0])
        raise ValueError(
            f"listener {int(users[k])}: {int(counts[k])} train "
            f"pairs in the files, {int(own[k])} in the CSR"
        )
    if not np.array_equal(np.diff(indptr), np.minimum(own, keep)):
        raise ValueError(
            "a listener has fewer first pairs than its train " "pairs allow"
        )
    for k, u in enumerate(users):
        a, b = int(indptr[k]), int(indptr[k + 1])
        row = np.asarray(
            u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])], dtype=np.int64
        )
        if not np.all(np.isin(items[a:b], row)):
            raise ValueError(f"listener {int(u)}: a first pair is not a " f"train item")
        if len(np.unique(items[a:b])) != b - a:
            raise ValueError(f"listener {int(u)}: a first pair repeats")
        tt, ii = ts[a:b], items[a:b]
        if np.any((tt[1:] < tt[:-1]) | ((tt[1:] == tt[:-1]) & (ii[1:] <= ii[:-1]))):
            raise ValueError(
                f"listener {int(u)}: the first pairs are not in "
                f"(first_ts, item_id) order"
            )


# The pass over the listeners
_E = {}  # the pass context, set before a fork


def _range(c, state, pre, a, b):
    """(scores (C, b - a), concepts) of slots a..b-1: p14.term_sums on the
    concepts of the range."""
    scorer, space = c["scorer"], c["space"]
    conc = {}
    for t in scorer.terms:
        if t.source not in conc:
            conc[t.source] = scorer.concept_range(state, space, pre, t.source, a, b)
    return p14.term_sums(scorer.terms, scorer.T, conc, scorer.tier[a:b]), conc


def _reference_check(c, S, conc, sub, a, u):
    """Raise RuntimeError unless S[:, sub] equals
    catalog.reference_scores() on those slots."""
    scorer = c["scorer"]
    full = {src: (v[sub], h[sub]) for src, (v, h) in conc.items()}
    ref = ct.reference_scores(full, scorer.terms, scorer.T, 0.0, scorer.tier[a + sub])
    if not np.array_equal(S[:, sub], ref):
        gap = float(np.max(np.abs(S[:, sub] - ref)))
        raise RuntimeError(
            f"listener {u}: the pass's scores differ from "
            f"catalog.reference_scores near slot {a} (largest "
            f"gap {gap:.1e})"
        )


def cut_ranks(c, state, picks, excl, u, cut=RANK_CUT):
    """((C, picks) ranks exact up to cut and cut + 1 beyond, slots scored,
    slots checked) under the P5 rule, with excl (sorted) and slot 0 out."""
    scorer, space = c["scorer"], c["space"]
    n, chunk = scorer.n, int(c["chunk"])
    pre = scorer.prepare(state, space)
    picks = np.asarray(picks, dtype=np.int64)
    first = list(dict.fromkeys(int(t) // chunk for t in picks))
    held = {r: _range(c, state, pre, r * chunk, min((r + 1) * chunk, n)) for r in first}
    s_pick = np.stack(
        [held[int(t) // chunk][0][:, int(t) % chunk] for t in picks]
    )  # (picks, C)
    count = np.zeros(s_pick.shape, dtype=np.int64)
    scored = checked = 0
    in_first = set(first)
    order = first + [r for r in range(-(-n // chunk)) if r not in in_first]
    for r in order:
        a, b = r * chunk, min((r + 1) * chunk, n)
        S, conc = held.pop(r) if r in held else _range(c, state, pre, a, b)
        sub = np.unique(
            np.concatenate(
                (
                    np.arange(0, b - a, int(c["stride"])),
                    picks[(picks >= a) & (picks < b)] - a,
                )
            )
        )
        _reference_check(c, S, conc, sub, a, u)
        checked += len(sub)
        lo, hi = np.searchsorted(excl, (a, b))
        S[:, excl[lo:hi] - a] = -np.inf
        if a == 0:
            S[:, 0] = -np.inf
        for p, t in enumerate(picks):
            sp_ = s_pick[p][:, None]
            count[p] += np.count_nonzero(S > sp_, axis=1)
            e = min(max(int(t) - a, 0), b - a)
            if e:
                count[p] += np.count_nonzero(S[:, :e] == sp_, axis=1)
        scored += b - a
        if np.all(1 + count > cut):
            break
    return np.minimum(1 + count, cut + 1).T, scored, checked


def _curve_one(c, k):
    """Every rank of listener k at every cut point (module docstring)."""
    scorer, space = c["scorer"], c["space"]
    u = int(c["users"][k])
    picks = np.asarray(c["picks"][k], dtype=np.int64)
    s, e = int(c["u_indptr"][u]), int(c["u_indptr"][u + 1])
    excl = np.unique(np.asarray(c["u_indices"][s:e], dtype=np.int64))
    scorer._check_picks(picks, excl)
    fi = c["first"][int(c["first_ptr"][k]) : int(c["first_ptr"][k + 1])]
    C = scorer.C
    ranks = np.empty((len(POINTS), C + 1, len(picks)), dtype=np.int64)
    scored = np.empty(len(POINTS), dtype=np.int64)
    checked = 0
    for j, N in enumerate(POINTS):
        items = np.sort(fi[:N])
        w = np.full(len(items), W_ONE)
        state = cn.build_user_state(items, w, space)
        r, scored[j], ch = cut_ranks(c, state, picks, excl, u)
        ranks[j, :C] = r
        checked += ch
        ks = knn.user_scores(c["t_mats"], items, w)["both"]
        ranks[j, C] = knn.pick_ranks(knn.candidate_scores(ks, excl), picks, "id")
    return {"ranks": ranks, "scored": scored, "checked": np.int64(checked)}


def _curve_task(task):
    lo, hi = task
    with threadpool_limits(1):  # many workers, one BLAS thread each
        outs = [_curve_one(_E, k) for k in range(lo, hi)]
    return {key: np.stack([o[key] for o in outs]) for key in outs[0]}


def curve_pass(
    st,
    scorer,
    space,
    users,
    picks,
    first_ptr,
    first,
    u_arrays,
    work,
    key,
    say,
    workers=None,
):
    """The pass over the listeners: per-listener arrays, saved in blocks
    as they finish, so a stopped run continues with the missing blocks."""
    n_slots, _ = st._id_slots()
    u_indptr, u_indices, _ = u_arrays
    n_blocks = max(1, -(-len(users) // BLOCK))
    tasks = s13._blocks(np.ones(len(users), dtype=np.int64), n_blocks)
    pdir = work / "pass"
    state_path = pdir / "state.json"
    k = {
        "key": s13._sha(key),
        "tables": s13._array_sha(scorer.T),
        "users": p14._array_key(users, picks, first_ptr, first),
        "blocks": [list(t) for t in tasks],
        "settings": [list(POINTS), RANK_CUT, CHUNK, CHECK_STRIDE, W_ONE],
    }
    state = load_json(state_path) or {}
    if state.get("k") != k:
        pdir.mkdir(parents=True, exist_ok=True)
        for f in pdir.glob("part_*.npz"):
            f.unlink()
        state = {"k": k, "parts": {}}
        atomic_json_write(state, state_path)

    def part(i):
        return pdir / f"part_{i:05d}.npz"

    done = {
        int(i)
        for i, sha in state["parts"].items()
        if part(int(i)).exists() and file_sha256(part(int(i))) == sha
    }
    pending = [i for i in range(len(tasks)) if i not in done]
    if done:
        say(f"{WORK_NAME}: pass resume, {len(done)} of {len(tasks)} blocks " f"done")
    if pending:
        _E.update(
            scorer=scorer,
            space=space,
            users=users,
            picks=picks,
            u_indptr=u_indptr,
            u_indices=u_indices,
            first=first,
            first_ptr=first_ptr,
            chunk=CHUNK,
            stride=CHECK_STRIDE,
            t_mats={kind: space.colisten_t[kind] for kind in knn.KINDS},
        )
        try:
            left = np.concatenate([np.arange(*tasks[i]) for i in pending])
            if workers is None:
                with cold.relabel_logs(WORK_NAME, COMMAND):
                    workers = st._selection_workers(
                        space, users[left], u_indptr, u_indices, n_slots
                    )
            take = np.unique(
                np.round(
                    np.linspace(0, len(left) - 1, min(PROBE_USERS, len(left)))
                ).astype(np.int64)
            )
            probe, secs = {}, []
            with threadpool_limits(1):
                for j, i in enumerate(take, start=1):
                    t1 = time.time()
                    probe[int(left[i])] = _curve_one(_E, int(left[i]))
                    secs.append(time.time() - t1)
                    say(
                        f"{WORK_NAME}: probe listener {j} of {len(take)}: "
                        f"{secs[-1]:.2f} s for {len(POINTS)} cut points"
                    )
            par = max(
                1.0,
                min(float(workers), s13b.PASS_SPEEDUP_PER_CORE * s13._physical_cores()),
            )
            say(
                f"{WORK_NAME}: probe mean {np.mean(secs):.2f} s per listener "
                f"in one process. Rough estimate for {len(left):,} listeners "
                f"on {workers} workers, with a parallel speedup of {par:.1f}: "
                f"{_clock(np.mean(secs) * len(left) / par)}. The progress "
                f"lines below measure the real rate"
            )
            prog = Progress(
                f"{WORK_NAME}: pass, {len(POINTS)} cut points, " f"{workers} workers",
                len(left),
                say,
                unit=" listeners",
            )
            with s13._Pool(workers) as pool:
                for j, out in pool.unordered(_curve_task, [tasks[i] for i in pending]):
                    i = pending[j]
                    s13._atomic_savez(part(i), **out)
                    state["parts"][str(i)] = file_sha256(part(i))
                    atomic_json_write(state, state_path)
                    prog.advance(tasks[i][1] - tasks[i][0])
            prog.finish()
            starts = np.array([t[0] for t in tasks])
            for kk, res in probe.items():
                i = int(np.searchsorted(starts, kk, side="right") - 1)
                z = np.load(part(i))
                jj = kk - tasks[i][0]
                if not all(np.array_equal(res[key], z[key][jj]) for key in res):
                    raise ValueError(
                        f"listener {int(users[kk])}: the pool's "
                        f"results differ from the probe's in "
                        f"this process"
                    )
        finally:
            _E.clear()
    zs = [np.load(part(i)) for i in range(len(tasks))]
    out = {key: np.concatenate([z[key] for z in zs]) for key in zs[0].files}
    if len(out["ranks"]) != len(users):
        raise ValueError("the pass blocks do not cover the listeners")
    return out


# Checks


def _p5_rank(s, t):
    """The P5 rank of slot t in s (-inf marks a non-candidate)."""
    return 1 + int(np.count_nonzero(s > s[t])) + int(np.count_nonzero(s[:t] == s[t]))


def verify(
    st,
    scorer,
    space,
    users,
    picks,
    first_ptr,
    first,
    u_arrays,
    P,
    p14_ranks,
    say,
    n_verify,
):
    """The dense and the full-history checks (module docstring). Returns
    their record."""
    n_slots = space.n_slots
    u_indptr, u_indices, u_data = u_arrays
    terms, T = scorer.terms, scorer.T
    C = scorer.C
    t_mats = {kind: space.colisten_t[kind] for kind in knn.KINDS}
    lists = {
        kind: (
            np.asarray(space.colisten[kind].indptr, np.int64),
            np.asarray(space.colisten[kind].indices, np.int32),
            np.asarray(space.colisten[kind].data, np.float64),
        )
        for kind in knn.KINDS
    }
    rng = np.random.default_rng(VERIFY_SEED)
    take = np.sort(
        rng.choice(len(users), size=min(n_verify, len(users)), replace=False)
    )
    ctx = {"scorer": scorer, "space": space, "chunk": CHUNK, "stride": CHECK_STRIDE}
    prog = Progress(
        f"{WORK_NAME}: the pass against the dense reference",
        len(take),
        say,
        unit=" listeners",
    )
    worst = 0.0
    for k in take:
        u = int(users[k])
        excl = np.unique(
            np.asarray(
                u_indices[int(u_indptr[u]) : int(u_indptr[u + 1])], dtype=np.int64
            )
        )
        cand = np.ones(n_slots, dtype=bool)
        cand[0] = False
        cand[excl] = False
        fi = first[int(first_ptr[k]) : int(first_ptr[k + 1])]
        for j, N in enumerate(POINTS):
            items = np.sort(fi[:N])
            w = np.full(len(items), W_ONE)
            state = cn.build_user_state(items, w, space)
            ref = ct.reference_scores(
                cn.full_catalog(state, space), terms, T, 0.0, space.tier
            )
            for c in range(C):
                s = np.where(cand, ref[c], -np.inf)
                for p in range(picks.shape[1]):
                    want = min(_p5_rank(s, int(picks[k, p])), RANK_CUT + 1)
                    got = int(P["ranks"][k, j, c, p])
                    if got != want:
                        raise ValueError(
                            f"listener {u}, N {N}, table {c}, "
                            f"pick {p}: rank {got}, the dense "
                            f"reference {want}"
                        )
            fast = knn.user_scores(t_mats, items, w)["both"]
            slow = knn.brute_scores(lists, items, w, n_slots)["both"]
            scale = max(1.0, float(np.max(np.abs(slow))))
            worst = max(worst, float(np.max(np.abs(fast - slow))) / scale)
            s_f = knn.candidate_scores(fast, excl)
            s_s = knn.candidate_scores(slow, excl)
            for p in range(picks.shape[1]):
                t = int(picks[k, p])
                got = int(P["ranks"][k, j, C, p])
                want = knn.sorted_rank(s_f, t, "id")
                tol = 1e-12 * max(1.0, abs(float(s_s[t])))
                lo, hi = knn.rank_bounds(s_s, t, tol)
                if got != want or not lo <= got <= hi:
                    raise ValueError(
                        f"listener {u}, N {N}, ItemKNN pick "
                        f"{p}: rank {got}, a full sort {want}, "
                        f"the second route {lo} to {hi}"
                    )
        # the full history against P14
        items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
        state = cn.build_user_state(items, weights, space)
        r, _, _ = cut_ranks(ctx, state, picks[k], excl, u)
        for c, arm in enumerate(("gam", "cold_gam")):
            want = np.minimum(p14_ranks[arm][k], RANK_CUT + 1)
            if not np.array_equal(r[c], want):
                raise ValueError(
                    f"listener {u}: the full-history ranks of "
                    f"{arm} {r[c].tolist()} differ from P14's "
                    f"{want.tolist()}"
                )
        hi_, hw = knn.history(u_indptr, u_indices, u_data, u)
        ks = knn.user_scores(t_mats, hi_, hw)["both"]
        got = knn.pick_ranks(knn.candidate_scores(ks, hi_), picks[k], "id")
        if not np.array_equal(got, p14_ranks["knn_both"][k]):
            raise ValueError(
                f"listener {u}: the full-history ItemKNN ranks " f"differ from P14's"
            )
        prog.advance(1)
    prog.finish()
    return {
        "listeners": int(len(take)),
        "knn_score_gap": worst,
        "full_history_equals_p14": True,
    }


# Results


def listener_metrics(r):
    """{metric: (n,) listener means over the picks} from (n, 2) ranks."""
    rf = np.asarray(r, dtype=np.float64)
    return {
        "ndcg_at_100": np.where(rf <= 100, 1.0 / np.log2(rf + 1.0), 0.0).mean(axis=1),
        "recall_at_100": (rf <= 100).astype(np.float64).mean(axis=1),
    }


def interval(x):
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    m = float(x.mean())
    se = float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan
    return {
        "mean": m,
        "se": p14._num(se),
        "lo": p14._num(m - Z95 * se),
        "hi": p14._num(m + Z95 * se),
        "n": int(n),
    }


def summarize(ranks, full, points=None):
    """The curve from the stored ranks: a pure function, so the read-back
    recomputes it. ranks: {arm: (n, len(points), 2)} of CURVE_ARMS. full:
    {arm: (n, 2)} P14's ranks of ARMS."""
    points = POINTS if points is None else tuple(int(N) for N in points)
    out = {"points": [int(N) for N in points] + ["full"], "arms": {}}
    for arm in ARMS:
        rows = []
        for j, N in enumerate(points):
            r = full["popularity"] if arm == "popularity" else ranks[arm][:, j, :]
            m = listener_metrics(r)
            rows.append(dict({k: interval(m[k]) for k in METRICS}, n=int(N)))
        m = listener_metrics(full[arm])
        rows.append(dict({k: interval(m[k]) for k in METRICS}, n="full"))
        out["arms"][arm] = rows
    return out


def provenance(model_dir):
    """The SHA-256 of the input files and the markers behind the large
    ones."""
    d = Path(model_dir)
    out = cold.provenance(d)
    for name in (
        "p14.json",
        "p14.npz",
        "p14_cold_gam.json",
        "p14_cold_gam.npz",
        "train_manifest.json",
        "csr_meta.json",
        "id_spaces.json",
    ):
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    for stage in ("p6_als", "p9b_colisten", "p4_features"):
        try:
            out[f"marker:{stage}"] = file_sha256(marker_path(stage))
        except FileNotFoundError:
            out[f"marker:{stage}"] = None
    return out


def run(
    st, io, model_dir, work, say=print, n_verify=VERIFY_USERS, inputs=None, workers=None
):
    """The cold-start curve (module docstring). st: a StageP13b2 whose
    loaders give the item side, the CSR and the slots. io: an Inputs.
    Returns the output doc."""
    t0 = time.time()
    d, work = Path(model_dir), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    base = st._base_context()
    n_slots, _ = st._id_slots()
    G = p14.load_models(d, base)
    terms = G["terms"]
    T = np.asarray(G["tables"][list(TABLE_ROWS)], dtype=np.float64)
    z = gkd.load_npz(d / p14.OUT_FILES[1])
    if not np.array_equal(np.asarray(z["tables"]), G["tables"]):
        raise ValueError(
            "p14.npz holds other tables than the models of " "p14.load_models"
        )
    keep = np.asarray(z["history"], dtype=np.int64) >= MIN_PAIRS
    users = np.asarray(z["user_id"], dtype=np.int64)[keep]
    picks = np.asarray(z["picks"], dtype=np.int64)[keep]
    p14_ranks = {a: np.asarray(z[f"rank_{a}"], dtype=np.int64)[keep] for a in ARMS}
    if not len(users):
        raise ValueError(f"no eval listener has {MIN_PAIRS} training pairs")
    say(
        f"{WORK_NAME}: {len(users):,} of {int(len(keep)):,} eval listeners "
        f"have at least {MIN_PAIRS} training pairs. Cut points "
        f"{', '.join(map(str, POINTS))}. Saved work: {work}"
    )
    u_arrays = st._csr()
    u_indptr, u_indices = u_arrays[0], u_arrays[1]
    if not np.array_equal(
        np.asarray(u_indptr[users + 1], np.int64)
        - np.asarray(u_indptr[users], np.int64),
        np.asarray(z["history"], np.int64)[keep],
    ):
        raise ValueError("p14.npz's history lengths differ from the CSR")
    kn = keep_n()
    fkey = {
        "version": VERSION,
        "inputs": io.key(),
        "keep": kn,
        "users": p14._array_key(users),
    }
    path, meta = work / "first_pairs.npz", work / "first_pairs.json"
    if path.exists() and load_json(meta) == {"k": s13._sha(fkey)}:
        f = np.load(path)
        first_ptr, first, first_ts, counts = (
            f["indptr"],
            f["items"],
            f["first_ts"],
            f["counts"],
        )
        say(f"{WORK_NAME}: the first pairs are saved, so no query runs")
    else:
        say(f"{WORK_NAME}: the first {kn} train pairs of each listener with " f"DuckDB")
        rows, cnt = io.first_pairs(users, kn, say)
        first_ptr, first, first_ts, counts = first_arrays(rows, cnt, users, kn)
        gr._atomic_savez(
            path, indptr=first_ptr, items=first, first_ts=first_ts, counts=counts
        )
        atomic_json_write({"k": s13._sha(fkey)}, meta)
    check_first(users, first_ptr, first, first_ts, counts, u_indptr, u_indices, kn)
    say(
        f"{WORK_NAME}: the first pairs follow the CSR and the time order. "
        f"Loading the item side (no progress lines in this step)"
    )
    t1 = time.time()
    space = st._space(st._sources(n_slots))
    scorer = ct.CatalogScorer(terms, T, space, chunk=CHUNK)
    if scorer.tilt is not None:
        raise ValueError("the pass supports no term with a multiplier")
    say(f"{WORK_NAME}: item side and scorer ready in " f"{_clock(time.time() - t1)}")
    key = {
        "version": VERSION,
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, knn, p14)
        },
    }
    t2 = time.time()
    P = curve_pass(
        st,
        scorer,
        space,
        users,
        picks,
        first_ptr,
        first,
        u_arrays,
        work,
        key,
        say,
        workers,
    )
    pass_secs = time.time() - t2
    P["ranks"] = np.asarray(P["ranks"], dtype=np.int64)
    if int(P["ranks"].min()) < 1 or int(P["ranks"][:, :, :2].max()) > RANK_CUT + 1:
        raise ValueError("a rank lies outside its range")
    chk = verify(
        st,
        scorer,
        space,
        users,
        picks,
        first_ptr,
        first,
        u_arrays,
        P,
        p14_ranks,
        say,
        n_verify,
    )
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    ranks = {arm: P["ranks"][:, :, c, :] for c, arm in enumerate(CURVE_ARMS)}
    res = summarize(ranks, p14_ranks)
    out = {
        "version": VERSION,
        "declared": DECLARED,
        "protocol": protocol(),
        "data": {
            "eval_listeners": int(len(keep)),
            "listeners": int(len(users)),
            "min_pairs": MIN_PAIRS,
            "n_slots": int(n_slots),
        },
        "models": {
            "final": G["rec"]["fit_name"],
            "final_kept": list(G["M"]["kept"]),
            "cold": G["cold"],
        },
        "results": res,
        "check": dict(
            chk,
            first_pairs=True,
            scores_equal_reference=True,
            slots_checked=int(np.sum(P["checked"])),
            probe_equal=True,
        ),
        "pass_info": {"mean_scored_share": float(np.mean(P["scored"]) / n_slots)},
        "inputs": dict(inputs or {}),
        "seconds": {"pass_this_run": pass_secs},
    }
    arrays = {
        "user_id": users,
        "picks": picks,
        "points": np.asarray(POINTS, dtype=np.int64),
        "scored": np.asarray(P["scored"], dtype=np.int64),
        "first_indptr": first_ptr,
        "first_items": first,
        "first_ts": first_ts,
    }
    arrays.update({f"rank_{a}": ranks[a] for a in CURVE_ARMS})
    arrays.update({f"p14_rank_{a}": p14_ranks[a] for a in ARMS})
    gr._atomic_savez(d / OUT_FILES[1], **arrays)
    atomic_json_write(p14._clean(out), d / OUT_FILES[0])
    check_outputs(d)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(p14._clean(out), d / OUT_FILES[0])
    write_report(p14._norm(out), d / "reports" / REPORT_NAME)
    g = res["arms"]["gam"]
    say(
        f"{WORK_NAME}: GAM NDCG@100 "
        + ", ".join(f"N {r['n']}: {r['ndcg_at_100']['mean']:.5f}" for r in g)
        + f". Done in {_clock(time.time() - t0)}"
    )
    return out


def check_outputs(d):
    """Read the outputs back: the results follow from the stored ranks,
    every rank lies in its range, and the stored P14 ranks equal
    p14.npz's."""
    d = Path(d)
    doc = load_json(d / OUT_FILES[0])
    if doc is None:
        raise ValueError(f"{OUT_FILES[0]} is missing")
    z = gkd.load_npz(d / OUT_FILES[1])
    zp = gkd.load_npz(d / p14.OUT_FILES[1])
    pos = np.searchsorted(zp["user_id"], z["user_id"])
    pos = np.minimum(pos, len(zp["user_id"]) - 1)
    if not np.array_equal(zp["user_id"][pos], z["user_id"]) or not np.array_equal(
        zp["picks"][pos], z["picks"]
    ):
        raise ValueError("the stored listeners or picks are not P14's")
    if np.any(zp["history"][pos] < int(doc["data"]["min_pairs"])):
        raise ValueError("a stored listener has too few training pairs")
    for a in ARMS:
        if not np.array_equal(zp[f"rank_{a}"][pos], z[f"p14_rank_{a}"]):
            raise ValueError(f"the stored P14 ranks of {a} differ from " f"p14.npz")
    ranks = {a: z[f"rank_{a}"] for a in CURVE_ARMS}
    for a in ("gam", "cold_gam"):
        if ranks[a].min() < 1 or ranks[a].max() > RANK_CUT + 1:
            raise ValueError(
                f"a stored rank of {a} lies outside 1 to " f"{RANK_CUT + 1}"
            )
    if [int(x) for x in z["points"]] != list(doc["protocol"]["points"]):
        raise ValueError("the stored cut points differ from the protocol's")
    again = p14._norm(
        summarize(
            {a: z[f"rank_{a}"] for a in CURVE_ARMS},
            {a: z[f"p14_rank_{a}"] for a in ARMS},
            z["points"],
        )
    )
    if again != p14._norm(doc["results"]):
        raise ValueError("the stored results do not follow from the stored " "ranks")


# Report


def write_report(out, path):
    res = out["results"]
    pts = res["points"]

    def cell(x):
        return (
            f"{x['mean']:.5f} ({x['lo']:.5f} to {x['hi']:.5f})"
            if x.get("lo") is not None
            else f"{x['mean']:.5f}"
        )

    L = [
        "# The cold-start curve",
        "",
        f"- Rules: the docstrings of pipeline/p14.py and "
        f"pipeline/cold_curve.py, declared {out['declared']}.",
        f"- Listeners: {out['data']['listeners']:,} of "
        f"{out['data']['eval_listeners']:,} eval listeners have at least "
        f"{out['data']['min_pairs']} training pairs. Each point ranks P5's "
        f"candidates of the full history. The history keeps the first N "
        f"training pairs by first listen, one play each.",
        f"- Models: {out['models']['final']} and the cold-track GAM "
        f"({', '.join(out['models']['cold']['kept'])}). Ranks are exact up "
        f"to {RANK_CUT}. Popularity ignores the history, so its value is "
        f"P14's at every point.",
        f"- Checks: {out['check']['slots_checked']:,} sampled slot scores "
        f"equal the dense reference exactly, "
        f"{out['check']['listeners']} listeners equal it over the full "
        f"catalog at every point, and their full-history ranks equal "
        f"P14's. The results read back.",
        "",
        "Values: the mean over the listeners with a 95 percent interval. "
        "Descriptive, no test.",
        "",
    ]
    for metric, title in (("ndcg_at_100", "NDCG@100"), ("recall_at_100", "R@100")):
        L += [
            f"## {title}",
            "",
            "| arm | "
            + " | ".join(
                f"N = {p}" if p != "full" else "full history (P14)" for p in pts
            )
            + " |",
            "|---|" + "---|" * len(pts),
        ]
        for arm in ARMS:
            L.append(
                f"| {ARM_LABELS[arm]} | "
                + " | ".join(cell(r[metric]) for r in res["arms"][arm])
                + " |"
            )
        L.append("")
    atomic_text_write("\n".join(L).replace(";", ",") + "\n", path)


# Runner stage


class StageColdCurve(ScriptStage):
    """The cold-start curve as a runner stage."""

    name = "cold_start_curve"
    upstream = ["p14_eval", "p14_cold_gam", "p13c_final", "p6_als", "p9b_colisten"]

    def __init__(self, model_dir=None, scratch_dir=None):
        super().__init__(model_dir, scratch_dir)
        self.n_verify = VERIFY_USERS
        self.workers = None

    def _stage(self):
        return s13b2.StageP13b2()

    def _io(self):
        return Inputs(self.model_dir)

    def outputs(self):
        d = self.model_dir
        return {
            "results": d / OUT_FILES[0],
            "arrays": d / OUT_FILES[1],
            "report": d / "reports" / REPORT_NAME,
        }

    def provenance(self):
        return provenance(self.model_dir)

    def recorded_inputs(self):
        doc = load_json(self.outputs()["results"])
        return None if doc is None else doc.get("inputs")

    def run(self, params, files):
        with log_file(self.model_dir / "logs" / LOG_NAME):
            run(
                self._stage(),
                self._io(),
                self.model_dir,
                self.scratch_dir / WORK_NAME,
                say=log.info,
                n_verify=self.n_verify,
                inputs=self.provenance(),
                workers=self.workers,
            )
        return self.output_info()

    def check(self):
        check_outputs(self.model_dir)
