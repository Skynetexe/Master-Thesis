"""Stage P13a: the audit before the fit, the base fit of the transparent
model with its penalty path, and the selection pass.

Chapter 4 of the thesis gives the design and its evidence. pipeline/gam.py
holds
the model, gam_audit.py the statistics and catalog.py the full-catalog
ranking. P13b (the experiment arms) and P13c (stability) build on this
stage's knots, audit and chosen penalty.

Steps:
1. Training rows: the train-split listeners of P11's fusion_pairs.npz,
   with their concept features from P12's fusion_features.npz. The
   selection listeners never train. Each listener has 1 or 2 positives
   and 32 negatives (16 uniform, 16 p_pop). Loss: the sampled softmax
   with the form-A correction (Blanc & Rendle 2018): each negative j gets
   the logit offset -log(32 q_mix(j)), with q_mix = 0.5 q_uniform
   + 0.5 q_pop from P11's exact probabilities, the balance heuristic for
   the 16 + 16 draw (Veach & Guibas 1995). Each positive has the weight
   1 / n_u, so every listener counts once, as in the evaluation metric
   (Covington, Adams & Sargin 2016 weighted users equally).
2. Mainstreaminess m_u: gam.mainstreaminess_raw() over the training
   history, standardized over the training listeners and clamped at their
   0.5 and 99.5 percentiles. The constants are frozen for P14.
3. Knots per concept (gam.knot_rule): 30 quantiles of the positives'
   values, 9 quantiles of the negatives' values, and 0. Only the shape
   part counts: rows with data, and for concepts with a zero state, a
   value above 0.
4. The audit before the fit:
   a. tier support: audio coverage and audio fit per bridge tier.
   b. missing states within popularity deciles: for every concept, the
      AUC of its "has data" flag between positives and each negative kind
      within deciles of the positives' popularity, with a 95 percent
      listener cluster bootstrap interval. A flag whose two intervals lie
      inside 0.5 +- 0.02 carries no signal beyond popularity, and its
      missing constant is tied to the neutral level (gam.Layout).
   c. concurvity: the mgcv-style worst and estimate indices of every term
      on the within-listener centered design (gam_audit.concurvity).
5. The penalty path: 12 weights of the second-difference penalty, from
   10^0.5 to 10^-5 in half decades, each fit warm-started from the one
   before (Friedman, Hastie & Tibshirani 2010), with a ridge of 1e-6.
   Each fit takes damped Newton steps with the exact Hessian (gam.fit),
   which reach a largest gradient component near 1e-12 in a few steps,
   also along the directions with little curvature. The tables are
   centered on the training p_pop negatives, the frozen
   popularity-weighted background of the thesis reports.
6. The selection pass: every configuration ranks the full catalog for
   the val_last and val_first picks of the 20,000 selection listeners,
   under the P5 rules, the ALS protocol of als_tune.py. Score: the mean
   NDCG@100 over the two picks. Choice: the one-standard-error rule on
   the paired per-listener differences (gam_audit.one_se_choice), where
   the simplest configuration has the strongest penalty. The ranks are
   exact up to RANK_CUT (100) and RANK_CUT + 1 beyond it, which is all
   that NDCG@100 and recall@100 read: catalog.ranks_cut() stops a
   listener as soon as both picks are known to rank below 100 in every
   configuration. A timed probe of a few listeners runs first and logs
   an estimate of the pass. The blocks are saved in the order they
   finish.
7. The two-run check: the chosen weight again, from a random start
   instead of the warm start. A convex fit must reach the same tables, to
   the solver tolerance.
8. After the fit, for the chosen configuration: the mean of each term
   over two backgrounds (p_pop and uniform negatives) with the share of
   positive rows whose contribution changes sign between them, three
   global importance measures, and the observed concurvity.

Outputs:
- p13a_model.json: parameters, knots, terms, audit, path (the parameters
  and the raw and centered tables of every configuration), the choice,
  the two-run check and the analyses after the fit.
- p13a_arrays.npz: the selection listeners, their picks and the ranks of
  every configuration (exact up to 100, 101 means beyond 100), the slots
  scored per selection listener, and the raw mainstreaminess of every
  listener in the pairs and in the selection sample.
- reports/p13a_report.md.

Resume: the mainstreaminess values, each fit of the path and each block
of the selection pass are saved in SCRATCH_DIR/p13a, keyed by the
parameters, the upstream outputs and the code. A stopped run continues.
The environment variable P13A_WORKERS sets the number of selection
workers (default: the CPU count, at most 16). It is in no key, because
the results do not depend on it, so a run can continue with fewer
workers after a memory warning.

verify():
- the knots, the audit AUCs and the neutral ties follow from the inputs.
- every configuration is a stationary point: the objective recomputed at
  its stored parameters equals the stored value, and the largest
  projected gradient component is below KKT_TOL.
- the centered tables have mean 0 over the background, and the tied
  missing constants sit at the neutral level.
- the scores of sampled training rows equal an independent evaluation
  with np.interp on the raw P12 values.
- for VERIFY_USERS selection listeners and three configurations, the
  catalog scorer's scores equal the independent dense reference of
  catalog.reference_scores() on concepts.full_catalog() within
  SCORE_TOL, and every stored rank equals the reference rank censored at
  RANK_CUT. The stored ranks also equal the full-catalog count of
  CatalogScorer.ranks() censored at the cut, and a new run of
  ranks_cut() gives them again.
- the choice follows from the stored ranks, and the two-run difference
  is below TWO_RUN_TOL.
"""

import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from . import artists as ar
from . import catalog as ct
from . import concept_data as cd
from . import concepts as cn
from . import evaluation as ev
from . import fusion_pairs as fp
from . import gam
from . import gam_audit as ga
from . import selection as sel
from .als import Progress, _clock, threadpool_limits
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    connect,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from .config import MODEL_DIR, SCRATCH_DIR

VERSION = "p13a-v1"
BASE_CONCEPTS = (
    "tag_affinity",
    "artist_tag_affinity",
    "genre_affinity",
    "artist_affinity",
    "colisten_same_artist",
    "colisten_other_artists",
    "audio_fit",
)
AUDIT_CONCEPTS = BASE_CONCEPTS + ("work_affinity",)
ZERO_STATE = set(cn.TERM_CONCEPTS) | set(cn.CREDIT_CONCEPTS) | set(cn.COLISTEN_CONCEPTS)
DEV_TIERS = (cn.MATCH_TIERS.index("fuzzy_dur"),)
KNOT_POS_LEVELS = tuple(float(x) for x in np.round(np.linspace(0.005, 0.995, 30), 6))
KNOT_NEG_LEVELS = (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
LAMBDAS = tuple(float(10.0 ** (k / 2.0)) for k in range(1, -11, -1))
RIDGE = 1e-6
SOLVER = {
    "method": "newton",
    "newton_steps": 60,
    "newton_target": 1e-12,
    "newton_accept": 1e-9,
    "maxcor": 20,
    "ftol": 1e-14,
    "gtol": 1e-10,
    "maxiter": 5000,
    "maxfun": 10000,
    "pg_target": 1e-9,
    "restarts": 3,
}
MIX = 0.5
N_STRATA = 10
N_BOOT = 200
AUDIT_SEED = 2026092713
NEUTRAL_MARGIN = 0.02
MAINSTREAM_CLAMP = (0.005, 0.995)
KKT_TOL = 1e-8
OBJ_RTOL = 1e-10
TWO_RUN_TOL = 1e-6
SCORE_TOL = 1e-9
FIT_BLOCKS = 64
TASKS_PER_WORKER = 8
FIT_WORKERS = max(1, min(16, os.cpu_count() or 1))
# Execution only, outside every key: P13A_WORKERS=8 python3 -m ... sets it.
SELECTION_WORKERS = max(
    1, int(os.environ.get("P13A_WORKERS", min(16, os.cpu_count() or 1)))
)
RANK_CUT = 100  # NDCG@100 reads no rank beyond 100
SELECTION_BLOCK = 25  # listeners per saved block
SELECTION_CHUNK = ct.RANK_CHUNK
PROBE_USERS = 6  # listeners timed before the pool starts
MEMORY_RESERVE = 2 << 30  # bytes kept free for the rest of the machine
MEMORY_LOG_SECONDS = 300.0
VERIFY_USERS = 24
VERIFY_ROWS = 200_000
VERIFY_MAINSTREAM = 2_000
SEED = 2026092714
SHAPE_LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

_CTX = {}


def _params():
    return {
        "version": VERSION,
        "base_concepts": list(BASE_CONCEPTS),
        "audit_concepts": list(AUDIT_CONCEPTS),
        "dev_tiers": [cn.MATCH_TIERS[t] for t in DEV_TIERS],
        "knot_pos_levels": list(KNOT_POS_LEVELS),
        "knot_neg_levels": list(KNOT_NEG_LEVELS),
        "lambdas": list(LAMBDAS),
        "ridge": RIDGE,
        "solver": SOLVER,
        "mix": MIX,
        "strata": N_STRATA,
        "boot": N_BOOT,
        "audit_seed": AUDIT_SEED,
        "neutral_margin": NEUTRAL_MARGIN,
        "mainstream_clamp": list(MAINSTREAM_CLAMP),
        "kkt_tol": KKT_TOL,
        "two_run_tol": TWO_RUN_TOL,
        "selection_seed": sel.SEED,
        "selection_size": sel.SIZE,
        "rank_cut": RANK_CUT,
    }


def _atomic_savez(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")  # np.savez keeps .npz
    np.savez(tmp, **arrays)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


def _array_sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode("utf-8"))
        h.update(str(a.shape).encode("utf-8"))
        h.update(a.tobytes())
    return h.hexdigest()


def _blocks(costs, n_tasks):
    """Contiguous index ranges with near-equal summed cost."""
    cum = np.cumsum(np.asarray(costs, dtype=np.int64))
    if len(cum) == 0:
        return []
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n_tasks) / n_tasks)
    edges = np.unique(np.concatenate(([0], cuts, [len(cum)])))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


class _Pool:
    """A fork pool that keeps its workers for many calls, or a loop when
    workers is 1. map() returns the results in task order, so a sum over
    them does not depend on the worker count. unordered() yields each
    result as its task finishes, for work whose results are stored by
    position. After an error or Ctrl+C the queued tasks are cancelled and
    the workers end at once, without waiting for the running tasks."""

    def __init__(self, workers):
        self.workers = int(workers)
        self.pool = None

    def __enter__(self):
        if self.workers > 1:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers, mp_context=mp.get_context("fork")
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.pool is None:
            return False
        if exc_type is None:
            self.pool.shutdown()
            return False
        procs = list((getattr(self.pool, "_processes", None) or {}).values())
        self.pool.shutdown(wait=False, cancel_futures=True)
        for p in procs:
            if p.is_alive():
                p.terminate()
        for p in procs:
            p.join(timeout=30)
        return False

    def map(self, fn, tasks):
        if self.pool is None:
            return [fn(t) for t in tasks]
        return list(self.pool.map(fn, tasks))

    def unordered(self, fn, tasks):
        """(position in tasks, result) pairs in the order the tasks
        finish."""
        if self.pool is None:
            for i, t in enumerate(tasks):
                yield i, fn(t)
            return
        futures = {self.pool.submit(fn, t): i for i, t in enumerate(tasks)}
        for f in as_completed(futures):
            yield futures[f], f.result()


# Workers


def _fit_task(task):
    b, T, D = task
    return gam.block_loss_grad(T, D, _CTX["blocks"][b], _CTX["L"])


def _gram_task(b):
    return gram_block(_CTX["blocks"][b], _CTX["L"])


def _hess_task(task):
    b, T, D = task
    with threadpool_limits(1):  # 16 workers, one BLAS thread each
        return gam.block_hessian(T, D, _CTX["blocks"][b], _CTX["L"])


def _stats_task(task):
    b, T, D, e_unif = task
    return stats_block(_CTX["blocks"][b], T, D, e_unif)


def _mainstream_task(task):
    lo, hi = task
    c = _CTX
    out = np.empty(hi - lo, dtype=np.float64)
    for j, k in enumerate(range(lo, hi)):
        items, weights = cd.history_of(
            c["u_indptr"], c["u_indices"], c["u_data"], int(c["m_users"][k])
        )
        out[j] = gam.mainstreaminess_raw(items, weights, c["popularity"])
    return out


def _selection_one(c, k):
    """(C, 2) ranks of selection listener k, exact up to RANK_CUT, and the
    number of slots scored."""
    scorer, space = c["scorer"], c["space"]
    items, weights = cd.history_of(
        c["u_indptr"], c["u_indices"], c["u_data"], int(c["sel_users"][k])
    )
    state = cn.build_user_state(items, weights, space)
    return scorer.ranks_cut(
        state,
        space,
        float(c["sel_m"][k]),
        c["sel_picks"][k],
        items,
        RANK_CUT,
        chunk=SELECTION_CHUNK,
    )


def _selection_task(task):
    lo, hi = task
    c = _CTX
    out = np.empty((c["scorer"].C, hi - lo, 2), dtype=np.int32)
    scored = np.empty(hi - lo, dtype=np.int64)
    with threadpool_limits(1):  # 16 workers, one BLAS thread each
        for j, k in enumerate(range(lo, hi)):
            out[:, j, :], scored[j] = _selection_one(c, k)
    return out, scored


def _memory():
    """(available bytes, swap bytes in use) from psutil, or None when
    psutil is missing."""
    try:
        import psutil
    except ImportError:
        return None
    return (int(psutil.virtual_memory().available), int(psutil.swap_memory().used))


def _physical_cores():
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


def _gb(n_bytes):
    return n_bytes / float(1 << 30)


# The design


def training_rows(p, m_neg):
    """The rows of the train-split listeners, grouped by listener, and the
    listener layout: (train positions into p's users, rows, row counts,
    positive counts). Raises unless every listener holds its positives
    first and then exactly m_neg negatives."""
    split, indptr, kind = p["split"], p["indptr"], p["kind"]
    tr = np.flatnonzero(split == fp.SPLIT_TRAIN)
    lens = np.diff(indptr)[tr]
    row_user = np.repeat(np.arange(len(split)), np.diff(indptr))
    rows = np.flatnonzero(split[row_user] == fp.SPLIT_TRAIN)
    n_pos = lens - m_neg
    if len(tr) == 0 or (n_pos < 1).any():
        raise ValueError(
            "a training listener lacks a positive or has " "fewer than m negatives"
        )
    k = kind[rows]
    starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
    place = np.arange(len(rows)) - np.repeat(starts, lens)
    want_pos = place < np.repeat(n_pos, lens)
    if not np.array_equal(k == fp.KIND_POSITIVE, want_pos):
        raise ValueError(
            "a training listener does not hold its positives "
            "first and then only negatives"
        )
    return tr, rows, lens.astype(np.int64), n_pos.astype(np.int64)


def row_offsets(
    kind, item, row_user, n_allowed, pop_allowed, p_pop_slot, m_neg, mix=MIX
):
    """Logit offsets of form A: 0 for a positive, -log(m q_mix(j)) for a
    negative, q_mix = mix / n_allowed + (1 - mix) p_pop(j) / pop_allowed
    of the row's listener."""
    q = (
        mix / n_allowed[row_user]
        + (1.0 - mix) * p_pop_slot[item] / pop_allowed[row_user]
    )
    off = -np.log(m_neg * q)
    return np.where(kind == fp.KIND_POSITIVE, 0.0, off)


def concept_column(f, rows, concept):
    j = cn.CONTINUOUS.index(concept)
    return (
        np.asarray(f["values"][rows, j], dtype=np.float64),
        np.asarray(f["has"][rows, j], dtype=bool),
    )


def compute_knots(f, rows, kind):
    """{concept: knots} from the training rows (gam.knot_rule)."""
    pos = kind == fp.KIND_POSITIVE
    out = {}
    for c in BASE_CONCEPTS + ("popularity",):
        v, h = concept_column(f, rows, c)
        shape = h & ((v != 0.0) if c in ZERO_STATE else True)
        out[c] = gam.knot_rule(
            v[shape & pos], v[shape & ~pos], KNOT_POS_LEVELS, KNOT_NEG_LEVELS
        )
    return out


def base_terms(knots):
    """The base configuration of the design: the
    seven concepts without work_affinity, popularity, the popularity tilt
    by mainstreaminess and the fuzzy_dur deviation curve of audio fit."""
    terms = [
        gam.Term(c, c, knots[c], zero_state=c in ZERO_STATE, missing_state=True)
        for c in BASE_CONCEPTS
    ]
    terms.append(gam.Term("popularity", "popularity", knots["popularity"]))
    terms.append(
        gam.Term(
            "popularity_tilt",
            "popularity",
            knots["popularity"],
            multiplier="mainstreaminess",
        )
    )
    terms.append(
        gam.Term(
            "audio_fit_fuzzy_dur",
            "audio_fit",
            knots["audio_fit"],
            tiers=DEV_TIERS,
            centered=True,
        )
    )
    return terms


def encode_rows(terms, f, rows):
    """(terms, rows) int16 slots and float64 positions."""
    slot = np.empty((len(terms), len(rows)), dtype=np.int16)
    w = np.empty((len(terms), len(rows)), dtype=np.float64)
    tier = np.asarray(f["match_tier"][rows], dtype=np.int64)
    cache = {}
    for k, t in enumerate(terms):
        if t.source not in cache:
            cache = {t.source: concept_column(f, rows, t.source)}
        v, h = cache[t.source]
        slot[k], w[k] = gam.encode(t, v, h, tier)
    return slot, w


def shape_quantiles(terms, f, rows, kind, slot):
    """{term: the source values at SHAPE_LEVELS among the positive rows on
    the term's shape}, the x positions of the report's shape table."""
    pos = kind == fp.KIND_POSITIVE
    out = {}
    for k, t in enumerate(terms):
        v, _ = concept_column(f, rows, t.source)
        m = pos & (slot[k] < t.K)
        out[t.name] = (
            [float(x) for x in np.quantile(v[m], SHAPE_LEVELS)] if m.any() else None
        )
    return out


def reference_means(terms, slot, w, mask_ref):
    """Mean basis vectors over the reference rows: all rows (centering),
    the rows with data (neutral level) and the rows where a centered term
    applies (its own reference)."""
    means, data, dev = {}, {}, {}
    for k, t in enumerate(terms):
        s, ww = slot[k], w[k]
        means[t.name] = gam.basis_mean(t, s, ww, mask_ref)
        if t.missing_state:
            data[t.name] = gam.basis_mean(t, s, ww, mask_ref & (s < t.K + gam.MISS))
        if t.centered:
            dev[t.name] = gam.basis_mean(t, s, ww, mask_ref & (s < t.K))
    return means, data, dev


def make_blocks(
    terms,
    layout,
    slot,
    w,
    m_row,
    offset,
    lens,
    n_pos,
    n_blocks,
    m_neg,
    user_weight=None,
):
    """The fit blocks: contiguous listener ranges with near-equal rows.
    Each block holds views of the row arrays and its local group layout.
    user_weight: None (every listener 1) or bootstrap counts."""
    U = len(lens)
    uw = (
        np.ones(U) if user_weight is None else np.asarray(user_weight, dtype=np.float64)
    )
    total = float(uw.sum())
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    off = np.asarray(layout.offsets, dtype=np.int64)
    mult_k = [k for k, t in enumerate(terms) if t.multiplier is not None]
    blocks = []
    for lo, hi in _blocks(lens, n_blocks):
        r0, r1 = int(starts[lo]), int(starts[hi - 1] + lens[hi - 1])
        ls = lens[lo:hi]
        ss = starts[lo:hi] - r0
        np_ = n_pos[lo:hi]
        pos_rows = np.concatenate([ss[j] + np.arange(np_[j]) for j in range(hi - lo)])
        pos_user = np.repeat(np.arange(hi - lo), np_)
        neg_rows = (ss + np_)[:, None] + np.arange(m_neg)[None, :]
        pos_weight = (uw[lo:hi] / np_)[pos_user] / total
        blocks.append(
            {
                "slot": slot[:, r0:r1],
                "w": w[:, r0:r1],
                "off": off,
                "mult": {k: m_row[r0:r1] for k in mult_k},
                "offset": offset[r0:r1],
                "pos_rows": pos_rows,
                "pos_user": pos_user,
                "pos_weight": pos_weight,
                "neg_rows": neg_rows,
                "row_user": np.repeat(np.arange(hi - lo), ls),
                "user_start": ss,
                "user_len": ls,
                "users": (lo, hi),
            }
        )
    return blocks


def gram_block(blk, L):
    """X'X - sum_u n_u xbar_u xbar_u' over one block: the Gram matrix of
    the design after each listener's mean row is subtracted."""
    n = blk["w"].shape[1]
    with threadpool_limits(1):
        G = gam.block_xax(blk, np.ones(n), L)
        U = len(blk["user_len"])
        A = sp.csr_matrix((np.ones(n), (blk["row_user"], np.arange(n))), shape=(U, n))
        S = (A @ gam.block_design(blk, L)).toarray()
        G -= S.T @ (S / blk["user_len"][:, None].astype(np.float64))
    return G


def stats_block(blk, T, D, e_unif):
    """Sums for the global importance and the background sign flips of
    one block, with centered tables T. e_unif: each term's mean over the
    uniform background (times m_u for the multiplier term)."""
    w = blk["w"]
    n_t, n = w.shape
    mult = blk.get("mult") or {}
    C = np.empty((n_t, n), dtype=np.float64)
    for k in range(n_t):
        v = gam.pwl_value(T, D, gam.block_index(blk, k), w[k])
        C[k] = mult[k] * v if k in mult else v
    s = np.zeros(n, dtype=np.float64)
    for k in range(n_t):
        s += C[k]
    pos = blk["pos_rows"]
    Cp = C[:, pos]
    flips = np.zeros(n_t)
    for k in range(n_t):
        shift = e_unif[k] * mult[k][pos] if k in mult else e_unif[k]
        flips[k] = float(np.count_nonzero(Cp[k] * (Cp[k] - shift) < 0))
    starts, lens = blk["user_start"], blk["user_len"]
    mC = np.add.reduceat(C, starts, axis=1) / lens
    ms = np.add.reduceat(s, starts) / lens
    dC = C - np.repeat(mC, lens, axis=1)
    ds = s - np.repeat(ms, lens)
    sd_u = np.sqrt(np.add.reduceat(dC * dC, starts, axis=1) / lens)
    return {
        "abs_pos": np.abs(Cp).sum(axis=1),
        "n_pos": int(len(pos)),
        "flips": flips,
        "sd_sum": sd_u.sum(axis=1),
        "n_users": int(len(lens)),
        "cov": (dC * ds).sum(axis=1),
        "var": float((ds * ds).sum()),
    }


# The stage


class StageP13a:
    name = "p13a_fit"
    upstream = [
        "p2_split",
        "p25_train",
        "p3_sampling",
        "p4_features",
        "p6_als",
        "p9b_colisten",
        "p11_fusion_pairs",
        "p12_pair_features",
    ]

    def __init__(self):
        self._cache = {}
        self._verified = None

    def outputs(self):
        return {
            "model": MODEL_DIR / "p13a_model.json",
            "arrays": MODEL_DIR / "p13a_arrays.npz",
            "report": MODEL_DIR / "reports" / "p13a_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        for label, mod in (
            ("gam_module", gam),
            ("gam_audit_module", ga),
            ("catalog_module", ct),
            ("concepts_module", cn),
            ("concept_data_module", cd),
            ("artists_module", ar),
            ("selection_module", sel),
        ):
            inputs[label] = file_sha256(Path(mod.__file__))
        return inputs

    def _code(self):
        return {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (gam, ga, ct, cn, cd, ar, sel)
        } | {Path(__file__).name: file_sha256(Path(__file__))}

    # The loaders below read the model folder and DuckDB. test_p13a.py
    # replaces them.
    def _pairs(self):
        if "pairs" not in self._cache:
            path = MODEL_DIR / "fusion_pairs.npz"
            z = np.load(path)
            self._cache["pairs"] = (
                {
                    k: np.asarray(z[k])
                    for k in (
                        "user_id",
                        "split",
                        "indptr",
                        "item",
                        "kind",
                        "n_allowed",
                        "pop_allowed",
                    )
                },
                json.loads(str(z["params"])),
                ev.npz_hash(path),
            )
        return self._cache["pairs"]

    def _features(self):
        if "features" not in self._cache:
            path = MODEL_DIR / "fusion_features.npz"
            z = np.load(path)
            self._cache["features"] = (
                {k: z[k] for k in ("values", "has", "match_tier")},
                json.loads(str(z["params"])),
                str(z["pairs_sha256"]),
                ev.npz_hash(path),
            )
        return self._cache["features"]

    def _proposals(self):
        z = np.load(MODEL_DIR / "neg_proposals.npz")
        return (
            np.asarray(z["item_ids"], dtype=np.int64),
            np.asarray(z["p_pop"], dtype=np.float64),
        )

    def _id_slots(self):
        return ev.load_id_slots()

    def _csr(self):
        mm = {"mmap_mode": "r"}
        return (
            np.load(MODEL_DIR / "csr_user_indptr.npy"),
            np.load(MODEL_DIR / "csr_user_indices.npy", **mm),
            np.load(MODEL_DIR / "csr_user_data.npy", **mm),
        )

    def _popularity(self, n_slots):
        """log1p(train listens) per slot, the same array as
        concept_data.load_sources() builds."""
        con = connect(4)
        try:
            cols = con.execute(
                f"SELECT s.item_id AS item_id, s.listens AS listens "
                f"FROM read_parquet('{MODEL_DIR / 'stats_item_train.parquet'}'"
                f") AS s"
            ).fetchnumpy()
        finally:
            con.close()
        return cd.popularity_array(cols["item_id"], cols["listens"], n_slots)

    def _selection(self):
        """The selection listeners and their (val_last, val_first) picks,
        through selection.py's own route."""
        con = connect(4)
        try:
            users = np.asarray(sel.selection_user_ids(con), dtype=np.int64)
            picks = np.asarray(sel.val_picks(con, users), dtype=np.int64)
        finally:
            con.close()
        return users, picks

    def _sources(self, n_slots):
        if "sources" not in self._cache:
            self._cache["sources"] = cd.load_sources(n_slots, log.info)
        return self._cache["sources"]

    def _space(self, src):
        if "space" not in self._cache:
            self._cache["space"] = cd.item_side(src, transposed=True)
        return self._cache["space"]

    def _als_reference(self):
        """The popularity reference and the ALS winner on the same picks,
        from als_tune.py's results, when present."""
        doc = load_json(MODEL_DIR / "als_tuning.json") or {}
        out = {}
        ref = doc.get("reference") or {}
        if "score" in ref:
            out["popularity"] = float(ref["score"])
        win = doc.get("winner") or {}
        if "score" in win:
            out["als"] = float(win["score"])
            out["als_label"] = (
                f"alpha {win.get('alpha')}, reg "
                f"{win.get('reg')}, {win.get('factors', 64)} "
                f"factors"
            )
        return out

    def _upstream_outputs(self):
        out = {}
        for stage in self.upstream:
            doc = load_json(marker_path(stage)) or {}
            out[stage] = {
                k: (v.get("sha256") if isinstance(v, dict) else v)
                for k, v in sorted((doc.get("outputs") or {}).items())
            }
        return out

    # Inputs

    def _inputs(self):
        """Everything the fit reads, checked. Cached for run() and
        verify()."""
        if "inputs" in self._cache:
            return self._cache["inputs"]
        p, p_params, pairs_sha = self._pairs()
        f, f_params, f_pairs_sha, features_sha = self._features()
        check(
            f_pairs_sha == pairs_sha,
            "fusion_features.npz was built from other pairs than the "
            "current fusion_pairs.npz",
        )
        check(
            list(f_params.get("concepts", [])) == list(cn.CONTINUOUS),
            "fusion_features.npz holds other concept columns than "
            "concepts.CONTINUOUS",
        )
        check(
            len(f["match_tier"]) == len(p["item"]),
            "the features and the pairs differ in length",
        )
        m_neg = 2 * int(p_params["neg_per_kind"])
        tr, rows, lens, n_pos = training_rows(p, m_neg)
        n_slots, id_min = self._id_slots()
        item_ids, p_pop = self._proposals()
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        row_user = np.repeat(np.arange(len(tr)), lens)
        kind = p["kind"][rows]
        item = p["item"][rows].astype(np.int64)
        offset = row_offsets(
            kind,
            item,
            row_user,
            p["n_allowed"][tr],
            p["pop_allowed"][tr],
            p_pop_slot,
            m_neg,
        )
        check(np.all(np.isfinite(offset)), "a logit offset is not finite")
        out = {
            "p": p,
            "p_params": p_params,
            "pairs_sha": pairs_sha,
            "f": f,
            "features_sha": features_sha,
            "m_neg": m_neg,
            "tr": tr,
            "rows": rows,
            "lens": lens,
            "n_pos": n_pos,
            "row_user": row_user,
            "kind": kind,
            "item": item,
            "offset": offset,
            "n_slots": n_slots,
            "id_min": id_min,
        }
        self._cache["inputs"] = out
        return out

    def _key(self):
        return {
            "params": _params(),
            "upstream": self._upstream_outputs(),
            "code": self._code(),
        }

    # Mainstreaminess

    def _mainstream(self, users, work, key):
        """Raw mainstreaminess of each listener in users, cached."""
        path = work / "mainstream.npz"
        state_path = work / "mainstream.json"
        k = {"key": _sha(key), "users": _array_sha(users)}
        if path.exists() and load_json(state_path) == k:
            z = np.load(path)
            if np.array_equal(z["users"], users):
                log.info("p13a: mainstreaminess resume")
                return z["raw"]
        n_slots, _ = self._id_slots()
        u_indptr, u_indices, u_data = self._csr()
        check(
            len(u_indptr) - 1 > int(users.max()),
            "a listener falls outside the user CSR",
        )
        hist = np.asarray(u_indptr[users + 1], dtype=np.int64) - np.asarray(
            u_indptr[users], dtype=np.int64
        )
        check(bool((hist > 0).all()), "a listener has no train history")
        tasks = _blocks(hist + 100, FIT_WORKERS * TASKS_PER_WORKER)
        _CTX.update(
            m_users=users,
            u_indptr=u_indptr,
            u_indices=u_indices,
            u_data=u_data,
            popularity=self._popularity(n_slots),
        )
        prog = Progress(
            "p13a: mainstreaminess",
            int(hist.sum()),
            log.info,
            unit=" M history entries",
            scale=10**6,
        )
        parts = []
        try:
            with _Pool(FIT_WORKERS) as pool:
                for (a, b), out in zip(tasks, pool.map(_mainstream_task, tasks)):
                    parts.append(out)
                    prog.advance(int(hist[a:b].sum()))
        finally:
            _CTX.clear()
        prog.finish()
        raw = np.concatenate(parts)
        _atomic_savez(path, users=users, raw=raw)
        atomic_json_write(k, state_path)
        return raw

    # Audit

    def _audit_missing(self, inp, f):
        """The within-decile AUC of every concept's data flag, against
        each negative kind, with the listener bootstrap interval."""
        rows, kind, row_user = inp["rows"], inp["kind"], inp["row_user"]
        U = len(inp["lens"])
        pop, _ = concept_column(f, rows, "popularity")
        pos = kind == fp.KIND_POSITIVE
        cuts = np.quantile(pop[pos], np.arange(1, N_STRATA) / N_STRATA)
        stratum = ga.strata_of(pop, cuts)
        out = {"cuts": [float(x) for x in cuts], "concepts": {}}
        for c in AUDIT_CONCEPTS:
            _, has = concept_column(f, rows, c)
            cp = ga.flag_counts(row_user[pos], stratum[pos], has[pos], U, N_STRATA)
            res = {}
            for kd in (fp.KIND_UNIFORM, fp.KIND_POP):
                m = kind == kd
                cn_ = ga.flag_counts(row_user[m], stratum[m], has[m], U, N_STRATA)
                res[fp.KIND_NAMES[kd]] = ga.stratified_flag_auc(
                    cp, cn_, N_BOOT, AUDIT_SEED
                )
            out["concepts"][c] = {
                "by_kind": res,
                "no_signal": bool(ga.no_signal(res, NEUTRAL_MARGIN)),
                "share_with_data": {
                    "positives": float(has[pos].mean()),
                    "uniform": float(has[kind == fp.KIND_UNIFORM].mean()),
                    "p_pop": float(has[kind == fp.KIND_POP].mean()),
                },
            }
            log.info(
                "p13a audit: %s has-data AUC within popularity deciles:"
                " uniform %.3f [%.3f, %.3f], p_pop %.3f [%.3f, %.3f]",
                c,
                res["uniform"]["auc"],
                res["uniform"]["lo"],
                res["uniform"]["hi"],
                res["p_pop"]["auc"],
                res["p_pop"]["lo"],
                res["p_pop"]["hi"],
            )
        return out

    def _audit_tiers(self, inp, f):
        rows, kind = inp["rows"], inp["kind"]
        tier = np.asarray(f["match_tier"][rows], dtype=np.int64)
        v, h = concept_column(f, rows, "audio_fit")
        out = []
        for t, name in enumerate(cn.MATCH_TIERS):
            row = {"tier": name}
            for label, m in (
                ("positives", kind == fp.KIND_POSITIVE),
                ("uniform", kind == fp.KIND_UNIFORM),
                ("p_pop", kind == fp.KIND_POP),
            ):
                mt = m & (tier == t)
                n = int(mt.sum())
                na = int((mt & h).sum())
                q = np.quantile(v[mt & h], (0.01, 0.5, 0.99)).tolist() if na else None
                row[label] = {"rows": n, "with_audio": na, "audio_fit_q": q}
            out.append(row)
        return out

    def run(self, params, files):
        t_start = time.time()
        check(
            RANK_CUT >= 100,
            "RANK_CUT must be at least 100, because "
            "NDCG@100 reads every rank up to 100",
        )
        outputs = self.outputs()
        work = SCRATCH_DIR / "p13a"
        work.mkdir(parents=True, exist_ok=True)
        key = self._key()
        inp = self._inputs()
        p, f = inp["p"], inp["f"]
        users_all = p["user_id"]
        sel_users, sel_picks = self._selection()
        check(
            len(sel_users) == len(np.unique(sel_users))
            and sel_picks.shape == (len(sel_users), 2),
            "the selection listeners or their picks are malformed",
        )
        check(
            not np.isin(sel_users, users_all[inp["tr"]]).any(),
            "a selection listener is a training listener",
        )
        log.info(
            "p13a: %d training listeners, %d training rows, %d " "selection listeners",
            len(inp["tr"]),
            len(inp["rows"]),
            len(sel_users),
        )

        # Mainstreaminess of every listener in the pairs and the selection.
        m_users = np.union1d(users_all, sel_users)
        m_raw_all = self._mainstream(m_users, work, key)
        tr_users = users_all[inp["tr"]]
        raw_tr = m_raw_all[np.searchsorted(m_users, tr_users)]
        mean, sd = float(raw_tr.mean()), float(raw_tr.std())
        check(sd > 0, "mainstreaminess does not vary over the listeners")
        z = (raw_tr - mean) / sd
        m_consts = {
            "mean": mean,
            "sd": sd,
            "lo": float(np.quantile(z, MAINSTREAM_CLAMP[0])),
            "hi": float(np.quantile(z, MAINSTREAM_CLAMP[1])),
        }
        m_user = gam.mainstreaminess(raw_tr, m_consts)
        m_row = m_user[inp["row_user"]]

        # Knots, terms, encodings.
        t0 = time.time()
        knots = compute_knots(f, inp["rows"], inp["kind"])
        terms = base_terms(knots)
        slot, w = encode_rows(terms, f, inp["rows"])
        log.info(
            "p13a: knots and the encoding of %d rows for %d terms in " "%s",
            len(inp["rows"]),
            len(terms),
            _clock(time.time() - t0),
        )
        ref_mask = inp["kind"] == fp.KIND_POP
        unif_mask = inp["kind"] == fp.KIND_UNIFORM
        means_ref, data_means, dev_means = reference_means(terms, slot, w, ref_mask)
        means_unif, _, _ = reference_means(terms, slot, w, unif_mask)
        n_zero = {
            t.name: int((slot[k] == t.K + gam.ZERO).sum()) for k, t in enumerate(terms)
        }
        n_miss = {
            t.name: int((slot[k] == t.K + gam.MISS).sum()) for k, t in enumerate(terms)
        }
        shape_q = shape_quantiles(terms, f, inp["rows"], inp["kind"], slot)

        # Audit.
        log.info(
            "p13a: audit of the missing states (%d bootstrap "
            "replicates per concept and kind)",
            N_BOOT,
        )
        missing = self._audit_missing(inp, f)
        neutral = tuple(c for c in BASE_CONCEPTS if missing["concepts"][c]["no_signal"])
        tiers = self._audit_tiers(inp, f)
        layout = gam.Layout(terms, n_zero, n_miss, neutral, data_means, dev_means)
        log.info(
            "p13a: %d terms, %d table entries, %d free parameters, "
            "neutral missing states: %s",
            len(terms),
            layout.L,
            layout.P,
            ", ".join(neutral) or "none",
        )
        blocks = make_blocks(
            terms,
            layout,
            slot,
            w,
            m_row,
            inp["offset"],
            inp["lens"],
            inp["n_pos"],
            FIT_BLOCKS,
            inp["m_neg"],
        )
        _CTX.update(blocks=blocks, L=layout.L)
        try:
            with _Pool(FIT_WORKERS) as pool:
                G = self._gram(pool, len(blocks), layout.L)
                path = self._fit_path(pool, layout, work, key, len(blocks))
        finally:
            _CTX.clear()
        conc_groups = {
            t.name: list(range(off, off + t.K + 2))
            for t, off in zip(terms, layout.offsets)
        }
        conc_full, conc_names, conc_pair = ga.concurvity(G, conc_groups)

        # Center every configuration on the p_pop negatives.
        for rec in path:
            T = layout.tables(rec["theta"])
            Tc, consts = gam.center(T, layout, means_ref)
            rec["tables_raw"] = T
            rec["tables"] = Tc
            rec["center_consts"] = consts
        # Free the training rows before the selection pass: it holds the
        # item side with the transposed co-listening lists, and every
        # worker holds one listener's concepts.
        del blocks, slot, w, m_row
        inp = p = f = None
        for key_ in ("inputs", "pairs", "features"):
            self._cache.pop(key_, None)

        # Selection pass over every configuration.
        m_sel = gam.mainstreaminess(
            m_raw_all[np.searchsorted(m_users, sel_users)], m_consts
        )
        t_sel = time.time()
        ranks, scored = self._selection_pass(
            terms,
            np.stack([r["tables"] for r in path]),
            sel_users,
            sel_picks,
            m_sel,
            work,
            key,
        )
        n_slots_all = int(self._id_slots()[0])
        sel_pass = {
            "rank_cut": RANK_CUT,
            "range_slots": SELECTION_CHUNK,
            "full_pass_share": float(np.mean(scored == n_slots_all)),
            "mean_scored_share": float(np.mean(scored) / n_slots_all),
            "seconds": time.time() - t_sel,
        }
        log.info(
            "p13a: selection pass: %.1f%% of the listeners needed the "
            "whole catalog, %.1f%% of all slots were scored",
            100 * sel_pass["full_pass_share"],
            100 * sel_pass["mean_scored_share"],
        )
        scores = ga.per_listener_scores(ranks)
        order = list(range(len(path)))  # strongest penalty first
        chosen, best, rows_se = ga.one_se_choice(scores, order)
        log.info(
            "p13a: selection scores %s. Best %d (lambda %.3g), chosen "
            "%d (lambda %.3g) by the one-standard-error rule",
            ", ".join(f"{r['mean']:.5f}" for r in rows_se),
            best,
            LAMBDAS[best],
            chosen,
            LAMBDAS[chosen],
        )

        # Two-run check and the analyses after the fit. The item side
        # stays cached for verify().
        inp = self._inputs()
        f = inp["f"]
        slot, w = encode_rows(terms, f, inp["rows"])
        m_row = m_user[inp["row_user"]]
        blocks = make_blocks(
            terms,
            layout,
            slot,
            w,
            m_row,
            inp["offset"],
            inp["lens"],
            inp["n_pos"],
            FIT_BLOCKS,
            inp["m_neg"],
        )
        _CTX.update(blocks=blocks, L=layout.L)
        try:
            with _Pool(FIT_WORKERS) as pool:
                cold = self._cold_fit(pool, layout, chosen, work, key, len(blocks))
                Tc_cold, _ = gam.center(layout.tables(cold["theta"]), layout, means_ref)
                two_run = float(np.max(np.abs(Tc_cold - path[chosen]["tables"])))
                T_ch = path[chosen]["tables"]
                D_ch = gam.diff_table(T_ch, layout.terms, layout.offsets)
                e_unif = gam.background_means(T_ch, layout, means_unif)
                e_list = [e_unif[t.name] for t in terms]
                stats = pool.map(
                    _stats_task, [(b, T_ch, D_ch, e_list) for b in range(len(blocks))]
                )
        finally:
            _CTX.clear()
        log.info(
            "p13a: two-run check, largest table difference %.2e " "(tolerance %.0e)",
            two_run,
            TWO_RUN_TOL,
        )
        importance = self._importance(terms, stats)
        e_ref = gam.background_means(T_ch, layout, means_ref)
        observed = ga.observed_concurvity(
            G,
            conc_groups,
            {
                t.name: T_ch[off : off + t.K + 2]
                for t, off in zip(terms, layout.offsets)
            },
        )

        doc = {
            "params": _params(),
            "pairs_sha256": inp["pairs_sha"],
            "features_sha256": inp["features_sha"],
            "upstream": self._upstream_outputs(),
            "code": self._code(),
            "rows": {
                "training_listeners": int(len(inp["tr"])),
                "training_rows": int(len(inp["rows"])),
                "positives": int((inp["kind"] == fp.KIND_POSITIVE).sum()),
                "m_neg": inp["m_neg"],
                "selection_listeners": int(len(sel_users)),
            },
            "mainstreaminess": m_consts,
            "knots": {k: list(v) for k, v in knots.items()},
            "shape_quantiles": shape_q,
            "terms": [t.to_json() for t in terms],
            "layout": {
                "L": layout.L,
                "P": layout.P,
                "labels": list(layout.labels),
                "neutral": list(neutral),
                "zero_free": layout.zero_free,
                "miss_free": layout.miss_free,
                "fingerprint": layout.fingerprint(),
                "n_zero": n_zero,
                "n_miss": n_miss,
            },
            "reference_means": {
                "p_pop": {k: v.tolist() for k, v in means_ref.items()},
                "uniform": {k: v.tolist() for k, v in means_unif.items()},
                "data": {k: v.tolist() for k, v in data_means.items()},
                "dev": {k: v.tolist() for k, v in dev_means.items()},
            },
            "audit": {
                "missing": missing,
                "tiers": tiers,
                "concurvity": {
                    "full": conc_full,
                    "names": conc_names,
                    "pairwise_worst": conc_pair.tolist(),
                    "observed": observed,
                },
            },
            "path": [
                {
                    "lambda": LAMBDAS[i],
                    "theta": r["theta"].tolist(),
                    "tables_raw": r["tables_raw"].tolist(),
                    "tables": r["tables"].tolist(),
                    "center_consts": r["center_consts"],
                    "fit": {
                        k: r[k]
                        for k in (
                            "objective",
                            "loss",
                            "smooth",
                            "ridge",
                            "proj_grad_max",
                            "proj_grad_before_newton",
                            "nit",
                            "nfev",
                            "restarts",
                            "newton_steps",
                            "method",
                            "message",
                            "success",
                            "seconds",
                        )
                    },
                    "selection": rows_se[i],
                }
                for i, r in enumerate(path)
            ],
            "choice": {
                "chosen": int(chosen),
                "best": int(best),
                "rule": "one standard error, strongest penalty " "first",
            },
            "selection_pass": sel_pass,
            "two_run": {
                "cold_fit": {
                    k: cold[k]
                    for k in (
                        "objective",
                        "proj_grad_max",
                        "proj_grad_before_newton",
                        "nit",
                        "nfev",
                        "newton_steps",
                        "method",
                        "message",
                        "seconds",
                    )
                },
                "max_table_diff": two_run,
            },
            "after_fit": {
                "background_means": {"p_pop": e_ref, "uniform": e_unif},
                "importance": importance,
            },
            "reference": self._als_reference(),
            "seconds": time.time() - t_start,
        }
        atomic_json_write(doc, outputs["model"])
        _atomic_savez(
            outputs["arrays"],
            sel_users=sel_users,
            sel_picks=sel_picks,
            ranks=ranks.astype(np.int32),
            scored=scored.astype(np.int64),
            m_users=m_users,
            m_raw=m_raw_all,
        )
        log.info("p13a: done in %s", _clock(time.time() - t_start))
        return self._info(outputs)

    def _info(self, outputs):
        return {
            "model": {
                "path": str(outputs["model"]),
                "sha256": file_sha256(outputs["model"]),
            },
            "arrays": {
                "path": str(outputs["arrays"]),
                "sha256": ev.npz_hash(outputs["arrays"]),
            },
        }

    # Fit steps

    def _gram(self, pool, n_blocks, L):
        prog = Progress(
            "p13a: Gram matrix for the concurvity audit",
            n_blocks,
            log.info,
            unit=" blocks",
        )
        G = np.zeros((L, L), dtype=np.float64)
        for g in pool.map(_gram_task, range(n_blocks)):
            G += g
            prog.advance(1)
        prog.finish()
        return 0.5 * (G + G.T)

    def _loss_grad(self, pool, n_blocks):
        def fn(T, D):
            res = pool.map(_fit_task, [(b, T, D) for b in range(n_blocks)])
            loss = 0.0
            grad = np.zeros_like(T)
            for l_b, g_b in res:
                loss += l_b
                grad += g_b
            return loss, grad

        return fn

    def _hess(self, pool, n_blocks):
        def fn(T, D):
            H = np.zeros((len(T), len(T)), dtype=np.float64)
            for h_b in pool.map(_hess_task, [(b, T, D) for b in range(n_blocks)]):
                H += h_b
            return H

        return fn

    def _fit_path(self, pool, layout, work, key, n_blocks):
        """The warm-started path, strongest penalty first. Each finished
        fit is saved, so a stopped run continues with the next weight."""
        state_path = work / "fits.json"
        k = {"key": _sha(key), "layout": layout.fingerprint()}
        state = load_json(state_path) or {}
        if state.get("k") != k:
            state = {"k": k, "fits": []}
        fits = state["fits"]
        theta = np.asarray(fits[-1]["theta"]) if fits else np.zeros(layout.P)
        if fits:
            log.info("p13a: path resume, %d of %d fits done", len(fits), len(LAMBDAS))
        loss_grad = self._loss_grad(pool, n_blocks)
        hess = self._hess(pool, n_blocks)
        for i in range(len(fits), len(LAMBDAS)):
            t0 = time.time()
            obj = gam.Objective(layout, loss_grad, LAMBDAS[i], RIDGE, hess)
            res = gam.fit(obj, theta, SOLVER)
            res["seconds"] = time.time() - t0
            theta = res["theta"]
            rec = {
                kk: (v.tolist() if isinstance(v, np.ndarray) else v)
                for kk, v in res.items()
            }
            fits.append(rec)
            atomic_json_write(state, state_path)
            left = (time.time() - t0) * (len(LAMBDAS) - i - 1)
            log.info(
                "p13a: fit %d of %d, lambda %.3g: objective %.8f, "
                "loss %.8f, %d iterations and %d Newton steps, largest "
                "projected gradient %.1e (%.1e before Newton), %s, %s. "
                "Path: about %s left",
                i + 1,
                len(LAMBDAS),
                LAMBDAS[i],
                res["objective"],
                res["loss"],
                res["nit"],
                res["newton_steps"],
                res["proj_grad_max"],
                res["proj_grad_before_newton"],
                res["message"],
                _clock(res["seconds"]),
                _clock(left),
            )
        return [dict(r, theta=np.asarray(r["theta"])) for r in fits]

    def _cold_fit(self, pool, layout, chosen, work, key, n_blocks):
        state_path = work / "cold_fit.json"
        k = {"key": _sha(key), "layout": layout.fingerprint(), "chosen": int(chosen)}
        state = load_json(state_path) or {}
        if state.get("k") == k:
            log.info("p13a: two-run check resume")
            return dict(state["fit"], theta=np.asarray(state["fit"]["theta"]))
        t0 = time.time()
        obj = gam.Objective(
            layout,
            self._loss_grad(pool, n_blocks),
            LAMBDAS[chosen],
            RIDGE,
            self._hess(pool, n_blocks),
        )
        # A random start, not zero: the first fit of the path already
        # started from zero, so a start there would repeat it bit for bit.
        start = np.random.default_rng(SEED + 3).normal(0.0, 0.5, layout.P)
        res = gam.fit(obj, start, SOLVER)
        res["seconds"] = time.time() - t0
        rec = {
            kk: (v.tolist() if isinstance(v, np.ndarray) else v)
            for kk, v in res.items()
        }
        atomic_json_write({"k": k, "fit": rec}, state_path)
        return res

    def _selection_pass(self, terms, tables, users, picks, m_sel, work, key):
        """(C, n, 2) ranks of every configuration, exact up to RANK_CUT and
        RANK_CUT + 1 beyond, and the slots scored per listener. Blocks of
        SELECTION_BLOCK listeners are saved in the order they finish, so a
        stopped run continues with the blocks still missing."""
        n_slots, _ = self._id_slots()
        u_indptr, u_indices, u_data = self._csr()
        n_blocks = max(1, -(-len(users) // SELECTION_BLOCK))
        tasks = _blocks(np.ones(len(users), dtype=np.int64), n_blocks)
        sel_dir = work / "selection"
        state_path = sel_dir / "state.json"
        k = {
            "key": _sha(key),
            "tables": _array_sha(tables),
            "users": _array_sha(users, picks, m_sel),
            "blocks": [list(t) for t in tasks],
            "rank_cut": RANK_CUT,
        }
        state = load_json(state_path) or {}
        if state.get("k") != k:
            for fp_ in sel_dir.glob("part_*.npz"):
                fp_.unlink()
            state = {"k": k, "parts": {}}
            atomic_json_write(state, state_path)

        def part(i):
            return sel_dir / f"part_{i:05d}.npz"

        done = {
            int(i)
            for i, sha in state["parts"].items()
            if part(int(i)).exists() and file_sha256(part(int(i))) == sha
        }
        pending = [i for i in range(len(tasks)) if i not in done]
        if done:
            log.info(
                "p13a: selection pass resume, %d of %d blocks done",
                len(done),
                len(tasks),
            )
        if pending:
            log.info(
                "p13a: loading the item side with the transposed "
                "co-listening lists (no progress lines in this step)"
            )
            t0 = time.time()
            src = self._sources(n_slots)
            space = self._space(src)
            log.info("p13a: item side ready in %s", _clock(time.time() - t0))
            t0 = time.time()
            scorer = ct.CatalogScorer(terms, tables, space, chunk=SELECTION_CHUNK)
            log.info(
                "p13a: catalog scorer ready for %d configurations in " "%s",
                scorer.C,
                _clock(time.time() - t0),
            )
            _CTX.update(
                scorer=scorer,
                space=space,
                sel_users=users,
                sel_picks=picks,
                sel_m=m_sel,
                u_indptr=u_indptr,
                u_indices=u_indices,
                u_data=u_data,
            )
            try:
                workers = self._selection_workers(
                    space,
                    users[np.concatenate([np.arange(*tasks[i]) for i in pending])],
                    u_indptr,
                    u_indices,
                    n_slots,
                )
                probe = self._probe(tasks, pending, workers, n_slots)
                n_left = sum(tasks[i][1] - tasks[i][0] for i in pending)
                prog = Progress(
                    f"p13a: selection pass, {len(tables)} "
                    f"configurations, {workers} workers",
                    n_left,
                    log.info,
                    unit=" listeners",
                )
                mem0 = _memory()
                last_mem, warned = time.time(), False
                with _Pool(workers) as pool:
                    for j, (out, scored) in pool.unordered(
                        _selection_task, [tasks[i] for i in pending]
                    ):
                        i = pending[j]
                        _atomic_savez(part(i), ranks=out, scored=scored)
                        state["parts"][str(i)] = file_sha256(part(i))
                        atomic_json_write(state, state_path)
                        prog.advance(tasks[i][1] - tasks[i][0])
                        if mem0 is not None and (
                            time.time() - last_mem >= MEMORY_LOG_SECONDS
                        ):
                            last_mem = time.time()
                            avail, swap = _memory()
                            log.info(
                                "p13a: memory: %.1f GB available, "
                                "%.1f GB of swap in use",
                                _gb(avail),
                                _gb(swap),
                            )
                            if swap - mem0[1] > (1 << 30) and not warned:
                                warned = True
                                log.warning(
                                    "p13a: swap use grew by %.1f GB during "
                                    "the selection pass, which slows it "
                                    "down. Fewer workers help: stop with "
                                    "Ctrl+C and start again with, for "
                                    "example, P13A_WORKERS=8 python3 -m "
                                    "pipeline.runner --only p13a_fit. The "
                                    "saved blocks and fits stay.",
                                    _gb(swap - mem0[1]),
                                )
                prog.finish()
                self._check_probe(probe, tasks, part)
            finally:
                _CTX.clear()
        ranks = np.concatenate(
            [np.load(part(i))["ranks"] for i in range(len(tasks))], axis=1
        )
        scored = np.concatenate([np.load(part(i))["scored"] for i in range(len(tasks))])
        check(
            ranks.shape == (len(tables), len(users), 2)
            and scored.shape == (len(users),),
            "the selection blocks do not cover the listeners",
        )
        check(
            int(ranks.min()) >= 1
            and int(ranks.max()) <= RANK_CUT + 1
            and int(scored.min()) >= 1
            and int(scored.max()) <= n_slots,
            "a selection rank or slot count lies outside its range",
        )
        return ranks, scored

    def _selection_workers(self, space, users, u_indptr, u_indices, n_slots):
        """SELECTION_WORKERS, or fewer when the free memory cannot hold one
        listener's working set per worker. The largest working set is the
        listener whose history has the most entries in a transposed
        co-listening list: ranks_cut() gathers those rows (4 + 8 bytes per
        entry) and keeps two numerators over the catalog."""
        indeg = [np.diff(m.indptr) for m in space.colisten_t.values()]
        worst = 0
        for u in users:
            s, e = int(u_indptr[u]), int(u_indptr[u + 1])
            items = np.asarray(u_indices[s:e], dtype=np.int64)
            for d in indeg:
                worst = max(worst, int(d[items].sum()))
        per = 12 * worst + 16 * n_slots + 400 * SELECTION_CHUNK + (64 << 20)
        mem = _memory()
        if mem is None:
            log.info(
                "p13a: psutil is missing, so the selection pass uses "
                "%d workers without a memory check",
                SELECTION_WORKERS,
            )
            return SELECTION_WORKERS
        fit = max(1, int((mem[0] - MEMORY_RESERVE) // per))
        workers = max(1, min(SELECTION_WORKERS, fit))
        log.info(
            "p13a: selection memory: up to %.2f GB per worker (the "
            "largest co-listening gather holds %s entries), %.1f GB "
            "available, %.1f GB of swap in use. Workers: %d",
            _gb(per),
            f"{worst:,}",
            _gb(mem[0]),
            _gb(mem[1]),
            workers,
        )
        return workers

    def _probe(self, tasks, pending, workers, n_slots):
        """Times PROBE_USERS listeners in this process before the pool
        starts, logs an estimate of the pass, and returns their results
        for a check against the pool's."""
        take = np.unique(
            np.round(
                np.linspace(0, len(pending) - 1, min(PROBE_USERS, len(pending)))
            ).astype(np.int64)
        )
        ks = [int(tasks[pending[int(i)]][0]) for i in take]
        out, secs = {}, []
        with threadpool_limits(1):
            for n, k in enumerate(ks, start=1):
                t0 = time.time()
                r, s = _selection_one(_CTX, k)
                secs.append(time.time() - t0)
                out[k] = (r.copy(), int(s))
                log.info(
                    "p13a: probe listener %d of %d: %.2f s, %.0f%% of "
                    "the catalog scored",
                    n,
                    len(ks),
                    secs[-1],
                    100.0 * s / n_slots,
                )
        n_left = sum(tasks[i][1] - tasks[i][0] for i in pending)
        par = min(float(workers), 1.25 * _physical_cores())
        est = float(np.mean(secs)) * n_left / par
        log.info(
            "p13a: probe mean %.2f s per listener in one process. "
            "Rough estimate for %s listeners on %d workers: %s. The "
            "progress lines below measure the real rate",
            np.mean(secs),
            f"{n_left:,}",
            workers,
            _clock(est),
        )
        return out

    def _check_probe(self, probe, tasks, part):
        """The pool must give the probe listeners the ranks and slot counts
        that this process gave them."""
        starts = np.array([t[0] for t in tasks])
        for k, (r, s) in probe.items():
            i = int(np.searchsorted(starts, k, side="right") - 1)
            z = np.load(part(i))
            j = k - tasks[i][0]
            check(
                np.array_equal(z["ranks"][:, j, :], r) and int(z["scored"][j]) == s,
                f"selection listener {k}: the pool's ranks differ from "
                f"the probe's in this process",
            )

    def _importance(self, terms, stats):
        abs_pos = sum(s["abs_pos"] for s in stats)
        n_pos = sum(s["n_pos"] for s in stats)
        flips = sum(s["flips"] for s in stats)
        sd_sum = sum(s["sd_sum"] for s in stats)
        n_users = sum(s["n_users"] for s in stats)
        cov = sum(s["cov"] for s in stats)
        var = sum(s["var"] for s in stats)
        return {
            t.name: {
                "mean_abs_positive": float(abs_pos[k] / n_pos),
                "sign_flip_share": float(flips[k] / n_pos),
                "within_listener_sd": float(sd_sum[k] / n_users),
                "cov_share": float(cov[k] / var) if var > 0 else math.nan,
            }
            for k, t in enumerate(terms)
        }

    # Verify

    def _load(self):
        doc = load_json(self.outputs()["model"])
        check(doc is not None, "p13a_model.json is missing or unreadable")
        z = np.load(self.outputs()["arrays"])
        return doc, {k: z[k] for k in z.files}

    def verify(self, params):
        self._verified = None
        gaps = {}
        doc, arr = self._load()
        check(
            doc["params"] == _params(),
            f"p13a_model.json was built with {doc['params']}, the stage "
            f"now says {_params()}",
        )
        inp = self._inputs()
        check(
            doc["pairs_sha256"] == inp["pairs_sha"]
            and doc["features_sha256"] == inp["features_sha"],
            "the model was fit on other pairs or features",
        )
        f = inp["f"]
        knots = compute_knots(f, inp["rows"], inp["kind"])
        check(
            {k: list(v) for k, v in knots.items()} == doc["knots"],
            "the stored knots differ from the knot rule on the inputs",
        )
        terms = base_terms(knots)
        check(
            [t.to_json() for t in terms] == doc["terms"],
            "the stored terms differ from the base configuration",
        )

        # The audit: point AUCs and the neutral decisions.
        missing = doc["audit"]["missing"]
        rows, kind, row_user = inp["rows"], inp["kind"], inp["row_user"]
        U = len(inp["lens"])
        pop, _ = concept_column(f, rows, "popularity")
        pos = kind == fp.KIND_POSITIVE
        cuts = np.quantile(pop[pos], np.arange(1, N_STRATA) / N_STRATA)
        check(
            np.allclose(cuts, missing["cuts"], rtol=0, atol=0),
            "the popularity decile cuts differ",
        )
        stratum = ga.strata_of(pop, cuts)
        worst_auc = 0.0
        for c in AUDIT_CONCEPTS:
            _, has = concept_column(f, rows, c)
            cp = ga.flag_counts(row_user[pos], stratum[pos], has[pos], U, N_STRATA).sum(
                axis=0
            )
            for kd in (fp.KIND_UNIFORM, fp.KIND_POP):
                m = kind == kd
                cn_ = ga.flag_counts(row_user[m], stratum[m], has[m], U, N_STRATA).sum(
                    axis=0
                )
                a = float(ga.pooled_auc(cp, cn_))
                stored = missing["concepts"][c]["by_kind"][fp.KIND_NAMES[kd]]
                worst_auc = max(worst_auc, abs(a - stored["auc"]))
                check(
                    stored["lo"] <= stored["auc"] <= stored["hi"]
                    or not np.isfinite(stored["auc"]),
                    f"{c}: the stored AUC lies outside its interval",
                )
            check(
                bool(missing["concepts"][c]["no_signal"])
                == bool(
                    ga.no_signal(missing["concepts"][c]["by_kind"], NEUTRAL_MARGIN)
                ),
                f"{c}: the neutral decision does not follow the rule",
            )
        check(
            worst_auc <= 1e-12,
            f"an audit AUC differs from its " f"recomputation by {worst_auc:.2e}",
        )
        gaps["audit_auc"] = worst_auc
        neutral = tuple(c for c in BASE_CONCEPTS if missing["concepts"][c]["no_signal"])
        check(
            list(neutral) == doc["layout"]["neutral"],
            "the neutral missing states differ from the audit",
        )

        # The layout, the stationary points and the centering.
        m_consts = doc["mainstreaminess"]
        m_raw = arr["m_raw"]
        m_users = arr["m_users"]
        tr_users = inp["p"]["user_id"][inp["tr"]]
        raw_tr = m_raw[np.searchsorted(m_users, tr_users)]
        check(
            np.array_equal(m_users[np.searchsorted(m_users, tr_users)], tr_users),
            "a training listener has no " "mainstreaminess",
        )
        check(
            abs(float(raw_tr.mean()) - m_consts["mean"])
            <= 1e-12 * max(1.0, abs(m_consts["mean"]))
            and abs(float(raw_tr.std()) - m_consts["sd"])
            <= 1e-12 * max(1.0, m_consts["sd"]),
            "the mainstreaminess constants differ from the stored values",
        )
        worst_m = self._verify_mainstream(m_users, m_raw)
        gaps["mainstream"] = worst_m
        m_user = gam.mainstreaminess(raw_tr, m_consts)
        slot, w = encode_rows(terms, f, rows)
        ref_mask = kind == fp.KIND_POP
        means_ref, data_means, dev_means = reference_means(terms, slot, w, ref_mask)
        n_zero = {
            t.name: int((slot[k] == t.K + gam.ZERO).sum()) for k, t in enumerate(terms)
        }
        n_miss = {
            t.name: int((slot[k] == t.K + gam.MISS).sum()) for k, t in enumerate(terms)
        }
        layout = gam.Layout(terms, n_zero, n_miss, neutral, data_means, dev_means)
        check(
            layout.fingerprint() == doc["layout"]["fingerprint"],
            "the parameter layout differs from the stored one",
        )
        blocks = make_blocks(
            terms,
            layout,
            slot,
            w,
            m_user[row_user],
            inp["offset"],
            inp["lens"],
            inp["n_pos"],
            FIT_BLOCKS,
            inp["m_neg"],
        )
        worst_pg, worst_obj, worst_center = 0.0, 0.0, 0.0
        _CTX.update(blocks=blocks, L=layout.L)
        try:
            with _Pool(FIT_WORKERS) as pool:
                lg = self._loss_grad(pool, len(blocks))
                for i, rec in enumerate(doc["path"]):
                    theta = np.asarray(rec["theta"], dtype=np.float64)
                    obj = gam.Objective(layout, lg, rec["lambda"], RIDGE)
                    loss, smooth, ridge, grad = obj.parts(theta)
                    F = loss + obj.lam_smooth * smooth + RIDGE * ridge
                    rel = abs(F - rec["fit"]["objective"]) / max(1.0, abs(F))
                    pg = float(
                        np.max(
                            np.abs(gam.projected_gradient(theta, grad, layout.bounds))
                        )
                    )
                    worst_obj = max(worst_obj, rel)
                    worst_pg = max(worst_pg, pg)
                    check(
                        rel <= OBJ_RTOL,
                        f"configuration {i}: the "
                        f"objective differs from its "
                        f"recomputation by {rel:.2e}",
                    )
                    check(
                        pg <= KKT_TOL,
                        f"configuration {i}: the largest "
                        f"projected gradient is {pg:.2e}, "
                        f"above {KKT_TOL:.0e}",
                    )
                    T = layout.tables(theta)
                    check(
                        np.array_equal(T, np.asarray(rec["tables_raw"])),
                        f"configuration {i}: the raw tables differ from " f"M theta",
                    )
                    Tc = np.asarray(rec["tables"])
                    means = gam.background_means(Tc, layout, means_ref)
                    for t in terms:
                        if not t.centered:
                            worst_center = max(worst_center, abs(means[t.name]))
                        if t.missing_state and t.name in neutral:
                            sl = layout.term_slice(t.name)
                            a = layout.data_means[t.name]
                            tt = Tc[sl]
                            gap = abs(tt[t.K + gam.MISS] - a @ tt[: t.K + 1])
                            check(
                                gap <= 1e-10,
                                f"{t.name}: the missing "
                                f"constant is not at the "
                                f"neutral level",
                            )
        finally:
            _CTX.clear()
        check(
            worst_center <= 1e-10,
            f"a centered table has mean " f"{worst_center:.2e} over the " f"background",
        )
        gaps.update(kkt=worst_pg, objective=worst_obj, center=worst_center)

        # Training rows: the encoded scores against np.interp.
        chosen = doc["choice"]["chosen"]
        Tc = np.asarray(doc["path"][chosen]["tables"])
        gaps["rows"] = self._verify_rows(
            terms, layout, Tc, blocks, f, rows, m_user[row_user]
        )
        check(
            gaps["rows"] <= SCORE_TOL,
            f"a training row's score differs "
            f"from np.interp by "
            f"{gaps['rows']:.2e}",
        )
        del blocks, slot, w
        # The choice and the two-run check.
        ranks = arr["ranks"]
        c2, b2, _ = ga.one_se_choice(
            ga.per_listener_scores(ranks), list(range(len(doc["path"])))
        )
        check(
            c2 == chosen and b2 == doc["choice"]["best"],
            "the choice does not follow from the stored ranks",
        )
        check(
            doc["two_run"]["max_table_diff"] <= TWO_RUN_TOL,
            f"the two runs differ by {doc['two_run']['max_table_diff']:.2e}"
            f", above {TWO_RUN_TOL:.0e}",
        )
        # Selection ranks against the independent dense reference.
        gaps.update(self._verify_selection(doc, arr, terms))
        self._verified = gaps
        log.info(
            "p13a: verify passed. Largest gaps: %s",
            ", ".join(f"{k} {v:.1e}" for k, v in gaps.items()),
        )

    def _verify_mainstream(self, m_users, m_raw):
        n_slots, _ = self._id_slots()
        u_indptr, u_indices, u_data = self._csr()
        popularity = self._popularity(n_slots)
        rng = np.random.default_rng(SEED)
        pick = rng.choice(
            len(m_users), size=min(VERIFY_MAINSTREAM, len(m_users)), replace=False
        )
        worst = 0.0
        for j in np.sort(pick):
            items, weights = cd.history_of(u_indptr, u_indices, u_data, int(m_users[j]))
            ref = math.fsum(
                float(a) * float(popularity[i]) for i, a in zip(items, weights)
            ) / math.fsum(float(a) for a in weights)
            worst = max(worst, abs(ref - float(m_raw[j])))
        check(
            worst <= 1e-9,
            f"a stored mainstreaminess differs from a " f"recomputation by {worst:.2e}",
        )
        return worst

    def _verify_rows(self, terms, layout, T, blocks, f, rows, m_rows):
        """Scores of sampled training rows: the encoded path against
        np.interp on the raw P12 values."""
        rng = np.random.default_rng(SEED + 1)
        n = len(rows)
        take = np.sort(rng.choice(n, size=min(VERIFY_ROWS, n), replace=False))
        D = gam.diff_table(T, layout.terms, layout.offsets)
        fast = np.empty(len(take))
        starts = np.cumsum([0] + [blk["w"].shape[1] for blk in blocks])
        for b, blk in enumerate(blocks):
            lo, hi = np.searchsorted(take, (starts[b], starts[b + 1]))
            if hi > lo:
                fast[lo:hi] = gam.block_scores(T, D, blk)[take[lo:hi] - starts[b]]
        tier = np.asarray(f["match_tier"][rows[take]], dtype=np.int64)
        full = {
            c: concept_column(f, rows[take], c) for c in set(t.source for t in terms)
        }
        m_take = m_rows[take]
        ref = np.zeros(len(take))
        for t, off in zip(terms, layout.offsets):
            v, h = full[t.source]
            tab = T[off : off + t.size]
            kn = t.knot_array
            val = np.interp(
                np.clip(np.where(h, v, kn[0]), kn[0], kn[-1]), kn, tab[: t.K]
            )
            if t.zero_state:
                val = np.where(h & (v == 0.0), tab[t.K + gam.ZERO], val)
            val = np.where(h, val, tab[t.K + gam.MISS] if t.missing_state else 0.0)
            if t.tiers is not None:
                val = np.where(np.isin(tier, t.tiers), val, 0.0)
            if t.multiplier is not None:
                val = m_take * val
            ref += val
        return float(np.max(np.abs(fast - ref)))

    def _verify_selection(self, doc, arr, terms):
        n_slots, _ = self._id_slots()
        src = self._sources(n_slots)
        space = self._space(src)
        u_indptr, u_indices, u_data = self._csr()
        users, picks = arr["sel_users"], arr["sel_picks"]
        ranks = arr["ranks"]
        scored = arr["scored"]
        cut1 = RANK_CUT + 1
        check(
            scored.shape == (len(users),)
            and int(scored.min()) >= 1
            and int(scored.max()) <= n_slots,
            "the slot counts of the selection pass are malformed",
        )
        check(
            int(ranks.min()) >= 1 and int(ranks.max()) <= cut1,
            f"a stored rank lies outside 1 to {cut1}",
        )
        chosen = doc["choice"]["chosen"]
        cfg = sorted({0, chosen, len(doc["path"]) - 1})
        tables = np.stack([np.asarray(doc["path"][c]["tables"]) for c in cfg])
        scorer = ct.CatalogScorer(terms, tables, space)
        # The pass as run: every configuration, the same slot ranges.
        scorer_all = ct.CatalogScorer(
            terms,
            np.stack([np.asarray(r["tables"]) for r in doc["path"]]),
            space,
            chunk=SELECTION_CHUNK,
        )
        m_consts = doc["mainstreaminess"]
        m_sel = gam.mainstreaminess(
            arr["m_raw"][np.searchsorted(arr["m_users"], users)], m_consts
        )
        rng = np.random.default_rng(SEED + 2)
        take = np.sort(
            rng.choice(len(users), size=min(VERIFY_USERS, len(users)), replace=False)
        )
        worst, near = 0.0, 0
        prog = Progress(
            "p13a verify: selection ranks against the dense " "reference",
            len(take),
            log.info,
            unit=" listeners",
        )
        for j in take:
            u = int(users[j])
            items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
            state = cn.build_user_state(items, weights, space)
            full = cn.full_catalog(state, space)
            ref = ct.reference_scores(full, terms, tables, float(m_sel[j]), space.tier)
            del full
            lst = scorer.listener(state, space, float(m_sel[j]))
            fast = scorer.scores(lst)
            ok = np.ones(n_slots, dtype=bool)
            ok[0] = False
            worst = max(worst, float(np.max(np.abs(fast[:, ok] - ref[:, ok]))))
            full_r = scorer.ranks(lst, picks[j], items)
            cut_r, _ = scorer.ranks_cut(
                state, space, float(m_sel[j]), picks[j], items, RANK_CUT
            )
            for ci, c in enumerate(cfg):
                for p in range(2):
                    lo, hi = ct.reference_rank(ref[ci], picks[j, p], items, SCORE_TOL)
                    near += int(hi > lo)
                    got = int(ranks[c, j, p])
                    check(
                        min(lo, cut1) <= got <= min(hi, cut1),
                        f"listener {u}, configuration {c}, pick {p}: the "
                        f"stored rank {got} is outside the reference "
                        f"range [{lo}, {hi}] censored at {cut1}",
                    )
                    check(
                        got == min(int(full_r[ci, p]), cut1),
                        f"listener {u}, configuration {c}, pick {p}: the "
                        f"stored rank {got} differs from the full-catalog "
                        f"count {int(full_r[ci, p])} censored at {cut1}",
                    )
                    check(
                        got == int(cut_r[ci, p]),
                        f"listener {u}, configuration {c}, pick {p}: a new "
                        f"run of ranks_cut() gives {int(cut_r[ci, p])}, "
                        f"not the stored {got}",
                    )
            all_r, all_sc = scorer_all.ranks_cut(
                state, space, float(m_sel[j]), picks[j], items, RANK_CUT
            )
            check(
                np.array_equal(all_r, ranks[:, j, :]) and int(all_sc) == int(scored[j]),
                f"listener {u}: a new run of the selection pass gives "
                f"other ranks or another slot count ({int(all_sc)} "
                f"against the stored {int(scored[j])})",
            )
            prog.advance(1)
        prog.finish()
        check(
            worst <= SCORE_TOL,
            f"the catalog scorer differs from the " f"dense reference by {worst:.2e}",
        )
        return {"catalog_scores": worst, "near_ties": float(near)}

    def report(self, params):
        doc, arr = self._load()
        lines = []
        add = lines.append
        add("# P13a report: the transparent model, first fit")
        add("")
        add(f"- parameters: {json.dumps(doc['params'])}")
        r = doc["rows"]
        add(
            f"- training rows: {r['training_rows']:,} of "
            f"{r['training_listeners']:,} train-split listeners "
            f"({r['positives']:,} positives, {r['m_neg']} negatives per "
            f"listener). Selection listeners: "
            f"{r['selection_listeners']:,}."
        )
        add(
            f"- inputs: fusion_pairs.npz {doc['pairs_sha256'][:16]}..., "
            f"fusion_features.npz {doc['features_sha256'][:16]}..."
        )
        lay = doc["layout"]
        add(
            f"- model: {len(doc['terms'])} terms, {lay['L']} table entries, "
            f"{lay['P']} free parameters."
        )
        mc = doc["mainstreaminess"]
        add(
            f"- mainstreaminess: raw mean {mc['mean']:.4f}, SD "
            f"{mc['sd']:.4f}. Standardized values are clamped to "
            f"[{mc['lo']:.2f}, {mc['hi']:.2f}], the 0.5 and 99.5 "
            f"percentiles of the training listeners."
        )
        if self._verified:
            add(
                "- verify(): "
                + ", ".join(f"{k} {v:.1e}" for k, v in self._verified.items())
                + ". kkt is the largest projected gradient over all "
                "configurations, rows and catalog_scores the largest "
                "score gaps against the independent evaluations, "
                "near_ties the checked ranks with a score within the "
                "tolerance of the pick's."
            )
        add(f"- run time: {_clock(doc['seconds'])}.")
        add("")
        self._report_knots(doc, add)
        self._report_audit(doc, add)
        self._report_path(doc, arr, add)
        self._report_model(doc, add)
        self._report_semantics(doc, add)
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p13a report written")

    def _report_knots(self, doc, add):
        add("## 1. Knots")
        add("")
        add(
            "Each shape is linear between its knots and constant beyond the "
            "outer ones. Knots sit at 30 quantiles of the positives' values, "
            "9 quantiles of the negatives' values and 0, so most segments "
            "cover the range where the positives lie."
        )
        add("")
        add("| concept | knots | lowest | highest |")
        add("|---|---:|---:|---:|")
        for k, v in doc["knots"].items():
            add(f"| {k} | {len(v)} | {v[0]:.4g} | {v[-1]:.4g} |")
        add("")

    def _report_audit(self, doc, add):
        a = doc["audit"]
        add("## 2. Audit before the fit")
        add("")
        add("### 2a. Missing states within popularity deciles")
        add("")
        add(
            'AUC of the "has data" flag between positives and negatives of '
            "one kind, within deciles of the positives' popularity, pooled "
            "over the deciles with the pair counts as weights. 0.5: the flag "
            "separates no better than popularity alone does. In brackets: "
            "the 95 percent interval of a listener cluster bootstrap with "
            f"{N_BOOT} replicates. A flag whose two intervals lie inside "
            f"0.5 +- {NEUTRAL_MARGIN} carries no signal beyond popularity, "
            "and the model ties its missing constant to the neutral level. "
            "work_affinity is out of the score and appears for the record."
        )
        add("")
        add(
            "| concept | with data: positives / uniform / p_pop | AUC vs "
            "uniform | AUC vs p_pop | missing constant |"
        )
        add("|---|---|---|---|---|")
        for c, e in a["missing"]["concepts"].items():
            s = e["share_with_data"]
            u, pp = e["by_kind"]["uniform"], e["by_kind"]["p_pop"]
            dec = "tied to neutral" if e["no_signal"] else "learned"
            if c not in BASE_CONCEPTS:
                dec = "not in the model"
            add(
                f"| {c} | {s['positives']:.1%} / {s['uniform']:.1%} / "
                f"{s['p_pop']:.1%} | {u['auc']:.3f} [{u['lo']:.3f}, "
                f"{u['hi']:.3f}] | {pp['auc']:.3f} [{pp['lo']:.3f}, "
                f"{pp['hi']:.3f}] | {dec} |"
            )
        add("")
        add("### 2b. Tier support")
        add("")
        add(
            "Rows with audio per bridge tier. The deviation curve of audio "
            "fit applies to fuzzy_dur only. The map tiers (isrc, isrc+url, "
            "url) share the reference curve."
        )
        add("")
        add(
            "| tier | positives: rows, with audio | uniform: rows, with "
            "audio | p_pop: rows, with audio | audio fit of positives: p1 / "
            "p50 / p99 |"
        )
        add("|---|---|---|---|---|")
        for row in a["tiers"]:
            cells = []
            for g in ("positives", "uniform", "p_pop"):
                x = row[g]
                share = (x["with_audio"] / x["rows"]) if x["rows"] else 0.0
                cells.append(f"{x['rows']:,}, {share:.1%}")
            q = row["positives"]["audio_fit_q"]
            qs = "n/a" if q is None else " / ".join(f"{v:.2f}" for v in q)
            add(f"| {row['tier']} | {' | '.join(cells)} | {qs} |")
        add("")
        add("### 2c. Concurvity")
        add("")
        add(
            "mgcv-style indices on the design after each listener's mean "
            "row is subtracted (a per-listener softmax sees only these "
            "differences). worst: the largest squared canonical correlation "
            "between a term's basis and the bases of all other terms. "
            "estimate: the mean squared canonical correlation. observed: "
            "the share of the fitted term that the other terms' bases can "
            "reproduce, for the chosen configuration. 0 means no "
            "concurvity, 1 a term that the others can fully replace."
        )
        add("")
        cc = a["concurvity"]
        add("| term | worst | estimate | observed |")
        add("|---|---:|---:|---:|")
        for name in cc["names"]:
            fl = cc["full"][name]
            add(
                f"| {name} | {fl['worst']:.3f} | {fl['estimate']:.3f} | "
                f"{cc['observed'].get(name, float('nan')):.3f} |"
            )
        add("")
        names = cc["names"]
        pair = np.asarray(cc["pairwise_worst"])
        top = sorted(
            (
                (pair[i, j], names[i], names[j])
                for i in range(len(names))
                for j in range(i + 1, len(names))
            ),
            reverse=True,
        )[:6]
        add(
            "Largest pairwise worst indices: "
            + ", ".join(f"{a_} with {b_} {v:.3f}" for v, a_, b_ in top)
            + "."
        )
        add("")

    def _report_path(self, doc, arr, add):
        add("## 3. Penalty path and selection")
        add("")
        add(
            "Each row is one fit of the path, strongest penalty first, "
            "warm-started from the row above. Selection score: the mean "
            "NDCG@100 over the val_last and val_first picks of the "
            "selection listeners, full catalog, P5 rules. Difference: the "
            "best score minus this row's score, with the standard error of "
            "the paired per-listener differences."
        )
        add("")
        add(
            "| lambda | iterations + Newton steps | largest projected "
            "gradient | objective | selection score | val_last NDCG@100 | "
            "val_first NDCG@100 | val_last R@100 | difference to best | "
            "choice |"
        )
        add("|---:|---:|---:|---:|---:|---:|---:|---:|---|---|")
        ranks = arr["ranks"]
        ch = doc["choice"]
        for i, rec in enumerate(doc["path"]):
            last = ev.metric_block(ranks[i, :, 0])
            first = ev.metric_block(ranks[i, :, 1])
            s = rec["selection"]
            mark = []
            if i == ch["best"]:
                mark.append("best")
            if i == ch["chosen"]:
                mark.append("chosen")
            add(
                f"| {rec['lambda']:.3g} | {rec['fit']['nit']} + "
                f"{rec['fit']['newton_steps']} | "
                f"{rec['fit']['proj_grad_max']:.1e} | "
                f"{rec['fit']['objective']:.6f} | {s['mean']:.5f} | "
                f"{last['ndcg_at_100']:.5f} | {first['ndcg_at_100']:.5f} | "
                f"{last['recall_at_100']:.4f} | {s['diff']:.5f} +- "
                f"{s['se']:.5f} | {', '.join(mark)} |"
            )
        add("")
        sp_ = doc["selection_pass"]
        add(
            f"- Ranks are exact up to {sp_['rank_cut']} and stored as "
            f"{sp_['rank_cut'] + 1} beyond, which is all that NDCG@100 and "
            f"R@100 read. The pass scored "
            f"{100 * sp_['mean_scored_share']:.1f} percent of all slots: "
            f"{100 * sp_['full_pass_share']:.1f} percent of the listeners "
            f"needed the whole catalog, and the others stopped once both "
            f"picks ranked below {sp_['rank_cut']} in every configuration "
            f"(catalog.ranks_cut(), ranges of {sp_['range_slots']:,} "
            f"slots)."
        )
        ref = doc.get("reference") or {}
        chosen = doc["path"][ch["chosen"]]["selection"]["mean"]
        if "popularity" in ref:
            pop_s = ref["popularity"]
            als_s = ref.get("als", float("nan"))
            add(
                f"- On the same picks, popularity scores {pop_s:.5f} and "
                f"the tuned ALS ({ref.get('als_label', '')}) {als_s:.5f} "
                f"(als_tuning.json). The chosen GAM scores {chosen:.5f}."
            )
        tr = doc["two_run"]
        add(
            f"- Two-run check: the chosen weight fit again from a random "
            f"start reached "
            f"tables within {tr['max_table_diff']:.1e} of the warm-started "
            f"fit ({tr['cold_fit']['nit']} iterations, "
            f"{tr['cold_fit']['message']}). Tolerance: {TWO_RUN_TOL:.0e}."
        )
        add(
            f"- Solver: damped Newton steps with the exact Hessian in "
            f"float64, until the largest gradient component falls below "
            f"{SOLVER['newton_target']:.0e}. SciPy's L-BFGS-B takes over "
            f"only if Newton stops above {SOLVER['newton_accept']:.0e}. "
            f"Settings: {json.dumps(SOLVER)}, ridge {RIDGE:g}."
        )
        add("")

    def _report_model(self, doc, add):
        ch = doc["choice"]["chosen"]
        rec = doc["path"][ch]
        T = np.asarray(rec["tables"])
        terms = [gam.Term.from_json(t) for t in doc["terms"]]
        offsets = np.cumsum([0] + [t.size for t in terms])[:-1]
        lay = doc["layout"]
        add(f"## 4. The chosen model (lambda {rec['lambda']:.3g})")
        add("")
        add(
            "Values in log-odds points, centered on the p_pop negatives: a "
            "value of 0 is what an average popularity-weighted candidate "
            "gets from the term. The shape columns give the term's value at "
            "percentiles of the positives' values. zero: the value when the "
            'data exists and shows no match. missing: the value for "no '
            'data", learned, or tied to the neutral level by the audit, or '
            "(without training rows in the state) tied by continuity."
        )
        add("")
        add(
            "| term | "
            + " | ".join(f"p{int(q * 100)}" for q in SHAPE_LEVELS)
            + " | zero | missing |"
        )
        add("|---|" + "---:|" * (len(SHAPE_LEVELS) + 2))
        for t, off in zip(terms, offsets):
            tab = T[off : off + t.size]
            kn = t.knot_array
            qx = doc["shape_quantiles"].get(t.name)
            vals = (
                np.interp(np.clip(qx, kn[0], kn[-1]), kn, tab[: t.K])
                if qx is not None
                else [math.nan] * len(SHAPE_LEVELS)
            )
            zero = f"{tab[t.K]:+.3f}" if t.zero_state else ""
            if t.zero_state and not lay["zero_free"].get(t.name, False):
                zero += " (continuity)"
            miss = ""
            if t.missing_state:
                miss = f"{tab[t.K + 1]:+.3f}"
                if t.name in lay["neutral"]:
                    miss += " (neutral)"
                elif not lay["miss_free"].get(t.name, False):
                    miss += " (tied)"
            add(
                f"| {t.name} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + f" | {zero} | {miss} |"
            )
        add("")
        add(
            "popularity_tilt is f_1: a listener with mainstreaminess m_u "
            "gets m_u times this value. audio_fit_fuzzy_dur is the change "
            "of the audio fit curve for fuzzy_dur tracks, with mean 0 over "
            "those tracks."
        )
        add("")

    def _report_semantics(self, doc, add):
        af = doc["after_fit"]
        imp = af["importance"]
        bm = af["background_means"]
        add("## 5. Explanation semantics")
        add("")
        add(
            "A contribution is the interventional Shapley value relative to "
            "a declared background D: f_j(x_j) - E_D f_j(X_j) (Bordt & von "
            "Luxburg 2023, Theorem 8(b), and Hiabu, Meyer & Wright 2023, "
            "Corollary 2.7). The tables are centered on the p_pop negatives. "
            "The shift to the uniform background moves each term by one "
            "constant. Sign flip: the share of positive rows whose "
            "contribution changes sign between the two backgrounds."
        )
        add("")
        add(
            "| term | mean over p_pop | mean over uniform | shift | sign "
            "flips | mean abs. contribution, positives | SD within a "
            "listener's candidates | covariance share |"
        )
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for name, e in imp.items():
            a, b = bm["p_pop"][name], bm["uniform"][name]
            add(
                f"| {name} | {a:+.4f} | {b:+.4f} | {a - b:+.4f} | "
                f"{e['sign_flip_share']:.1%} | {e['mean_abs_positive']:.3f} "
                f"| {e['within_listener_sd']:.3f} | {e['cov_share']:+.3f} |"
            )
        add("")
        add(
            "The SD within a listener's candidates measures what moves the "
            "ranking, because only differences between the candidates of "
            "one listener matter. The covariance shares Cov(f_j, F) / Var(F) "
            "within listeners add to 1 over the terms. For "
            "popularity_tilt the means are those of f_1, and each "
            "listener's shift is m_u times the listed shift."
        )
        add("")
