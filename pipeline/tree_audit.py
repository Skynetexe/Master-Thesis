"""The tree-GAM audit (protocol declared before any code or result).

A tree-based GAM by the EBM method (Lou, Caruana & Gehrke 2012, Nori et al.
2019), fit on the rows, terms, states and loss of the free fit. It is
compared with the free fit (gam_refit's model) and with the final model
(p13c_final.json). Chang et al. (KDD 2021) found that equally accurate GAM
algorithms learn different shapes, and they recommend tree-based GAMs as
the check. The audit changes no model and no reason rule.

Data and loss: the model's level of the wide file (64 on the real data).
Per training listener its positives and its m p_pop draws. The p_pop
form-B loss of gam.softmax_block with stage_p13b.arm_block's offsets,
-log(m q_p) on every row, a positive weight of 1 / n_u, and 0 for a
positive with q_p = 0. The terms and their states follow the free layout.
A missing state that the free layout ties to the neutral level stays
tied: the term's mean over P11's p_pop rows with data, which are the first
16 p_pop draws of each listener (the prefix property).

The tree-GAM, with InterpretML's documented defaults of
ExplainableBoostingClassifier (as documented when the protocol was
declared) and greedy_ratio 0,
cyclic_progress on and no smoothing rounds, that is pure cyclic boosting,
the method of the EBM papers:
- Bins: per term, the edges of P13a's knot rule (gam.knot_rule) on a fine
  grid: 767 quantiles of the positives' values on the shape, 255 of the
  negatives', and the anchor 0. At most 1,024 value bins (max_bins). The
  bin of x is the number of edges below x. The zero and missing states
  have their own bins.
- Per round, one depth-1 tree per term (max_leaves 2), cyclic over the
  terms in the model's order. The tree splits the term's ordered value
  bins once, at the largest Newton gain, and each free state gets a leaf
  of its own (missing "separate"). Leaf value: minus the learning rate
  times the sum of gradients over the sum of the diagonal Hessian. The
  rows of a tied missing state enter each leaf by the chain rule, and the
  state follows the neutral level.
- Learning rate 0.015, at most 50,000 rounds, early stop after 100 rounds
  without a relative gain of 1e-5 in the validation loss (the bag keeps
  its best round), 14 outer bags, each with 15 percent of the training
  listeners held out, min_samples_leaf 4, min_hessian 1e-4. The model is
  the mean of the bags' tables.
- Each edge e becomes two knots, e and the next float above it, so
  gam.encode and catalog.CatalogScorer read the step shapes exactly.
- The sums run over fixed listener blocks in block order, and inside a
  block over fixed chunks of CHUNK listeners, so the fit does not depend on
  the worker count. Pinned workers keep their rows' scores and reuse work
  arrays of one chunk's size, which stay in the CPU cache. They compute
  the scores from the tables at the first round of a run and every
  CHECKPOINT_ROUNDS rounds, and add each term's change in between. The
  workers make no BLAS call and run under threadpool_limits(1), as P13a's
  workers do: in the first version, 16 processes with 16 BLAS threads each
  and large temporary arrays made the steps about 5 times slower
  (measured on two blocks of the real size).

Rules, declared before any result. The report has a section for each.
- T1, shape points: per term and explanation point (P13a's shape
  quantiles p5 to p95 of the positives, the zero state and the missing
  state, centered on the level's p_pop draws), the tree value against the
  free fit and against the final model, with that model's 95 percent
  listener band from P13c. A point separates when the difference is
  larger than TAU (0.5) and the tree value lies outside the band.
- T2, falls: per term, the net change from p5 to p95 in each bag. A term
  falls when the change is below 0 in all bags of the run, and rises when
  it is above 0 in all of them.
- T3, reasons: on P13c's reason sample, the signed top-3 agreement of the
  tree-GAM with the free fit and with the final model, and the share of
  picks whose artist reason differs by more than TAU.
- T4, accuracy: selection NDCG@100 of the tree-GAM and of each bag, and
  the paired difference to the free fit and to the final model, with
  gam_audit's interval and SESOI (2 percent of the model's score).
A result can change the text of the thesis, never a model or a reason
rule.

Checks while it runs: the same loss as
arm_block and gam.softmax_block, the step layout against the bins on every
fit row, the neutral background against P13a's, each bag's best loss
again (to a relative 1e-12: between two recomputations the scores come from
added changes), the pass against the dense reference, a read-back of the
outputs, and current markers of the stages it reads.

Not a runner stage: no later step reads its outputs. Usage, from the
repository root:
    python3 -m pipeline.tree_audit --pilot
    python3 -m pipeline.tree_audit
The pilot runs PILOT_ROUNDS rounds of bag 1, prints the time per round, a
projected stop round and the projected totals, and stops. The full command
continues from it. After Ctrl+C the run continues from the last checkpoint.
Fewer bags (finished bags count for any number), or fewer workers:
    python3 -m pipeline.tree_audit --bags 4
    TREE_WORKERS=8 python3 -m pipeline.tree_audit
"""

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import multiprocessing as mp
import os
import time
import traceback
from pathlib import Path

import numpy as np

from pipeline import catalog as ct
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_audit as ga
from pipeline import gam_refit as gr
from pipeline import p13c as pc
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2
from pipeline.als import PROGRESS_SECONDS, _clock, threadpool_limits
from pipeline.common import (
    atomic_json_write,
    atomic_text_write,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from pipeline.config import MODEL_DIR, SCRATCH_DIR

LOG_PATH = MODEL_DIR / "logs" / "tree_audit.log"
VERSION = "tree-audit-v1"
WORK_NAME = "tree_audit"
DECLARED = "before any code or result"
OUT_FILES = ("tree_audit.json", "tree_audit.npz")
REPORT_NAME = "tree_audit.md"
ARM = ("p_pop", "B")
POS_LEVELS = tuple((i + 1) / 768 for i in range(767))
NEG_LEVELS = tuple((i + 1) / 256 for i in range(255))
LEARNING_RATE = 0.015
MAX_ROUNDS = 50_000
PATIENCE = 100
TOLERANCE = 1e-5
BAGS = 14
VALIDATION = 0.15
MIN_SAMPLES_LEAF = 4
MIN_HESSIAN = 1e-4
N_BLOCKS = 64
CHUNK = 1024  # listeners per chunk of a block
WORKERS = max(1, int(os.environ.get("TREE_WORKERS", s13.FIT_WORKERS)))
CHECKPOINT_ROUNDS = 100
PILOT_ROUNDS = 1000
DECAY_WINDOW = 10  # rounds per gain in the pilot's fit
PILOT_BAGS = (14, 7, 4, 3, 2)  # bag counts in the pilot's table
SEED = 2026100401
TAU = s13b2.TAU
SESOI_SHARE = 0.02
VERIFY_USERS = 6
CHECK_BLOCKS = 3
CHECK_RTOL = 1e-12
BEST_RTOL = 1e-12  # check 4, relative
PASS_MINUTES = (26.0, 2.0)  # P13d's fit of the pass time per config
POINTS = pc.POINTS
STAGES = ("gam_refit", "p13c_stability", "p13d_bounds", "p13c_final")
ROBUST_LEVEL = 0.818  # P13d: signed top-3 agreement, bounded
WEAK_LEVEL = 0.644  # members and level-4096 fits

_W = {}  # the fit arrays, set before a fork


# Helpers


def _num(x):
    return None if x is None or not math.isfinite(float(x)) else float(x)


def _clean(x):
    """x with every non-finite float as None and numpy scalars as Python
    values, as the json file holds it."""
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _num(x)
    return x


def _norm(x):
    return json.loads(json.dumps(_clean(x), sort_keys=True))


def _text(x):
    """Text for the report, which holds no semicolon."""
    return str(x).replace(";", ",")


@contextlib.contextmanager
def relabel_logs():
    """P13's pass lines carry this module's name while it runs them."""

    class Relabel(logging.Filter):
        def filter(self, record):
            if isinstance(record.msg, str):
                record.msg = gr._STAGE_PREFIX.sub(
                    WORK_NAME, record.msg, count=1
                ).replace(
                    "python3 -m pipeline.runner --only p13a_fit",
                    "python3 -m pipeline.tree_audit",
                )
            return True

    flt = Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def settings():
    """The fit's settings, as the saved work's key holds them. The number
    of bags is not part of it: each bag stands alone."""
    return {
        "arm": "-".join(ARM),
        "pos_levels": list(POS_LEVELS),
        "neg_levels": list(NEG_LEVELS),
        "learning_rate": LEARNING_RATE,
        "max_rounds": MAX_ROUNDS,
        "patience": PATIENCE,
        "tolerance": TOLERANCE,
        "validation": VALIDATION,
        "min_samples_leaf": MIN_SAMPLES_LEAF,
        "min_hessian": MIN_HESSIAN,
        "blocks": N_BLOCKS,
        "chunk": CHUNK,
        "seed": SEED,
    }


def check_markers(stages=None):
    """Check 6: the markers of STAGES are current, by the runner's own
    test. stages: stage objects that replace the runner's (the smoke test).
    Returns {"marker:<stage>": SHA-256}."""
    from pipeline import runner as rn  # the runner imports every stage

    by = dict(rn.STAGE_BY_NAME)
    by.update({s.name: s for s in (stages or [])})
    params = dict(rn.DEFAULT_PARAMS)  # these stages read no parameter
    out = {}
    for name in STAGES:
        st = by[name]
        if not rn._stage_current(
            st, rn.stage_params_digest(st, params), rn.stage_source_sha(st), []
        ):
            raise ValueError(
                f"the marker of {name} is not current: run "
                f"python3 -m pipeline.runner first"
            )
        out[f"marker:{name}"] = file_sha256(marker_path(name))
    return out


def provenance(model_dir):
    """The input files' SHA-256: P13c's provenance of the final model and
    the files that p13c_final.json names."""
    d = Path(model_dir)
    out = pc.provenance(d, "p13d")
    ptr = load_json(d / pc.FINAL_FILE) or {}
    names = [pc.FINAL_FILE] + sorted((ptr.get("files") or {}).values())
    for name in names:
        p = d / name
        out[name] = file_sha256(p) if p.exists() else None
    return out


# Step layout


def step_knots(edges):
    """Each edge e and the next float above it: a PWL table with equal
    heights on the two knots of a bin is the bin's step, for every float."""
    e = np.asarray(edges, dtype=np.float64)
    k = np.empty(2 * len(e), dtype=np.float64)
    k[0::2] = e
    k[1::2] = np.nextafter(e, np.inf)
    return k


class StepLayout:
    """The tree-GAM's tables in gam's table format. Per term j: n_cont[j]
    value bins (edges[j] has n_cont[j] - 1 edges), then the zero and the
    missing bin. Knot 2k is edge k and closes bin k. Knot 2k + 1 opens bin
    k + 1. It has the terms, offsets and L that catalog, gam.center and
    p13c.Reasons read."""

    def __init__(self, terms, edges):
        self.edges = [np.asarray(e, dtype=np.float64) for e in edges]
        self.terms = tuple(
            dataclasses.replace(t, knots=tuple(float(x) for x in step_knots(e)))
            for t, e in zip(terms, self.edges)
        )
        sizes = [t.size for t in self.terms]
        self.offsets = [int(x) for x in np.cumsum([0] + sizes[:-1])]
        self.L = int(sum(sizes))
        self.n_cont = [len(e) + 1 for e in self.edges]
        self.nb = [n + 2 for n in self.n_cont]

    def bin_of(self, j, x):
        """Value bin of each continuous value x of term j."""
        return np.searchsorted(
            self.edges[j], np.asarray(x, dtype=np.float64), side="left"
        )

    def term_table(self, j, v):
        t, n = self.terms[j], self.n_cont[j]
        v = np.asarray(v, dtype=np.float64)
        T = np.zeros(t.size, dtype=np.float64)
        T[0 : t.K : 2] = v[: n - 1]
        T[1 : t.K : 2] = v[1:n]
        T[t.K + gam.ZERO] = v[n]
        T[t.K + gam.MISS] = v[n + 1]
        return T

    def flat(self, values):
        """(L,) flat tables from the per-term bin values."""
        return np.concatenate([self.term_table(j, v) for j, v in enumerate(values)])

    def to_json(self):
        return [t.to_json() for t in self.terms]


def _bins(t, e, v, h):
    """The bin of every row: value bins 0 .. n - 1, then zero (n) and
    missing (n + 1). For a term without a missing state, bin n + 1 holds
    the rows without data, which gam.encode calls not applicable (0)."""
    n = len(e) + 1
    b = np.full(np.shape(v), n + 1, dtype=np.uint16)
    cont = h & (v != 0.0) if t.zero_state else h
    b[cont] = np.searchsorted(e, v[cont], side="left")
    if t.zero_state:
        b[h & (v == 0.0)] = n
    return b


# The rows


def fit_rows(st, base, M, say):
    """The rows of the p_pop form-B loss at the model's level, binned, with
    checks 1 (offsets), 2 (step layout) and the neutral background. The
    level's rows are freed before it returns."""
    t0 = time.time()
    lay = M["lay"]
    terms = list(lay.terms)
    for t in terms:
        if t.tiers is not None or t.multiplier is not None or t.centered:
            raise ValueError(f"{t.name}: the audit supports plain terms " f"only")
        if t.zero_state and not lay.zero_free[t.name]:
            raise ValueError(
                f"{t.name}: a zero state without training " f"rows is not supported"
            )
    level = int(M["level"])
    m = s13b2.n_negatives(ARM, level)
    say(
        f"{WORK_NAME}: loading level {level} of the wide file (no progress "
        f"lines in this step)"
    )
    ctx = st._level_context(base, "wide", level, None)
    inp = st._inputs()
    lens, n_pos = np.asarray(inp["lens"]), np.asarray(inp["n_pos"])
    m_neg = int(inp["m_neg"])
    U = len(lens)
    cols = np.asarray(s13b.neg_cols(ARM[0], m_neg // 2), dtype=np.int64)
    if len(cols) != m:
        raise ValueError(
            f"the level holds {len(cols)} p_pop draws per " f"listener, not {m}"
        )
    starts = np.concatenate(([0], np.cumsum(lens)[:-1])).astype(np.int64)
    P = int(n_pos.sum())
    pos_user = np.repeat(np.arange(U), n_pos)
    first = np.repeat(np.cumsum(n_pos) - n_pos, n_pos)
    pos_idx = starts[pos_user] + (np.arange(P) - first)
    neg_idx = (starts + n_pos)[:, None] + cols[None, :]
    kind = inp["kind"]
    if not (
        np.all(kind[pos_idx] == fp.KIND_POSITIVE)
        and np.all(kind[neg_idx] == fp.KIND_POP)
    ):
        raise ValueError("the level's rows do not follow P11's layout")
    with np.errstate(divide="ignore"):
        off_pos = -np.log(m * ctx["q_p"][pos_idx])
        off_neg = -np.log(m * ctx["q_p"][neg_idx])
    ok_pos = np.isfinite(off_pos)
    off_pos = np.where(ok_pos, off_pos, 0.0)
    if not np.all(np.isfinite(off_neg)):
        raise ValueError("a p_pop draw has no p_pop probability")
    check_offsets(ctx, starts, lens, n_pos, m_neg, off_pos, off_neg, ok_pos, pos_user)
    rng = np.random.default_rng(SEED)
    edges, bins_pos, bins_neg, neutral, bg = [], [], [], {}, []
    n16 = s13b.NEG_PER_KIND
    for j, t in enumerate(terms):
        v, h = s13.concept_column(inp["f"], inp["rows"], t.source)
        vp, hp, vn, hn = v[pos_idx], h[pos_idx], v[neg_idx], h[neg_idx]
        del v, h
        cp = hp & (vp != 0.0) if t.zero_state else hp
        cq = hn & (vn != 0.0) if t.zero_state else hn
        e = np.asarray(
            gam.knot_rule(vp[cp], vn[cq], POS_LEVELS, NEG_LEVELS), dtype=np.float64
        )
        bp, bn = _bins(t, e, vp, hp), _bins(t, e, vn, hn)
        check_step(t, e, [(vp, hp, bp), (vn.ravel(), hn.ravel(), bn.ravel())], rng)
        if t.missing_state and not lay.miss_free[t.name]:
            neutral[j] = neutral_shares(
                t, e, vn[:, :n16], hn[:, :n16], bn[:, :n16], base
            )
        bg.append(np.bincount(bn.ravel(), minlength=len(e) + 3) / float(bn.size))
        edges.append(e)
        bins_pos.append(bp)
        bins_neg.append(bn)
        say(f"{WORK_NAME}: {t.name}: {len(e) + 1} value bins")
    del ctx
    st._drop_inputs()
    layout = StepLayout(terms, edges)
    ranges = []
    pos_start = np.concatenate(([0], np.cumsum(n_pos))).astype(np.int64)
    for l0, l1 in s13._blocks(n_pos + m, N_BLOCKS):
        ranges.append((l0, l1, int(pos_start[l0]), int(pos_start[l1])))
    D = {
        "U": U,
        "P": P,
        "m": m,
        "n_pos": n_pos.astype(np.int64),
        "pos_user": pos_user.astype(np.int64),
        "ok_pos": ok_pos,
        "off_pos": off_pos,
        "off_neg": off_neg,
        "bins_pos": np.stack(bins_pos),
        "bins_neg": np.stack(bins_neg),
        "nb": list(layout.nb),
        "J": len(terms),
        "ranges": ranges,
        "layout": layout,
        "neutral": neutral,
        "bg": bg,
        "zero_free": [bool(t.zero_state) for t in terms],
        "miss_free": [bool(lay.miss_free[t.name]) for t in terms],
        "tr": np.asarray(inp["tr"]),
    }
    for nb in layout.nb:
        if nb > np.iinfo(np.uint16).max:
            raise ValueError("more bins than uint16 holds")
    say(
        f"{WORK_NAME}: {U:,} listeners, {P:,} positives, {U * m:,} p_pop "
        f"draws, binned in {_clock(time.time() - t0)}"
    )
    return D


def check_offsets(ctx, starts, lens, n_pos, m_neg, off_pos, off_neg, ok_pos, pos_user):
    """Check 1, offsets: on CHECK_BLOCKS listener ranges, arm_block's form-B
    offsets and positive weights equal the audit's, exactly."""
    U = len(lens)
    pos_first = np.concatenate(([0], np.cumsum(n_pos))).astype(np.int64)
    for lo, hi in _check_ranges(U):
        r0 = int(starts[lo])
        r1 = int(starts[hi - 1] + lens[hi - 1])
        ss = starts[lo:hi] - r0
        np_ = n_pos[lo:hi]
        pos_rows = np.concatenate([ss[k] + np.arange(np_[k]) for k in range(hi - lo)])
        neg_rows = (ss + np_)[:, None] + np.arange(m_neg)[None, :]
        blk = {
            "neg_rows": neg_rows,
            "pos_rows": pos_rows,
            "pos_weight": np.ones(len(pos_rows)),
            "offset": np.zeros(r1 - r0),
            "q_u": ctx["q_u"][r0:r1],
            "q_p": ctx["q_p"][r0:r1],
        }
        out = s13b.arm_block(blk, ARM)
        p0, p1 = int(pos_first[lo]), int(pos_first[hi])
        if not (
            np.array_equal(out["offset"][pos_rows], off_pos[p0:p1])
            and np.array_equal(out["offset"][out["neg_rows"]], off_neg[lo:hi])
            and np.array_equal(out["pos_weight"] > 0, ok_pos[p0:p1])
        ):
            raise ValueError(
                "the audit's offsets or positive weights differ "
                "from stage_p13b.arm_block's"
            )


def _check_ranges(U):
    k = max(1, min(CHECK_BLOCKS, U))
    edges = np.linspace(0, U, k + 1).astype(np.int64)
    size = max(1, min(200, U // k))
    return [(int(a), int(min(a + size, U))) for a in edges[:-1]]


def check_step(t, e, groups, rng):
    """Check 2: with random bin values, the step layout's value of every
    row (gam.encode, gam.pwl_value) equals the value of the row's bin,
    exactly."""
    lay = StepLayout([t], [e])
    v_bins = rng.normal(size=lay.nb[0])
    if not t.missing_state:
        v_bins[-1] = 0.0  # "not applicable": gam.encode adds 0
    T = lay.term_table(0, v_bins)
    D = gam.diff_table(T, lay.terms, lay.offsets)
    for v, h, b in groups:
        slot, w = gam.encode(lay.terms[0], v, h)
        got = gam.pwl_value(T, D, slot.astype(np.int64), w)
        bad = int(np.count_nonzero(got != v_bins[b.astype(np.int64)]))
        if bad:
            raise ValueError(
                f"{t.name}: the step layout differs from the " f"bins on {bad} rows"
            )


def neutral_shares(t, e, v16, h16, b16, base):
    """The neutral level's weights per bin: the shares of P11's p_pop rows
    with data (the first 16 p_pop draws) in each bin. Checked: the same
    rows give P13a's mean basis of the term."""
    slot, w = gam.encode(t, v16.ravel(), h16.ravel())
    a_pwl = gam.basis_mean(t, slot, w, slot < t.K + gam.MISS)
    ref = np.asarray(base["data_means"][t.name], dtype=np.float64)
    if a_pwl.shape != ref.shape or not np.allclose(a_pwl, ref, rtol=0.0, atol=1e-12):
        raise ValueError(
            f"{t.name}: the neutral background differs from "
            f"P13a's (P11's p_pop rows are not the first "
            f"{v16.shape[1]} draws)"
        )
    n = len(e) + 1
    has = h16.ravel()
    counts = np.bincount(b16.ravel()[has], minlength=n + 2).astype(np.float64)
    counts[n + 1] = 0.0
    return counts / counts.sum()


# Workers
# A block runs in fixed chunks of CHUNK listeners. Each worker reuses three
# work arrays of one chunk's size, so a step makes no large temporary array
# and a chunk's arrays stay in the CPU cache. The chunks depend on the block
# only, so the sums do not depend on the worker count.
def _softmax(zp, zn, pu, e=None):
    """The pieces of gam.softmax_block's loss for one block or chunk: the
    negatives' exponentials shifted by their listener's largest score (into
    e when given), their sums, and the log-sums per positive."""
    mx = zn.max(axis=1)
    e = np.subtract(zn, mx[:, None], out=e)
    np.exp(e, out=e)
    S = e.sum(axis=1)
    lneg = mx + np.log(S)
    ln = lneg[pu]
    lse = np.logaddexp(zp, ln)
    return e, S, ln, lse


def _grad(zp, zn, pu, w_tr, e=None, g=None):
    """Loss pieces, and the gradient and diagonal Hessian of the training
    loss per row, for one block or chunk. e, g: optional work arrays of
    zn's shape. Returns (lse, gp, hp, g, h); h is e's array."""
    e, S, ln, lse = _softmax(zp, zn, pu, e)
    sp = np.exp(zp - lse)
    gp = w_tr * (sp - 1.0)
    hp = w_tr * sp * (1.0 - sp)
    r = np.exp(ln - lse)
    wr = w_tr * r
    U = zn.shape[0]
    a = np.bincount(pu, weights=wr, minlength=U)
    b = np.bincount(pu, weights=wr * r, minlength=U)
    g = np.multiply(e, (a / S)[:, None], out=g)  # q a, q = e / S
    np.multiply(e, e, out=e)
    np.multiply(e, (b / (S * S))[:, None], out=e)  # q^2 b
    np.subtract(g, e, out=e)  # h = q a - q^2 b
    return lse, gp, hp, g, e


def _rows(b):
    return _W["ranges"][b]


def _chunks(pu, n_users):
    """A block's chunks: (c0, c1, q0, q1, the positives' listener index in
    the chunk) for the listeners c0..c1-1 and the positives q0..q1-1 of the
    block (pu: the positives' listener index in the block, ascending)."""
    out = []
    for c0 in range(0, n_users, CHUNK):
        c1 = min(c0 + CHUNK, n_users)
        q0, q1 = (int(x) for x in np.searchsorted(pu, [c0, c1]))
        out.append((c0, c1, q0, q1, pu[q0:q1] - c0))
    return out


def _work(S, shape):
    """The worker's work arrays (bins, two float arrays) for one chunk of
    the given shape, as views of arrays that live for the whole run."""
    n = shape[0] * shape[1]
    B = S.get("buf")
    if B is None or B[1].size < n:
        B = S["buf"] = (np.empty(n, dtype=np.intp), np.empty(n), np.empty(n))
    return tuple(x[:n].reshape(shape) for x in B)


def _bag_block(S, b, val):
    l0, l1, p0, p1 = _rows(b)
    pu = _W["pos_user"][p0:p1] - l0
    w = (1.0 / _W["n_pos"][l0:l1])[pu] * _W["ok_pos"][p0:p1]
    vb = np.asarray(val[l0:l1], dtype=bool)
    S["pu"][b] = pu
    S["chunks"][b] = _chunks(pu, l1 - l0)
    S["w_tr"][b] = np.where(vb[pu], 0.0, w)
    S["w_val"][b] = np.where(vb[pu], w, 0.0)
    trp = S["w_tr"][b] > 0
    out = []
    for j in range(_W["J"]):
        nb = _W["nb"][j]
        c = np.bincount(_W["bins_pos"][j, p0:p1][trp], minlength=nb)
        c += np.bincount(_W["bins_neg"][j, l0:l1][~vb].ravel(), minlength=nb)
        out.append(c)
    return out


def _set_block(S, b, tables):
    """The block's scores from the tables: the offset plus each term's bin
    value, added in term order."""
    l0, l1, p0, p1 = _rows(b)
    zp = _W["off_pos"][p0:p1].copy()
    for j in range(_W["J"]):
        zp += tables[j][_W["bins_pos"][j, p0:p1]]
    zn = S["zn"].get(b)
    if zn is None:
        zn = S["zn"][b] = np.empty(_W["off_neg"][l0:l1].shape)
    for c0, c1, _, _, _ in S["chunks"][b]:
        zc = zn[c0:c1]
        idx, t, _ = _work(S, zc.shape)
        np.copyto(zc, _W["off_neg"][l0 + c0 : l0 + c1])
        for j in range(_W["J"]):
            np.copyto(idx, _W["bins_neg"][j, l0 + c0 : l0 + c1])
            np.take(tables[j], idx, out=t, mode="clip")
            zc += t
    S["zp"][b] = zp
    return None


def _step_block(S, b, jp, delta, jn):
    """Applies term jp's change, then returns (G, H) of term jn's bins and
    the block's training and validation loss sums (jn None: losses only).
    The sums run over the block's chunks in order."""
    l0, l1, p0, p1 = _rows(b)
    zp, zn = S["zp"][b], S["zn"][b]
    if jp is not None:
        zp += delta[_W["bins_pos"][jp, p0:p1]]
    w_tr, w_val = S["w_tr"][b], S["w_val"][b]
    G = H = None
    tl = vl = 0.0
    for c0, c1, q0, q1, pc in S["chunks"][b]:
        zc, zpc = zn[c0:c1], zp[q0:q1]
        idx, t1, t2 = _work(S, zc.shape)
        if jp is not None:
            np.copyto(idx, _W["bins_neg"][jp, l0 + c0 : l0 + c1])
            np.take(delta, idx, out=t2, mode="clip")
            zc += t2
        wt = w_tr[q0:q1]
        if jn is None:
            lse = _softmax(zpc, zc, pc, t1)[3]
        else:
            lse, gp, hp, g, h = _grad(zpc, zc, pc, wt, t1, t2)
            nb = _W["nb"][jn]
            np.copyto(idx, _W["bins_neg"][jn, l0 + c0 : l0 + c1])
            bp = _W["bins_pos"][jn, p0 + q0 : p0 + q1]
            Gc = np.bincount(
                idx.ravel(), weights=g.ravel(), minlength=nb
            ) + np.bincount(bp, weights=gp, minlength=nb)
            Hc = np.bincount(
                idx.ravel(), weights=h.ravel(), minlength=nb
            ) + np.bincount(bp, weights=hp, minlength=nb)
            G = Gc if G is None else G + Gc
            H = Hc if H is None else H + Hc
        lp = lse - zpc
        tl += float(np.einsum("i,i->", wt, lp))  # no BLAS threads
        vl += float(np.einsum("i,i->", w_val[q0:q1], lp))
    return G, H, tl, vl


def _new_state():
    return {"zp": {}, "zn": {}, "pu": {}, "chunks": {}, "w_tr": {}, "w_val": {}}


def _do(S, blocks, cmd):
    op = cmd[0]
    if op == "bag":
        return [_bag_block(S, b, cmd[1]) for b in blocks]
    if op == "set":
        return [_set_block(S, b, cmd[1]) for b in blocks]
    if op == "step":
        return [_step_block(S, b, *cmd[1:]) for b in blocks]
    raise KeyError(op)


def _serve(conn, blocks):
    S = _new_state()
    try:
        with threadpool_limits(1):  # 16 workers, one BLAS thread each
            while True:
                cmd = conn.recv()
                if cmd[0] == "quit":
                    return
                conn.send(("ok", _do(S, blocks, cmd)))
    except (EOFError, KeyboardInterrupt):
        return
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class Pinned:
    """Workers that each keep a fixed set of the fit blocks, with the
    blocks' scores, for the whole run. call() returns one result per block
    in block order. With one worker the blocks run in this process."""

    def __init__(self, workers):
        self.workers = max(1, min(int(workers), len(_W["ranges"])))
        self.procs, self.conns = [], []

    def __enter__(self):
        B = len(_W["ranges"])
        self.assign = [list(range(k, B, self.workers)) for k in range(self.workers)]
        if self.workers == 1:
            self.S = _new_state()
            return self
        ctx = mp.get_context("fork")
        for blocks in self.assign:
            mine, theirs = ctx.Pipe()
            p = ctx.Process(target=_serve, args=(theirs, blocks), daemon=True)
            p.start()
            theirs.close()
            self.procs.append(p)
            self.conns.append(mine)
        return self

    def call(self, *cmd):
        out = [None] * len(_W["ranges"])
        if self.workers == 1:
            for b, r in zip(self.assign[0], _do(self.S, self.assign[0], cmd)):
                out[b] = r
            return out
        for c in self.conns:
            c.send(cmd)
        for c, blocks in zip(self.conns, self.assign):
            status, res = c.recv()
            if status != "ok":
                raise RuntimeError(f"{WORK_NAME}: a worker stopped:\n{res}")
            for b, r in zip(blocks, res):
                out[b] = r
        return out

    def __exit__(self, exc_type, exc, tb):
        for c in self.conns:
            try:
                c.send(("quit",))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join(timeout=10)
        return False


def check_loss(D):
    """Check 1, loss: on CHECK_BLOCKS listener ranges, with random tables,
    the audit's loss and gradient equal gam.softmax_block's, to a relative
    CHECK_RTOL (the order of the float operations differs)."""
    rng = np.random.default_rng(SEED + 1)
    tables = [rng.normal(scale=0.5, size=nb) for nb in D["nb"]]
    pos_first = np.concatenate(([0], np.cumsum(D["n_pos"]))).astype(np.int64)
    for lo, hi in _check_ranges(D["U"]):
        p0, p1 = int(pos_first[lo]), int(pos_first[hi])
        zp, zn = D["off_pos"][p0:p1].copy(), D["off_neg"][lo:hi].copy()
        sp_, sn = np.zeros(p1 - p0), np.zeros(zn.shape)
        for j in range(D["J"]):
            sp_ += tables[j][D["bins_pos"][j, p0:p1]]
            sn += tables[j][D["bins_neg"][j, lo:hi]]
        zp += sp_
        zn += sn
        pu = D["pos_user"][p0:p1] - lo
        w = (1.0 / D["n_pos"][lo:hi])[pu] * D["ok_pos"][p0:p1]
        lse, gp, _, g, _ = _grad(zp.copy(), zn.copy(), pu, w)
        loss = float(np.dot(w, lse - zp))
        P_b = p1 - p0
        s = np.concatenate((sp_, sn.ravel()))
        off = np.concatenate((D["off_pos"][p0:p1], D["off_neg"][lo:hi].ravel()))
        neg_rows = P_b + np.arange(sn.size).reshape(sn.shape)
        ref_loss, ref_g = gam.softmax_block(s, off, np.arange(P_b), pu, w, neg_rows)
        mine = np.concatenate((gp, g.ravel()))
        scale = float(np.max(np.abs(ref_g))) or 1.0
        if (
            abs(loss - ref_loss) > CHECK_RTOL * abs(ref_loss)
            or float(np.max(np.abs(mine - ref_g))) > CHECK_RTOL * scale
        ):
            raise ValueError(
                "the audit's loss or gradient differs from " "gam.softmax_block's"
            )


# The fit


def bag_split(U, bag):
    """The validation listeners of a bag (seeded)."""
    rng = np.random.default_rng([SEED, int(bag)])
    val = np.zeros(U, dtype=bool)
    val[rng.permutation(U)[: int(round(VALIDATION * U))]] = True
    return val


def tree_step(G, H, C, n, zero_free, miss_free, a=None, lr=None):
    """(the change of one term's bin values, the split) from one depth-1
    tree over its n value bins, a leaf for each free state, and the tie of
    a missing state to the neutral level (a: the neutral weights per bin,
    or None for a free missing state)."""
    lr = LEARNING_RATE if lr is None else float(lr)
    G = np.asarray(G, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    C = np.asarray(C, dtype=np.int64)
    tied = a is not None
    Gm, Hm = float(G[n + 1]), float(H[n + 1])
    delta = np.zeros(n + 2, dtype=np.float64)
    split = None

    def eff(g, h, A):
        return (g + A * Gm, h + A * A * Hm) if tied else (g, h)

    if n >= 2:
        cg, ch, cc = np.cumsum(G[:n]), np.cumsum(H[:n]), np.cumsum(C[:n])
        ca = np.cumsum(a[:n]) if tied else np.zeros(n)
        gl, hl, cl, al = cg[:-1], ch[:-1], cc[:-1], ca[:-1]
        gle, hle = eff(gl, hl, al)
        gre, hre = eff(cg[-1] - gl, ch[-1] - hl, ca[-1] - al)
        cr = cc[-1] - cl
        ok = (
            (cl >= MIN_SAMPLES_LEAF)
            & (cr >= MIN_SAMPLES_LEAF)
            & (hle >= MIN_HESSIAN)
            & (hre >= MIN_HESSIAN)
        )
        if ok.any():
            with np.errstate(divide="ignore", invalid="ignore"):
                gain = np.where(ok, gle * gle / hle + gre * gre / hre, -np.inf)
            k = int(np.argmax(gain))
            split = k + 1
            delta[:split] = -lr * gle[k] / hle[k]
            delta[split:n] = -lr * gre[k] / hre[k]
    if zero_free and C[n] >= MIN_SAMPLES_LEAF:
        gz, hz = eff(float(G[n]), float(H[n]), float(a[n]) if tied else 0.0)
        if hz >= MIN_HESSIAN:
            delta[n] = -lr * gz / hz
    if tied:
        delta[n + 1] = float(np.dot(a[: n + 1], delta[: n + 1]))
    elif miss_free and C[n + 1] >= MIN_SAMPLES_LEAF and Hm >= MIN_HESSIAN:
        delta[n + 1] = -lr * Gm / Hm
    return delta, split


def _sum_blocks(res):
    G = H = None
    tl = vl = 0.0
    for g, h, t, v in res:
        if g is not None:
            G = g.copy() if G is None else G + g
            H = h.copy() if H is None else H + h
        tl += t
        vl += v
    return G, H, tl, vl


class BagStore:
    """Each finished bag and the running bag's checkpoint, keyed by the
    run's inputs, the settings and the fit's code. A file with another key
    counts as absent."""

    def __init__(self, work, key):
        self.dir = Path(work)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.k = s13._sha(key)

    def _get(self, name):
        doc = load_json(self.dir / name)
        return doc if doc is not None and doc.get("k") == self.k else None

    def bag(self, b):
        return self._get(f"bag_{b:02d}.json")

    def checkpoint(self, b):
        return self._get(f"checkpoint_{b:02d}.json")

    def put_bag(self, b, rec):
        atomic_json_write(dict(rec, k=self.k), self.dir / f"bag_{b:02d}.json")

    def put_checkpoint(self, b, state):
        atomic_json_write(dict(state, k=self.k), self.dir / f"checkpoint_{b:02d}.json")


def run_bag(pool, D, bag, store, say, stop_at=None):
    """One outer bag (module docstring). The state is saved before the
    bookkeeping of every CHECKPOINT_ROUNDS-th round. The workers compute
    the scores again from the tables at those rounds and at the first round
    of a run, and add each term's change in between, so a resumed bag
    repeats the same steps bit for bit. stop_at: the pilot's round, a
    multiple of CHECKPOINT_ROUNDS. Returns the bag record, or None when the
    pilot stops first."""
    done = store.bag(bag)
    if done is not None:
        return done
    J = D["J"]
    val = bag_split(D["U"], bag)
    counts = [sum(c[j] for c in pool.call("bag", val)) for j in range(J)]
    n_tr, n_val = int((~val).sum()), int(val.sum())
    ck = store.checkpoint(bag)
    if ck is not None:
        tables = [np.asarray(x, dtype=np.float64) for x in ck["tables"]]
        rnd, best, best_round = (
            int(ck["round"]),
            float(ck["best"]),
            int(ck["best_round"]),
        )
        best_tables = (
            [np.asarray(x, dtype=np.float64) for x in ck["best_tables"]]
            if ck["best_tables"] is not None
            else None
        )
        since, curve_v, curve_t = (
            int(ck["since"]),
            list(ck["curve_val"]),
            list(ck["curve_train"]),
        )
        secs = float(ck["seconds"])
        say(
            f"{WORK_NAME}: bag {bag + 1} of {BAGS}: continues at round "
            f"{rnd:,} from the saved work"
        )
    else:
        tables = [np.zeros(nb, dtype=np.float64) for nb in D["nb"]]
        rnd, best, best_round, best_tables = 0, math.inf, -1, None
        since, curve_v, curve_t, secs = 0, [], [], 0.0
    last_ck, fresh, delta = rnd, True, None
    t_bag = time.time() - secs
    t_last = time.time()
    a_tie = [D["neutral"].get(j) for j in range(J)]
    n_cont = D["layout"].n_cont
    while True:
        if (
            rnd % CHECKPOINT_ROUNDS == 0
            and rnd > last_ck
            or (stop_at is not None and rnd >= stop_at)
        ):
            store.put_checkpoint(
                bag,
                {
                    "round": rnd,
                    "tables": [x.tolist() for x in tables],
                    "best": best,
                    "best_round": best_round,
                    "best_tables": (
                        None
                        if best_tables is None
                        else [x.tolist() for x in best_tables]
                    ),
                    "since": since,
                    "curve_val": curve_v,
                    "curve_train": curve_t,
                    "seconds": time.time() - t_bag,
                },
            )
            last_ck = rnd
        if stop_at is not None and rnd >= stop_at:
            return None
        if fresh or rnd % CHECKPOINT_ROUNDS == 0:
            pool.call("set", tables)
            G, H, tl, vl = _sum_blocks(pool.call("step", None, None, 0))
        else:
            G, H, tl, vl = _sum_blocks(pool.call("step", J - 1, delta, 0))
        fresh = False
        vl /= n_val
        curve_v.append(vl)
        curve_t.append(tl / n_tr)
        if vl < best - TOLERANCE * abs(best) or best_tables is None:
            best, best_round, since = vl, rnd, 0
            best_tables = [x.copy() for x in tables]
        else:
            since += 1
        if since >= PATIENCE or rnd >= MAX_ROUNDS:
            break
        for j in range(J):
            if j > 0:
                G, H, _, _ = _sum_blocks(pool.call("step", j - 1, delta, j))
            delta, _ = tree_step(
                G,
                H,
                counts[j],
                n_cont[j],
                D["zero_free"][j],
                D["miss_free"][j],
                a_tie[j],
            )
            tables[j] = tables[j] + delta
        rnd += 1
        now = time.time()
        if now - t_last >= PROGRESS_SECONDS:
            t_last = now
            el = now - t_bag
            per = el / max(rnd, 1)
            say(
                f"{WORK_NAME}: bag {bag + 1} of {BAGS}, round {rnd:,} "
                f"({rnd / MAX_ROUNDS:.1%} of the cap of {MAX_ROUNDS:,}): "
                f"validation loss {vl:.8f}, best {best:.8f} at round "
                f"{best_round:,}, {since} rounds without a gain (stop at "
                f"{PATIENCE}), {per:.2f} s per round, {_clock(el)} elapsed, "
                f"at most {_clock(per * (MAX_ROUNDS - rnd))} left"
            )
    pool.call("set", best_tables)
    _, _, _, again = _sum_blocks(pool.call("step", None, None, None))
    again /= n_val
    if abs(again - best) > BEST_RTOL * abs(best):
        raise ValueError(
            f"bag {bag + 1}: the best tables give the validation "
            f"loss {again!r}, not the stored {best!r}"
        )
    rec = {
        "bag": bag,
        "rounds": rnd,
        "best_round": best_round,
        "stop": "early" if since >= PATIENCE else "cap",
        "best_loss": best,
        "validation_listeners": n_val,
        "tables": [x.tolist() for x in best_tables],
        "curve_val": curve_v,
        "curve_train": curve_t,
        "seconds": time.time() - t_bag,
    }
    store.put_bag(bag, rec)
    return rec


def fit_code():
    """SHA-256 of the fit's code: a change in it starts the bags anew."""
    src = "".join(
        inspect.getsource(f)
        for f in (
            _bins,
            step_knots,
            _softmax,
            _grad,
            _chunks,
            _work,
            _bag_block,
            _set_block,
            _step_block,
            bag_split,
            tree_step,
            _sum_blocks,
            run_bag,
        )
    )
    return hashlib.sha256(src.encode("utf-8")).hexdigest()


def decay_fit(curve):
    """A power law of the validation gain per round, log gain on log round,
    fitted over the second half of the curve in windows of DECAY_WINDOW
    rounds. On the first 200 rounds of the real pilot it predicted the gains
    of rounds 135 to 195 within 1 percent, and an exponential fit 13 to 56
    percent too low. Returns {slope, stop: the round where PATIENCE rounds
    gain less than TOLERANCE of the loss, plus PATIENCE (at most the cap),
    loss_at(r), limit: the loss the law approaches (slope below -1)}, or
    None when the gains do not fall."""
    c = np.asarray(curve, dtype=np.float64)
    w = DECAY_WINDOW
    k = np.arange(1, (len(c) - 1) // w + 1)
    x = k * w - w / 2.0
    g = (c[(k - 1) * w] - c[k * w]) / w
    keep = (x >= (len(c) - 1) / 2.0) & (g > 0)
    if keep.sum() < 3:
        return None
    beta, alpha = (float(v) for v in np.polyfit(np.log(x[keep]), np.log(g[keep]), 1))
    if beta >= 0.0:
        return None
    r0, A = len(c) - 1, math.exp(alpha)

    def loss_at(r):
        r = max(float(r), float(r0))
        if abs(beta + 1.0) < 1e-12:
            return float(c[-1] - A * math.log(r / r0))
        return float(
            c[-1] - A / (beta + 1.0) * (r ** (beta + 1.0) - r0 ** (beta + 1.0))
        )

    g_stop = TOLERANCE * abs(float(c[-1])) / PATIENCE
    r_stop = math.exp((math.log(g_stop) - alpha) / beta) + PATIENCE
    return {
        "slope": beta,
        "stop": int(min(MAX_ROUNDS, max(r0, r_stop))),
        "loss_at": loss_at,
        "limit": (
            float(c[-1] - A * r0 ** (beta + 1.0) / (-beta - 1.0))
            if beta < -1.0
            else None
        ),
    }


def free_loss(d):
    """The free fit's training loss on the level's rows (gam_refit.json)."""
    doc = load_json(Path(d) / "gam_refit.json")
    f = doc["fits"][int(doc["chosen"]["index"])]
    return float((f.get("fit") or f["p13b2_fit"])["loss"])


def pilot_report(store, say, t_rounds, ref_loss=None):
    """The pilot's time per round, the decay of the gains, and the
    projected totals for several bag counts."""
    ck = store.checkpoint(0)
    done = store.bag(0)
    rec = done or ck
    rounds = int(rec["rounds"] if done else rec["round"])
    per = float(rec["seconds"]) / max(rounds, 1)
    c = rec["curve_val"]
    fit = None if done else decay_fit(c)
    stop = rounds if done else (fit["stop"] if fit else None)
    a, b = PASS_MINUTES

    def total(n, r):
        return n * r * per + 60.0 * (a + b * (1 + n))

    say(
        f"{WORK_NAME}: pilot done: {rounds:,} rounds of bag 1 in "
        f"{_clock(rec['seconds'])}, {per:.3f} s per round (loading and "
        f"bins {_clock(t_rounds)}). Validation loss {c[0]:.6f} at round 0, "
        f"{c[-1]:.6f} at round {len(c) - 1:,}"
        + (
            ""
            if ref_loss is None
            else f"; the free fit's training loss on these rows {ref_loss:.6f}"
        )
        + "."
    )
    if done:
        say(f"{WORK_NAME}: bag 1 stopped early at round {rounds:,}.")
    elif fit is None:
        say(
            f"{WORK_NAME}: the gains do not fall yet, so no stop round can "
            f"be projected. The totals below use the cap of "
            f"{MAX_ROUNDS:,} rounds."
        )
    else:
        caps = [r for r in (5_000, 10_000, 20_000) if r < fit["stop"]]
        say(
            f"{WORK_NAME}: the gain per round falls as a power law (slope "
            f"{fit['slope']:.2f} on rounds {(len(c) - 1) // 2:,} to "
            f"{len(c) - 1:,}). A bag stops when {PATIENCE} rounds gain less "
            f"than {TOLERANCE:g} of the loss: projected near round "
            f"{fit['stop']:,}. Projected validation loss: "
            + ", ".join(
                f"{fit['loss_at'](r):.5f} at round {r:,}" for r in caps + [fit["stop"]]
            )
            + ("" if fit["limit"] is None else f"; the law's limit {fit['limit']:.5f}")
            + "."
        )
    r = stop or MAX_ROUNDS
    say(
        f"{WORK_NAME}: {_clock(r * per)} per bag. Projected totals with the "
        f"selection pass: "
        + ", ".join(
            f"{n} bags {_clock(total(n, r))}"
            for n in sorted({min(n, BAGS) for n in PILOT_BAGS} | {BAGS}, reverse=True)
        )
        + "."
    )
    say(
        f"{WORK_NAME}: next: python3 -m pipeline.tree_audit --bags N (it "
        f"continues from round {rounds:,} of bag 1, and finished bags count "
        f"for any N; the protocol's N is {BAGS}). If the total is above "
        f"12 h, choose the number of bags before the full run."
    )
    return {
        "pilot_rounds": rounds,
        "seconds_per_round": per,
        "projected_stop": stop,
        "projected_total": total(BAGS, r),
        "slope": None if fit is None else fit["slope"],
    }


# Evaluation


def centered(values, bg):
    """Each term's bin values minus their mean over the level's p_pop
    draws (P13c's background)."""
    return [v - float(np.dot(w, v)) for v, w in zip(values, bg)]


def term_points(layout, values_c, xq):
    """{term: the 7 values at POINTS, None where a term has none}."""
    out = {}
    for j, t in enumerate(layout.terms):
        xs = list(xq.get(t.name) or [])[:5]
        n = layout.n_cont[j]
        row = [float(values_c[j][int(i)]) for i in layout.bin_of(j, xs)] if xs else []
        row += [None] * (5 - len(row))
        row.append(float(values_c[j][n]) if t.zero_state else None)
        row.append(float(values_c[j][n + 1]) if t.missing_state else None)
        out[t.name] = row
    return out


def band_table(bands):
    """{term: {point: (value, lo, hi)}} from a P13c results file."""
    return {
        name: {
            r["point"]: (r["value"], r["listener"]["lo"], r["listener"]["hi"])
            for r in rows
        }
        for name, rows in bands.items()
    }


def summarize(layout, bag_vals, bg, xq, bands, val_tree, reasons, ranks, model_ranks):
    """T1 to T4 from the stored tables and ranks (protocol). bands:
    {"free", "final"}: P13c's band tables. reasons: {"tree": R, "free": (R,
    values), "final": (R, values)}. model_ranks: {"free", "final"}: (n, 2)
    ranks."""
    names = [t.name for t in layout.terms]
    mean = [
        np.mean(np.stack([bv[j] for bv in bag_vals]), axis=0) for j in range(len(names))
    ]
    pts = term_points(layout, centered(mean, bg), xq)
    bag_pts = [term_points(layout, centered(bv, bg), xq) for bv in bag_vals]
    t1, sep = [], []
    for name in names:
        for p, label in enumerate(POINTS):
            tv = pts[name][p]
            if tv is None:
                continue
            bvals = [bp[name][p] for bp in bag_pts]
            row = {
                "term": name,
                "point": label,
                "tree": tv,
                "tree_bags": [min(bvals), max(bvals)],
            }
            for which in ("free", "final"):
                ref = bands[which].get(name, {}).get(label)
                if ref is None:
                    row[which] = None
                    continue
                v, lo, hi = ref
                diff = tv - v
                outside = tv < lo or tv > hi
                flag = abs(diff) > TAU and outside
                row[which] = {
                    "value": v,
                    "lo": lo,
                    "hi": hi,
                    "diff": diff,
                    "outside_band": outside,
                    "separates": flag,
                }
                if flag:
                    sep.append(f"{name} {label} ({which})")
            t1.append(row)
    t2 = {}
    for j, name in enumerate(names):
        xs = list(xq.get(name) or [])
        if len(xs) < 5:
            continue
        i5, i95 = (int(i) for i in layout.bin_of(j, [xs[0], xs[4]]))
        nets = [float(bv[j][i95] - bv[j][i5]) for bv in bag_vals]
        row = {
            "bag_net": nets,
            "net": float(mean[j][i95] - mean[j][i5]),
            "falls": all(x < 0 for x in nets),
            "rises": all(x > 0 for x in nets),
        }
        for which in ("free", "final"):
            ref = bands[which].get(name, {})
            row[f"{which}_net"] = (
                ref["p95"][0] - ref["p5"][0] if "p95" in ref and "p5" in ref else None
            )
        t2[name] = row
    R_t = reasons["tree"]
    groups = [g for g, _ in R_t.groups]
    art = groups.index("artist") if "artist" in groups else None
    top_t = R_t.signed_top(val_tree)
    t3 = {
        "picks": int(R_t.cand.shape[1]),
        "groups": groups,
        "top3_positive": {
            "tree": [float(x) for x in (top_t[0] & top_t[1]).mean(axis=1)]
        },
        "agreement": {},
        "artist_differs": {},
    }
    for which in ("free", "final"):
        R_m, val_m = reasons[which]
        top_m = R_m.signed_top(val_m)
        agree = float(pc.agreement(top_t, top_m).mean())
        t3["agreement"][which] = agree
        t3["top3_positive"][which] = [
            float(x) for x in (top_m[0] & top_m[1]).mean(axis=1)
        ]
        if art is not None:
            c = R_t.cand[art]
            t3["artist_differs"][which] = (
                float(np.mean(np.abs(val_tree[art] - val_m[art])[c] > TAU))
                if c.any()
                else None
            )
        t3[f"level_{which}"] = (
            "robust"
            if agree >= ROBUST_LEVEL
            else "family-dependent" if agree <= WEAK_LEVEL else "between"
        )
    sc = ga.per_listener_scores(ranks)
    s_free = ga.per_listener_scores(model_ranks["free"][None])[0]
    s_final = ga.per_listener_scores(model_ranks["final"][None])[0]
    sesoi = SESOI_SHARE * float(s_final.mean())

    def cmp(a, b):
        dd = ga.paired_summary(a, b)
        dd["decision"] = (
            "above"
            if dd["mean"] > sesoi and dd["lo"] > 0
            else "below" if dd["mean"] < -sesoi and dd["hi"] < 0 else "within"
        )
        return dd

    t4 = {
        "tree": float(sc[0].mean()),
        "bags": [float(x) for x in sc[1:].mean(axis=1)],
        "free": float(s_free.mean()),
        "final": float(s_final.mean()),
        "sesoi": sesoi,
        "tree_minus_free": cmp(sc[0], s_free),
        "tree_minus_final": cmp(sc[0], s_final),
        "final_minus_free": ga.paired_summary(s_final, s_free),
    }
    out = {
        "points": {"rows": t1, "separating": sep},
        "falls": t2,
        "reasons": t3,
        "scores": t4,
        "shapes": {name: {"tree": pts[name]} for name in names},
    }
    return _norm(out), mean


def model_reasons(d, base, M, M_final):
    """The reason sample and the free and final models' reason values."""
    S = pc.reason_sample(d, base)
    R_free = pc.Reasons(base, M, S)
    R_final = pc.Reasons(base, M_final, S)
    return S, {
        "free": (R_free, R_free.values(M["tables"])[0]),
        "final": (R_final, R_final.values(M_final["tables"])[0]),
    }


def tree_reasons(d, base, layout, kept, S_union):
    S = pc.reason_sample(
        d, {"union": list(layout.terms), "sel_users": base["sel_users"]}
    )
    if not (
        np.array_equal(S["users"], S_union["users"])
        and np.array_equal(S["lst"], S_union["lst"])
        and np.array_equal(S["is_pos"], S_union["is_pos"])
    ):
        raise ValueError("the reason samples of the two layouts differ")
    base_t = {"union": list(layout.terms), "union_layout": layout, "L_u": layout.L}
    M_t = {"kept": list(kept), "idx": np.arange(layout.L), "lay": layout}
    return pc.Reasons(base_t, M_t, S)


def _stack_values(values):
    return np.concatenate([np.asarray(v, dtype=np.float64) for v in values])


def _split_values(flat, nb):
    out, o = [], 0
    for n in nb:
        out.append(np.asarray(flat[o : o + n], dtype=np.float64))
        o += n
    return out


def check_outputs(d, layout, bg, xq, bands, reasons, model_ranks, files):
    """Check 5: every stored result follows from the stored tables and
    ranks."""
    doc = load_json(d / files[0])
    z = np.load(d / files[1])
    nb = [int(x) for x in z["nb"]]
    if nb != list(layout.nb):
        raise ValueError("the stored bins differ from the layout")
    bag_vals = [_split_values(row, nb) for row in z["bag_values"]]
    R_t = reasons["tree"]
    mean = [
        np.mean(np.stack([bv[j] for bv in bag_vals]), axis=0) for j in range(len(nb))
    ]
    val_t = R_t.values(layout.flat(mean))[0]
    if not np.array_equal(val_t, z["tree_reason_value"]):
        raise ValueError(
            "the stored reason values do not follow from the " "stored tables"
        )
    again, mean2 = summarize(
        layout, bag_vals, bg, xq, bands, val_t, reasons, z["ranks"], model_ranks
    )
    if not np.array_equal(_stack_values(mean2), z["mean_values"]):
        raise ValueError("the stored mean model is not the mean of the bags")
    for k in ("points", "falls", "reasons", "scores", "shapes"):
        if again[k] != doc[k]:
            raise ValueError(
                f"the stored results '{k}' do not follow from "
                f"the stored tables and ranks"
            )


def run(
    st,
    model_dir,
    work,
    say=print,
    n_verify=VERIFY_USERS,
    inputs=None,
    pilot=False,
    stages=None,
):
    """The audit (module docstring). st: a StageP13b2 whose loaders give the
    level, the item side, the CSR and the slots. stages: stage objects for
    check 6 (the smoke test). Returns the output doc, or the pilot's
    record."""
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME  # labels of the dense check's lines
    try:
        return _run(
            st, Path(model_dir), Path(work), say, n_verify, inputs, pilot, stages
        )
    finally:
        gr.WORK_NAME = prev
        _W.clear()


def _run(st, d, work, say, n_verify, inputs, pilot, stages):
    t0 = time.time()
    if PILOT_ROUNDS % CHECKPOINT_ROUNDS:
        raise ValueError(
            "PILOT_ROUNDS must be a multiple of "
            "CHECKPOINT_ROUNDS (the resume rule of run_bag)"
        )
    inputs = dict(inputs or {})
    inputs.update(check_markers(stages))
    base = st._base_context()
    M, free_ranks = pc.load_model(d, base)
    rec = load_json(d / pc.P13D_FILES[0])["model"]
    ptr = pc.check_pointer(d, rec)
    M_final = pc.load_p13d_model(d, base)[0] if rec["changed"] else M
    res_free = load_json(d / pc.OUT_FILES[0])
    res_final = load_json(d / ptr["files"]["results"])
    ranks_free = np.asarray(np.load(d / pc.OUT_FILES[1])["ranks"][0])
    ranks_final = np.asarray(np.load(d / ptr["files"]["arrays"])["ranks"][0])
    if not np.array_equal(ranks_free, free_ranks):
        raise ValueError("P13c's ranks of the free fit differ from " "gam_refit's")
    bands = {
        "free": band_table(res_free["bootstrap"]["bands"]),
        "final": band_table(res_final["bootstrap"]["bands"]),
    }
    say(
        f"{WORK_NAME}: free fit '{M['name']}' ({len(M['kept'])} terms), "
        f"level {M['level']}. Final model: {rec['fit_name']} "
        f"({'changed' if rec['changed'] else 'unchanged'}). "
        f"{WORKERS} workers, {N_BLOCKS} blocks. Saved work: {work}"
    )
    D = fit_rows(st, base, M, say)
    check_loss(D)
    _W.clear()
    _W.update(
        {
            k: D[k]
            for k in (
                "ranges",
                "bins_pos",
                "bins_neg",
                "off_pos",
                "off_neg",
                "pos_user",
                "n_pos",
                "ok_pos",
                "nb",
                "J",
            )
        }
    )
    layout = D["layout"]
    key = {
        "version": VERSION,
        "settings": settings(),
        "fit_code": fit_code(),
        "gam_refit": file_sha256(d / "gam_refit.json"),
        "p13b2_key": st._key(),
        "level": int(M["level"]),
        "edges": s13._array_sha(*layout.edges),
    }
    store = BagStore(work, key)
    t_load = time.time() - t0
    bags = []
    with Pinned(WORKERS) as pool:
        for bag in range(1 if pilot else BAGS):
            t_b = time.time()
            out = run_bag(
                pool, D, bag, store, say, stop_at=PILOT_ROUNDS if pilot else None
            )
            if pilot:
                return pilot_report(store, say, t_load, free_loss(d))
            bags.append(out)
            done_s = [b["seconds"] for b in bags]
            left = (BAGS - len(bags)) * float(np.mean(done_s))
            say(
                f"{WORK_NAME}: bag {bag + 1} of {BAGS}: {out['stop']} stop "
                f"after {out['rounds']:,} rounds, best round "
                f"{out['best_round']:,}, validation loss "
                f"{out['best_loss']:.8f}, {_clock(out['seconds'])} in all "
                f"({_clock(time.time() - t_b)} in this run). About "
                f"{_clock(left)} left for the other bags"
            )
    bg, nb, J = D["bg"], list(layout.nb), D["J"]
    del D
    _W.clear()
    bag_vals = [[np.asarray(x, dtype=np.float64) for x in b["tables"]] for b in bags]
    mean = [np.mean(np.stack([bv[j] for bv in bag_vals]), axis=0) for j in range(J)]
    tables = np.stack([layout.flat(mean)] + [layout.flat(bv) for bv in bag_vals])
    users, picks = base["sel_users"], base["sel_picks"]
    say(
        f"{WORK_NAME}: the bags are ready. Selection pass over "
        f"{len(tables)} configurations (the mean and each bag)"
    )
    pkey = {
        "version": VERSION,
        "stage": "pass",
        "p13a_params": s13._params(),
        "code": {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in (ct, cn, cd, gam, s13)
        },
    }
    t2 = time.time()
    with relabel_logs():
        ranks, scored = st._selection_pass(
            list(layout.terms), tables, users, picks, base["m_sel"], work / "pass", pkey
        )
    pass_secs = time.time() - t2
    dense = gr.verify_dense(
        st, dict(base, union=list(layout.terms)), tables, ranks, say, n_verify
    )
    st._cache.pop("space", None)
    st._cache.pop("sources", None)
    say(f"{WORK_NAME}: reason values on P11's selection rows")
    S, reasons = model_reasons(d, base, M, M_final)
    reasons["tree"] = tree_reasons(d, base, layout, M["kept"], S)
    if not np.array_equal(reasons["tree"].cand, reasons["free"][0].cand):
        raise ValueError(
            "the tree-GAM's reason candidates differ from the " "free fit's"
        )
    val_t = reasons["tree"].values(layout.flat(mean))[0]
    model_ranks = {"free": ranks_free, "final": ranks_final}
    xq = base["shape_q"]
    out, _ = summarize(
        layout, bag_vals, bg, xq, bands, val_t, reasons, ranks, model_ranks
    )
    out.update(
        _norm(
            {
                "version": VERSION,
                "declared": DECLARED,
                "protocol": "the docstring of pipeline/tree_audit.py",
                "settings": dict(
                    settings(),
                    pos_levels=len(POS_LEVELS),
                    neg_levels=len(NEG_LEVELS),
                    bags=BAGS,
                    workers=WORKERS,
                ),
                "model": {
                    "free": M["name"],
                    "kept": list(M["kept"]),
                    "level": int(M["level"]),
                    "arm": "-".join(ARM),
                    "m": s13b2.n_negatives(ARM, M["level"]),
                    "final": {k: rec[k] for k in ("fit_name", "tag", "changed")},
                },
                "bins": {
                    t.name: {"value_bins": n, "edges": len(e)}
                    for t, n, e in zip(layout.terms, layout.n_cont, layout.edges)
                },
                "bags": [
                    {
                        k: b[k]
                        for k in (
                            "bag",
                            "rounds",
                            "best_round",
                            "stop",
                            "best_loss",
                            "validation_listeners",
                            "seconds",
                        )
                    }
                    for b in bags
                ],
                "check": {
                    "offsets_equal_arm_block": True,
                    "loss_equals_softmax_block": True,
                    "bins_equal_step_layout": True,
                    "neutral_equals_p13a": True,
                    "bag_losses_again": True,
                    "dense": dense,
                    "markers_current": True,
                },
                "inputs": inputs,
                "seconds": {"loading": t_load, "pass_this_run": pass_secs},
            }
        )
    )
    files = OUT_FILES
    curves = np.full((len(bags), max(len(b["curve_val"]) for b in bags)), np.nan)
    for i, b in enumerate(bags):
        curves[i, : len(b["curve_val"])] = b["curve_val"]
    s13._atomic_savez(
        d / files[1],
        nb=np.asarray(nb, dtype=np.int64),
        edges=np.concatenate(layout.edges),
        n_edges=np.asarray([len(e) for e in layout.edges], dtype=np.int64),
        bag_values=np.stack([_stack_values(bv) for bv in bag_vals]),
        mean_values=_stack_values(mean),
        tables=tables,
        ranks=ranks.astype(np.int32),
        scored=scored.astype(np.int64),
        user_id=users,
        curve_val=curves,
        tree_reason_value=val_t,
        background=_stack_values(bg),
    )
    atomic_json_write(out, d / files[0])
    check_outputs(d, layout, bg, xq, bands, reasons, model_ranks, files)
    out["seconds"]["total_this_run"] = time.time() - t0
    atomic_json_write(out, d / files[0])
    write_report(out, d / "reports" / REPORT_NAME)
    t4 = out["scores"]
    say(
        f"{WORK_NAME}: tree-GAM {t4['tree']:.5f} against the final model "
        f"{t4['final']:.5f} ({t4['tree_minus_final']['decision']} SESOI). "
        f"Separating points: {len(out['points']['separating'])}. Signed "
        f"top-3 agreement with the final model "
        f"{out['reasons']['agreement']['final']:.1%}. Done in "
        f"{_clock(time.time() - t0)}"
    )
    return out


# Report


def _f(x, fmt="+.3f"):
    return "" if x is None else format(x, fmt)


def _iv(dd, fmt="+.5f"):
    if dd is None or dd.get("mean") is None:
        return "n/a"
    if dd.get("lo") is None:
        return format(dd["mean"], fmt)
    return (
        f"{format(dd['mean'], fmt)} ({format(dd['lo'], fmt)} to "
        f"{format(dd['hi'], fmt)})"
    )


def write_report(out, path):
    s, m, t4 = out["settings"], out["model"], out["scores"]
    pts, t2, t3 = out["points"], out["falls"], out["reasons"]
    fin = m["final"]
    L = [
        "# The tree-GAM audit",
        "",
        f"- Free fit: '{m['free']}' ({len(m['kept'])} terms), arm "
        f"{m['arm']}, level {m['level']} (m = {m['m']}). Final model: "
        f"{fin['fit_name']} ({'changed' if fin['changed'] else 'unchanged'}"
        f"). Protocol: {out['protocol']}, declared {out['declared']}.",
        f"- Tree-GAM: cyclic boosting of depth-1 trees, learning rate "
        f"{s['learning_rate']}, at most {s['max_rounds']:,} rounds, early "
        f"stop after {s['patience']} rounds without a relative gain of "
        f"{s['tolerance']:g}, outer bags: {s['bags']}, each with "
        f"{s['validation']:.0%} validation listeners, min_samples_leaf "
        f"{s['min_samples_leaf']}, min_hessian {s['min_hessian']:g}. The "
        f"model is the mean of the bags.",
        "- Value bins: "
        + ", ".join(f"{k} {v['value_bins']}" for k, v in out["bins"].items())
        + ".",
        f"- Checks: offsets equal arm_block's, loss and gradient equal "
        f"gam.softmax_block's, the step layout equals the bins on every fit "
        f"row, the neutral background equals P13a's, each bag's best loss "
        f"again, {out['check']['dense']['listeners']} listeners equal the "
        f"dense reference ({out['check']['dense']['near_ties']} near ties), "
        f"markers current, and the results follow from the stored outputs.",
        "",
        "## Bags",
        "",
        "| bag | rounds | best round | stop | validation loss | time |",
        "|---:|---:|---:|---|---:|---:|",
    ]
    for b in out["bags"]:
        L.append(
            f"| {b['bag'] + 1} | {b['rounds']:,} | {b['best_round']:,} "
            f"| {b['stop']} | {b['best_loss']:.8f} | "
            f"{_clock(b['seconds'])} |"
        )
    L += [
        "",
        "## T4: accuracy",
        "",
        f"Selection NDCG@100: tree-GAM {t4['tree']:.5f}, free fit "
        f"{t4['free']:.5f}, final model {t4['final']:.5f}. Bags: "
        f"{min(t4['bags']):.5f} to {max(t4['bags']):.5f}. SESOI "
        f"{t4['sesoi']:.5f}.",
        "",
        f"- tree-GAM minus the final model: "
        f"{_iv(t4['tree_minus_final'])}: "
        f"{t4['tree_minus_final']['decision']} SESOI.",
        f"- tree-GAM minus the free fit: {_iv(t4['tree_minus_free'])}: "
        f"{t4['tree_minus_free']['decision']} SESOI.",
        f"- final model minus the free fit: " f"{_iv(t4['final_minus_free'])}.",
        "",
        "## T1: shape points",
        "",
        f"Centered on the level's p_pop draws, in log-odds points. A point "
        f"separates when it differs by more than TAU ({TAU}) and lies "
        f"outside the model's 95% listener band.",
        "",
        "| term | point | tree-GAM (bags) | free (band) | diff | final "
        "(band) | diff |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in pts["rows"]:
        cells = [
            r["term"],
            r["point"],
            f"{r['tree']:+.3f} ({r['tree_bags'][0]:+.3f} to "
            f"{r['tree_bags'][1]:+.3f})",
        ]
        for which in ("free", "final"):
            c = r[which]
            if c is None:
                cells += ["", ""]
                continue
            mark = " (separates)" if c["separates"] else ""
            cells += [
                f"{c['value']:+.3f} ({c['lo']:+.3f} to {c['hi']:+.3f})",
                f"{c['diff']:+.3f}{mark}",
            ]
        L.append("| " + " | ".join(cells) + " |")
    L += [
        "",
        "Separating points: "
        + (", ".join(pts["separating"]) if pts["separating"] else "none")
        + ".",
        "",
        "## T2: falls",
        "",
        "Net change of each term from p5 to p95. A term falls (rises) "
        "under the tree-GAM when the change is below (above) 0 in every "
        "bag.",
        "",
        "| term | tree-GAM | bags | verdict | free | final |",
        "|---|---:|---:|---|---:|---:|",
    ]
    for name, r in t2.items():
        verdict = "falls" if r["falls"] else "rises" if r["rises"] else "neither"
        L.append(
            f"| {name} | {r['net']:+.3f} | {min(r['bag_net']):+.3f} to "
            f"{max(r['bag_net']):+.3f} | {verdict} | "
            f"{_f(r['free_net'])} | {_f(r['final_net'])} |"
        )
    L += [
        "",
        "## T3: reasons",
        "",
        f"{t3['picks']:,} selection val picks, background 1 (the "
        f"listener's own p_pop draws). Signed top-3 agreement of the "
        f"tree-GAM with the free fit {t3['agreement']['free']:.1%} "
        f"({t3['level_free']}), with the final model "
        f"{t3['agreement']['final']:.1%} ({t3['level_final']}). Reference "
        f"levels: {ROBUST_LEVEL:.1%} (bounded members) and "
        f"{WEAK_LEVEL:.1%} (level-4096 fits).",
        "",
    ]
    if t3["artist_differs"]:
        L.append(
            f"Artist reason more than TAU apart: "
            f"{t3['artist_differs']['free']:.1%} of its candidate picks "
            f"(free fit), {t3['artist_differs']['final']:.1%} (final "
            f"model)."
        )
        L.append("")
    L += [
        "| reason | top 3 and positive: tree-GAM | free | final |",
        "|---|---:|---:|---:|",
    ]
    for g, name in enumerate(t3["groups"]):
        L.append(
            f"| {name} | {t3['top3_positive']['tree'][g]:.1%} | "
            f"{t3['top3_positive']['free'][g]:.1%} | "
            f"{t3['top3_positive']['final'][g]:.1%} |"
        )
    L += [
        "",
        "## Decisions (thesis text only, protocol)",
        "",
        f"- T4: the tree-GAM is {t4['tree_minus_final']['decision']} SESOI "
        f"of the final model.",
        f"- T1: {len(pts['separating'])} separating points.",
        "- T2: "
        + ", ".join(
            f"{k} {'falls' if v['falls'] else 'rises' if v['rises'] else 'neither'}"
            for k, v in t2.items()
        )
        + ".",
        f"- T3: {t3['level_final']} (agreement with the final model "
        f"{t3['agreement']['final']:.1%}).",
    ]
    atomic_text_write(_text("\n".join(L)) + "\n", path)


def main(argv=None):
    global BAGS
    parser = argparse.ArgumentParser(
        description="The tree-GAM audit (protocol in the " "module docstring)"
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help=f"run {PILOT_ROUNDS} rounds of bag 1, print the "
        f"time per round and the projected totals, and "
        f"stop",
    )
    parser.add_argument(
        "--bags",
        type=int,
        default=BAGS,
        help=f"the number of outer bags (default {BAGS}). "
        f"Finished bags count for any number",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.bags <= 64:
        parser.error("--bags must be 1 to 64")
    BAGS = args.bags
    prev = gr.WORK_NAME
    gr.WORK_NAME = WORK_NAME
    gr.setup_logging(LOG_PATH)
    gr.WORK_NAME = prev
    say = log.info
    try:
        say(f"{WORK_NAME}: started, process id {os.getpid()}, log file " f"{LOG_PATH}")
        run(
            s13b2.StageP13b2(),
            MODEL_DIR,
            SCRATCH_DIR / WORK_NAME,
            say=say,
            inputs=provenance(MODEL_DIR),
            pilot=args.pilot,
        )
    except KeyboardInterrupt:
        log.info(
            f"{WORK_NAME}: stopped by Ctrl+C. Run the same command "
            f"again: it continues from the last checkpoint."
        )
        raise SystemExit(130)
    except Exception:
        log.exception(f"{WORK_NAME}: stopped by an error")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
