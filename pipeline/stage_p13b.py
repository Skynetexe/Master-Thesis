"""Stage P13b, part 1: the experiment arms of the transparent model that
run on the CPU.

The experiments follow the design of P13 (Chapter 4 of the thesis), with
the choices that the P13a results forced. P13a fixed the knots, the
audit, the mainstreaminess constants and the base penalty path. This
stage reads them and changes none of them. Every rule below was declared
before its step ran.

Steps:
1. Path extension. P13a's best penalty was the weakest on its path
   (lambda 1e-5), so the one-standard-error rule had no simpler model to
   choose. The base model is fit at the weights of EXT_LAMBDAS (down to
   1e-8, then 0, the ridge alone), warm-started from P13a's weakest fit.
   The one-standard-error rule then runs over all 19 weights, strongest
   penalty first. P13a's stored ranks serve its 12 weights. The chosen
   weight is lambda_base.
2. The factorial at lambda_base: 16 cells over four factors,
   - monotone: every affinity term (the seven base concepts, and
     work_affinity when present) is non-decreasing, and its zero state
     stays at or below the height of its first knot (gam_arms). This
     set was chosen after P13a: P13a's artist_affinity shape
     falls from +2.4 to -5.6 as a listener plays the artist more. The
     deviation curve and the tilt stay free: in a monotone cell with the
     curve, the audio fit of a fuzzy_dur track is the bounded reference
     curve plus a free deviation.
   - work_affinity in the score,
   - the fuzzy_dur deviation curve of audio fit,
   - the popularity tilt by mainstreaminess.
   The base cell (free shapes, no work_affinity, curve and tilt) is
   P13a's model, so it reuses the base fit. Every other cell starts from
   the base tables.
3. The loss and sampler arms at lambda_base, on the base terms:
   {uniform, p_pop, mixture} negatives by {no correction, form A, form B}
   with 16 negatives each (P11 drew 16 per kind, so the mixture arm takes
   the first 8 of each kind), and BPR on all 32. Form A subtracts
   log(m q(j)) from each negative (Blanc & Rendle 2018). Form B subtracts
   it from every candidate, the positive included (the sampling of
   alternatives of McFadden 1978, as Train 2009 states it, and the
   correction of Yi et al. 2019). A positive that the arm's proposal
   cannot draw has no finite form B term and gets weight 0. For uniform
   draws form B adds one constant per listener, so it equals the
   uncorrected arm, which this stage checks.
4. Decisions, declared here. SESOI is 2 percent of the base cell's
   score (the smallest effect of interest).
   The factorial gives each factor a main effect: the mean, over the
   selection listeners, of the average score of the 8 cells with the
   factor minus that of the 8 cells without it.
   - work_affinity, the deviation curve and the tilt stay in the model
     only if their main effect exceeds SESOI.
   - The monotone bounds stay unless they cost more than SESOI.
   The loss arms decide nothing yet: the loss waits for the large-m
   reference of P13b part 2.
5. The final path: the chosen cell over the 19 weights, with the one-
   standard-error rule. When the chosen cell is the base cell, step 1
   is the final path.

Selection: full-catalog NDCG@100 of the 20,000 selection listeners, as in
P13a (catalog.ranks_cut(), ranks exact up to 100). One scorer serves
every configuration: the terms of all cells, with zero tables for the
terms that a cell leaves out, which then add exactly 0.

Resume: every fit and every selection block is saved under
SCRATCH_DIR/p13b, keyed by the parameters, the upstream outputs (P13a's
included) and the code. A step whose fits are all saved builds no
design.

The selection passes run P13a's code (StageP13a._selection_pass). Its
log lines start with "p13a". A log filter renames them to "p13b" while
this stage runs a pass. The probe estimate here assumes a smaller
parallel speedup than P13a's: P13a's pass on the real data ran about 3.6
times faster on 16 workers than in one process, on 8 physical cores,
because the gathers over the item side wait on memory.

verify():
- P13a's knots, terms and layout hold: the context rebuilds them from
  the training rows and compares them with P13a's model file.
- every stored fit is a stationary point: the projected gradient at the
  stored parameters, recomputed, stays below KKT_TOL, the objective
  equals its stored value, the tables equal M theta, and a monotone cell
  keeps its order.
- the arm offsets equal an independent computation from P11's values on
  sampled rows, and the uniform arms with and without form B agree.
- sampled selection ranks equal the dense reference, censored at 101.
- two passes give the same ranks for a configuration they share: the
  base cell and the base path at lambda_base, and the chosen cell and
  the final path at lambda_base.
- lambda_base, the factorial decisions and the final choice follow from
  the stored ranks.
"""

import contextlib
import dataclasses
import itertools
import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from . import artists as ar
from . import catalog as ct
from . import concept_data as cd
from . import concepts as cn
from . import fusion_pairs as fp
from . import gam
from . import gam_arms
from . import gam_audit as ga
from . import selection as sel
from . import stage_p13a as s13
from .als import Progress, _clock, threadpool_limits
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    file_sha256,
    load_json,
    log,
    marker_path,
)
from .config import MODEL_DIR, SCRATCH_DIR

VERSION = "p13b-v1"
EXT_LAMBDAS = (10.0**-5.5, 1e-6, 10.0**-6.5, 1e-7, 10.0**-7.5, 1e-8, 0.0)
SESOI = 0.02
FACTORS = ("monotone", "work_affinity", "tier_curve", "tilt")
BASE_CELL = (0, 0, 1, 1)
CELLS = tuple(itertools.product((0, 1), repeat=4))
AFFINITY = s13.BASE_CONCEPTS + ("work_affinity",)
SAMPLERS = ("uniform", "p_pop", "mixture")
CORRECTIONS = ("none", "A", "B")
ARMS = tuple((s, c) for s in SAMPLERS for c in CORRECTIONS) + (("mix32", "bpr"),)
NEG_PER_KIND = 16
SOLVER = dict(s13.SOLVER, bounded_newton_steps=200)
VERIFY_OFFSET_USERS = 300
MONOTONE_TOL = 1e-12  # rounding of M theta, not a break
PASS_SPEEDUP_PER_CORE = 0.45  # P13a's pass: 3.6 on 8 physical cores

_CTX = {}


class _Relabel(logging.Filter):
    """Renames the log lines of P13a's selection pass while P13b runs it,
    and points the advice of its swap warning at this stage."""

    def filter(self, record):
        msg = record.msg
        if isinstance(msg, str) and msg.startswith("p13a"):
            record.msg = "p13b" + msg[4:].replace("--only p13a_fit", "--only p13b_arms")
        return True


@contextlib.contextmanager
def _relabel_logs():
    flt = _Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


def _params():
    return {
        "version": VERSION,
        "ext_lambdas": list(EXT_LAMBDAS),
        "sesoi": SESOI,
        "factors": list(FACTORS),
        "base_cell": list(BASE_CELL),
        "affinity": list(AFFINITY),
        "arms": [list(a) for a in ARMS],
        "neg_per_kind": NEG_PER_KIND,
        "ridge": s13.RIDGE,
        "solver": SOLVER,
        "rank_cut": s13.RANK_CUT,
        "kkt_tol": s13.KKT_TOL,
    }


def cell_label(cell):
    return "m{}w{}t{}p{}".format(*cell)


def arm_label(arm):
    return "base" if arm is None else f"{arm[0]}-{arm[1]}"


def fit_id(cell, arm, lam):
    return f"{cell_label(cell)}|{arm_label(arm)}|{float(lam)!r}"


def neg_cols(sampler, per_kind=None):
    """Columns of a listener's negatives that an arm uses. P11 stores the
    uniform draws first, then the p_pop draws, each in draw order."""
    k = NEG_PER_KIND if per_kind is None else int(per_kind)
    if sampler == "uniform":
        return np.arange(k)
    if sampler == "p_pop":
        return np.arange(k, 2 * k)
    if sampler == "mixture":
        return np.concatenate([np.arange(k // 2), np.arange(k, k + k // 2)])
    if sampler == "mix32":
        return np.arange(2 * k)
    raise KeyError(sampler)


def arm_q(q_u, q_p, sampler):
    """The proposal density of an arm at each row: q_u = 1 / n_allowed,
    q_p = p_pop / pop_allowed, and the mixture of the two with equal
    weights (the balance heuristic of an equal stratified draw)."""
    if sampler == "uniform":
        return q_u
    if sampler == "p_pop":
        return q_p
    if sampler in ("mixture", "mix32"):
        return 0.5 * q_u + 0.5 * q_p
    raise KeyError(sampler)


def arm_block(blk, arm):
    """A fit block as one loss arm sees it: its negatives, its offsets and
    its positive weights. arm None: the base (all 32 negatives, form A)."""
    if arm is None:
        return blk
    sampler, corr = arm
    cols = neg_cols(sampler, blk["neg_rows"].shape[1] // 2)
    out = dict(blk, neg_rows=blk["neg_rows"][:, cols])
    if corr == "bpr":
        return out
    n = len(blk["q_u"])
    if corr == "none":
        out["offset"] = np.zeros(n, dtype=np.float64)
        return out
    m = len(cols)
    with np.errstate(divide="ignore"):
        off = -np.log(m * arm_q(blk["q_u"], blk["q_p"], sampler))
    is_pos = np.zeros(n, dtype=bool)
    is_pos[blk["pos_rows"]] = True
    if corr == "A":
        out["offset"] = np.where(is_pos | ~np.isfinite(off), 0.0, off)
        return out
    if corr == "B":
        ok = np.isfinite(off[blk["pos_rows"]])
        out["offset"] = np.where(np.isfinite(off), off, 0.0)
        out["pos_weight"] = np.where(ok, blk["pos_weight"], 0.0)
        return out
    raise KeyError(corr)


# Workers


def _task_loss(task):
    b, T, D, arm = task
    blk = arm_block(_CTX["blocks"][b], arm)
    if arm is not None and arm[1] == "bpr":
        return gam_arms.bpr_block_loss_grad(T, D, blk, _CTX["L"])
    return gam.block_loss_grad(T, D, blk, _CTX["L"])


def _task_hess(task):
    b, T, D, arm = task
    blk = arm_block(_CTX["blocks"][b], arm)
    with threadpool_limits(1):
        if arm is not None and arm[1] == "bpr":
            return gam_arms.bpr_block_hessian(T, D, blk, _CTX["L"])
        return gam.block_hessian(T, D, blk, _CTX["L"])


# Fit store


class _FitStore:
    """The finished fits, saved one by one, so a stopped run continues
    with the next fit."""

    def __init__(self, path, key_sha):
        self.path = Path(path)
        state = load_json(self.path) or {}
        if state.get("k") != key_sha:
            state = {"k": key_sha, "fits": {}}
        self.state = state

    def __contains__(self, fid):
        return fid in self.state["fits"]

    def get(self, fid):
        rec = dict(self.state["fits"][fid])
        rec["theta"] = np.asarray(rec["theta"], dtype=np.float64)
        return rec

    def put(self, fid, res):
        self.state["fits"][fid] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in res.items()
        }
        atomic_json_write(self.state, self.path)


def main_effects(scores, cells, factors=FACTORS):
    """Main effects and two-factor interactions of a two-level factorial
    from per-listener scores (C, n): with codes +1 (on) and -1 (off),
    effect = (2 / C) * sum over cells of code * score, per listener. For
    a main effect this is the mean of the cells with the factor minus the
    mean of the cells without it. Returns {name: {mean, se, n}}."""
    s = np.asarray(scores, dtype=np.float64)
    C, n = s.shape
    codes = np.array(
        [[1.0 if c[i] else -1.0 for i in range(len(factors))] for c in cells]
    )
    out = {}
    terms = [(f,) for f in range(len(factors))] + list(
        itertools.combinations(range(len(factors)), 2)
    )
    for tt in terms:
        code = np.prod(codes[:, list(tt)], axis=1)
        e = (2.0 / C) * (code @ s)
        name = " x ".join(factors[i] for i in tt)
        out[name] = {
            "mean": float(e.mean()),
            "se": float(e.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
            "n": int(n),
        }
    return out


def decide(effects, base_score, sesoi=SESOI):
    """The declared rules of step 4. Returns the chosen cell and the
    record of each decision."""
    thr = sesoi * base_score
    rec = {"sesoi_abs": thr}
    mono = effects["monotone"]["mean"] > -thr
    keep = {
        f: effects[f]["mean"] > thr for f in ("work_affinity", "tier_curve", "tilt")
    }
    rec.update({"monotone": bool(mono), **{f: bool(v) for f, v in keep.items()}})
    cell = (
        int(mono),
        int(keep["work_affinity"]),
        int(keep["tier_curve"]),
        int(keep["tilt"]),
    )
    return cell, rec


# Stage


class StageP13b(s13.StageP13a):
    name = "p13b_arms"
    upstream = s13.StageP13a.upstream + ["p13a_fit"]

    def outputs(self):
        return {
            "model": MODEL_DIR / "p13b_model.json",
            "arrays": MODEL_DIR / "p13b_arrays.npz",
            "report": MODEL_DIR / "reports" / "p13b_report.md",
        }

    def _modules(self):
        return (gam, gam_arms, ga, ct, cn, cd, ar, sel, s13)

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        for mod in self._modules():
            inputs[Path(mod.__file__).name] = file_sha256(Path(mod.__file__))
        return inputs

    def _code(self):
        return {
            Path(m.__file__).name: file_sha256(Path(m.__file__))
            for m in self._modules()
        } | {Path(__file__).name: file_sha256(Path(__file__))}

    def _key(self):
        return {
            "params": _params(),
            "p13a_params": s13._params(),
            "upstream": self._upstream_outputs(),
            "code": self._code(),
        }

    # The P13a outputs. test_p13b.py runs P13a in its own folder first.
    def _p13a(self):
        if "p13a" not in self._cache:
            doc = load_json(MODEL_DIR / "p13a_model.json")
            check(doc is not None, "p13a_model.json is missing: run " "p13a_fit first")
            z = np.load(MODEL_DIR / "p13a_arrays.npz")
            self._cache["p13a"] = (doc, {k: z[k] for k in z.files})
        return self._cache["p13a"]

    # Context

    def _context(self):
        """Everything the steps share, derived from P13a's outputs and the
        training rows, with the checks that tie them to P13a."""
        if "ctx" in self._cache:
            return self._cache["ctx"]
        a_doc, a_arr = self._p13a()
        check(
            a_doc["params"] == s13._params(),
            "p13a_model.json was built with other P13a parameters",
        )
        inp = self._inputs()
        check(
            a_doc["pairs_sha256"] == inp["pairs_sha"]
            and a_doc["features_sha256"] == inp["features_sha"],
            "P13a was fit on other pairs or features",
        )
        f, rows, kind = inp["f"], inp["rows"], inp["kind"]
        knots = s13.compute_knots(f, rows, kind)
        check(
            {k: list(v) for k, v in knots.items()} == a_doc["knots"],
            "the knots differ from P13a's",
        )
        base = s13.base_terms(knots)
        check(
            [t.to_json() for t in base] == a_doc["terms"],
            "the base terms differ from P13a's",
        )
        pos = kind == fp.KIND_POSITIVE
        v, h = s13.concept_column(f, rows, "work_affinity")
        shape = h & (v != 0.0)
        wk = gam.knot_rule(
            v[shape & pos], v[shape & ~pos], s13.KNOT_POS_LEVELS, s13.KNOT_NEG_LEVELS
        )
        union = base + [
            gam.Term(
                "work_affinity",
                "work_affinity",
                wk,
                zero_state=True,
                missing_state=True,
            )
        ]
        slot, w = s13.encode_rows(union, f, rows)
        ref_mask = kind == fp.KIND_POP
        means_ref, data_means, dev_means = s13.reference_means(union, slot, w, ref_mask)
        n_zero = {
            t.name: int((slot[k] == t.K + gam.ZERO).sum()) for k, t in enumerate(union)
        }
        n_miss = {
            t.name: int((slot[k] == t.K + gam.MISS).sum()) for k, t in enumerate(union)
        }
        shape_q = s13.shape_quantiles(union, f, rows, kind, slot)
        del slot, w
        audit = a_doc["audit"]["missing"]["concepts"]
        neutral = list(a_doc["layout"]["neutral"])
        if audit["work_affinity"]["no_signal"]:
            neutral.append("work_affinity")
        base_lay = gam.Layout(
            base,
            n_zero,
            n_miss,
            [n for n in neutral if n != "work_affinity"],
            data_means,
            dev_means,
        )
        check(
            base_lay.fingerprint() == a_doc["layout"]["fingerprint"],
            "the base layout differs from P13a's",
        )
        union_lay = gam.Layout(union, n_zero, n_miss, neutral, data_means, dev_means)
        # Mainstreaminess and the selection listeners, as P13a stored them.
        m_consts = a_doc["mainstreaminess"]
        m_users, m_raw = a_arr["m_users"], a_arr["m_raw"]
        tr_users = inp["p"]["user_id"][inp["tr"]]
        at = np.searchsorted(m_users, tr_users)
        check(
            np.array_equal(m_users[at], tr_users),
            "a training listener has no mainstreaminess in P13a",
        )
        m_row = gam.mainstreaminess(m_raw[at], m_consts)[inp["row_user"]]
        sel_users, sel_picks = a_arr["sel_users"], a_arr["sel_picks"]
        m_sel = gam.mainstreaminess(
            m_raw[np.searchsorted(m_users, sel_users)], m_consts
        )
        # The proposal densities of every row and the order of the draws.
        n_slots = inp["n_slots"]
        item_ids, p_pop = self._proposals()
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        tr = inp["tr"]
        ru = inp["row_user"]
        q_u = 1.0 / inp["p"]["n_allowed"][tr][ru].astype(np.float64)
        q_p = p_pop_slot[inp["item"]] / inp["p"]["pop_allowed"][tr][ru]
        pos_no_pop = int(np.count_nonzero(q_p[pos] == 0.0))
        m_neg = inp["m_neg"]
        check(
            m_neg == 2 * NEG_PER_KIND,
            f"P11 drew {m_neg // 2} negatives per kind, this stage "
            f"expects {NEG_PER_KIND}",
        )
        starts = np.concatenate(([0], np.cumsum(inp["lens"])[:-1]))
        first_neg = starts + inp["n_pos"]
        k_neg = kind[first_neg[:, None] + np.arange(m_neg)[None, :]]
        check(
            bool(
                (k_neg[:, :NEG_PER_KIND] == fp.KIND_UNIFORM).all()
                and (k_neg[:, NEG_PER_KIND:] == fp.KIND_POP).all()
            ),
            "the negatives are not stored as 16 uniform, then 16 p_pop " "draws",
        )
        L_u = union_lay.L
        union_off = {t.name: int(o) for t, o in zip(union, union_lay.offsets)}
        ctx = {
            "a_doc": a_doc,
            "a_arr": a_arr,
            "knots": knots,
            "base": base,
            "union": union,
            "union_layout": union_lay,
            "L_u": L_u,
            "union_off": union_off,
            "n_zero": n_zero,
            "n_miss": n_miss,
            "neutral": neutral,
            "means_ref": means_ref,
            "data_means": data_means,
            "dev_means": dev_means,
            "shape_q": shape_q,
            "m_row": m_row,
            "m_consts": m_consts,
            "sel_users": sel_users,
            "sel_picks": sel_picks,
            "m_sel": m_sel,
            "q_u": q_u,
            "q_p": q_p,
            "pos_no_pop": pos_no_pop,
            "work_knots": wk,
            "path_lambdas": list(s13.LAMBDAS) + list(EXT_LAMBDAS),
        }
        self._cache["ctx"] = ctx
        return ctx

    def _cell_layout(self, ctx, cell):
        """The layout of one cell and the positions of its table entries
        in the union tables."""
        mono, wk, tier, tilt = cell
        terms = []
        for t in ctx["union"]:
            if t.name == "work_affinity" and not wk:
                continue
            if t.name == "audio_fit_fuzzy_dur" and not tier:
                continue
            if t.name == "popularity_tilt" and not tilt:
                continue
            if (
                mono
                and t.source in AFFINITY
                and t.multiplier is None
                and not t.centered
                and t.tiers is None
            ):
                t = dataclasses.replace(t, monotone=1)
            terms.append(t)
        names = [t.name for t in terms]
        lay = gam.Layout(
            terms,
            {n: ctx["n_zero"][n] for n in names},
            {n: ctx["n_miss"][n] for n in names},
            [n for n in ctx["neutral"] if n in names],
            {n: v for n, v in ctx["data_means"].items() if n in names},
            {n: v for n, v in ctx["dev_means"].items() if n in names},
        )
        if mono:
            gam_arms.order_zero_states(lay)
        idx = np.concatenate(
            [ctx["union_off"][t.name] + np.arange(t.size) for t in terms]
        )
        return lay, idx

    def _centered(self, ctx, lay, idx, theta):
        """(raw tables, centered tables, constants, centered union
        tables) of one fit, centered on the p_pop negatives."""
        T = lay.tables(theta)
        Tc, consts = gam.center(
            T, lay, {t.name: ctx["means_ref"][t.name] for t in lay.terms}
        )
        Tu = np.zeros(ctx["L_u"], dtype=np.float64)
        Tu[idx] = Tc
        return T, Tc, consts, Tu

    # Design

    @contextlib.contextmanager
    def _design(self, ctx):
        """The union design in fit blocks and a worker pool that reads
        it. The design is freed on exit, before a selection pass."""
        for k in ("space", "sources"):
            self._cache.pop(k, None)
        inp = self._inputs()
        slot, w = s13.encode_rows(ctx["union"], inp["f"], inp["rows"])
        blocks = s13.make_blocks(
            ctx["union"],
            ctx["union_layout"],
            slot,
            w,
            ctx["m_row"],
            inp["offset"],
            inp["lens"],
            inp["n_pos"],
            s13.FIT_BLOCKS,
            inp["m_neg"],
        )
        lens = inp["lens"]
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        for blk in blocks:
            lo, hi = blk["users"]
            r0, r1 = int(starts[lo]), int(starts[hi - 1] + lens[hi - 1])
            blk["q_u"] = ctx["q_u"][r0:r1]
            blk["q_p"] = ctx["q_p"][r0:r1]
        _CTX.update(blocks=blocks, L=ctx["L_u"])
        try:
            with s13._Pool(s13.FIT_WORKERS) as pool:
                yield pool
        finally:
            _CTX.clear()
            del blocks, slot, w

    @contextlib.contextmanager
    def _design_for(self, ctx, store, fids):
        """The design and pool of _design(), or None when the store holds
        every fit in fids already."""
        if all(fid in store for fid in fids):
            yield None
            return
        with self._design(ctx) as pool:
            yield pool

    def _drop_inputs(self):
        """Frees the training rows before a selection pass, as P13a does.
        _inputs() loads them again when a later step needs them."""
        for k in ("inputs", "pairs", "features", "space", "sources"):
            self._cache.pop(k, None)

    def _objective(self, pool, ctx, lay, idx, arm, lam):
        L_u = ctx["L_u"]
        terms_u, off_u = ctx["union"], ctx["union_layout"].offsets
        n_blocks = len(_CTX["blocks"])

        def lift(T):
            Tu = np.zeros(L_u, dtype=np.float64)
            Tu[idx] = T
            return Tu, gam.diff_table(Tu, terms_u, off_u)

        def loss_grad(T, D):
            Tu, Du = lift(T)
            loss = 0.0
            g = np.zeros(L_u, dtype=np.float64)
            for l_b, g_b in pool.map(
                _task_loss, [(b, Tu, Du, arm) for b in range(n_blocks)]
            ):
                loss += l_b
                g += g_b
            return loss, g[idx]

        def hess(T, D):
            Tu, Du = lift(T)
            H = np.zeros((L_u, L_u), dtype=np.float64)
            for h in pool.map(_task_hess, [(b, Tu, Du, arm) for b in range(n_blocks)]):
                H += h
            return H[np.ix_(idx, idx)]

        return gam.Objective(lay, loss_grad, lam, s13.RIDGE, hess)

    def _fit(self, pool, ctx, store, cell, arm, lam, theta0, label):
        fid = fit_id(cell, arm, lam)
        if fid in store:
            return store.get(fid)
        check(pool is not None, f"fit {fid} is not stored and no design " f"is loaded")
        lay, idx = self._cell_layout(ctx, cell)
        obj = self._objective(pool, ctx, lay, idx, arm, lam)
        t0 = time.time()
        if any(lo is not None or hi is not None for lo, hi in lay.bounds):
            res = gam_arms.fit_bounded(obj, theta0, SOLVER)
        else:
            res = gam.fit(obj, theta0, SOLVER)
        res["seconds"] = time.time() - t0
        store.put(fid, res)
        log.info(
            "p13b: %s, cell %s, arm %s, lambda %.3g: objective %.8f, "
            "largest projected gradient %.1e, %s, %s",
            label,
            cell_label(cell),
            arm_label(arm),
            lam,
            res["objective"],
            res["proj_grad_max"],
            res["message"],
            _clock(res["seconds"]),
        )
        return store.get(fid)

    def _pass(self, ctx, label, tables_u, work, key):
        """A selection pass over configurations given as union tables:
        (ranks, slots scored per listener, seconds)."""
        log.info(
            "p13b: selection pass '%s' over %d configurations", label, len(tables_u)
        )
        self._drop_inputs()
        t0 = time.time()
        with _relabel_logs():
            ranks, scored = self._selection_pass(
                ctx["union"],
                np.stack(tables_u),
                ctx["sel_users"],
                ctx["sel_picks"],
                ctx["m_sel"],
                work / f"pass_{label}",
                key,
            )
        secs = time.time() - t0
        self._cache.pop("space", None)
        self._cache.pop("sources", None)
        return ranks, scored, secs

    def _probe(self, tasks, pending, workers, n_slots):
        """P13a's probe (StageP13a._probe) with a lower estimate of the
        parallel speedup: PASS_SPEEDUP_PER_CORE per physical core, at most
        one per worker."""
        take = np.unique(
            np.round(
                np.linspace(0, len(pending) - 1, min(s13.PROBE_USERS, len(pending)))
            ).astype(np.int64)
        )
        ks = [int(tasks[pending[int(i)]][0]) for i in take]
        out, secs = {}, []
        with threadpool_limits(1):
            for n, k in enumerate(ks, start=1):
                t0 = time.time()
                r, s = s13._selection_one(s13._CTX, k)
                secs.append(time.time() - t0)
                out[k] = (r.copy(), int(s))
                log.info(
                    "p13b: probe listener %d of %d: %.2f s, %.0f%% of "
                    "the catalog scored",
                    n,
                    len(ks),
                    secs[-1],
                    100.0 * s / n_slots,
                )
        n_left = sum(tasks[i][1] - tasks[i][0] for i in pending)
        par = max(
            1.0, min(float(workers), PASS_SPEEDUP_PER_CORE * s13._physical_cores())
        )
        est = float(np.mean(secs)) * n_left / par
        log.info(
            "p13b: probe mean %.2f s per listener in one process. "
            "Rough estimate for %s listeners on %d worker%s, with a "
            "parallel speedup of %.1f: %s. The progress lines below "
            "measure the real rate",
            np.mean(secs),
            f"{n_left:,}",
            workers,
            "" if workers == 1 else "s",
            par,
            _clock(est),
        )
        return out

    def run(self, params, files):
        t_start = time.time()
        check(s13.RANK_CUT >= 100, "RANK_CUT must be at least 100")
        outputs = self.outputs()
        work = SCRATCH_DIR / "p13b"
        work.mkdir(parents=True, exist_ok=True)
        key = self._key()
        ctx = self._context()
        a_doc, a_arr = ctx["a_doc"], ctx["a_arr"]
        store = _FitStore(work / "fits.json", s13._sha(key))
        base_lay, base_idx = self._cell_layout(ctx, BASE_CELL)
        lams = ctx["path_lambdas"]
        n12 = len(s13.LAMBDAS)
        passes = {}

        # Step 1: the path extension, warm-started from P13a's weakest fit.
        theta = np.asarray(a_doc["path"][-1]["theta"], dtype=np.float64)
        fids = [fit_id(BASE_CELL, None, lam) for lam in EXT_LAMBDAS]
        ext = []
        with self._design_for(ctx, store, fids) as pool:
            for lam in EXT_LAMBDAS:
                rec = self._fit(
                    pool, ctx, store, BASE_CELL, None, lam, theta, "path extension"
                )
                theta = rec["theta"]
                ext.append(rec)
        ext_tables = [
            self._centered(ctx, base_lay, base_idx, r["theta"])[3] for r in ext
        ]
        ranks_ext, scored_ext, secs = self._pass(ctx, "ext", ext_tables, work, key)
        passes["ext"] = (scored_ext, secs)
        ranks_19 = np.concatenate([a_arr["ranks"].astype(np.int64), ranks_ext], axis=0)
        s19 = ga.per_listener_scores(ranks_19)
        i_base, best19, rows19 = ga.one_se_choice(s19, list(range(len(lams))))
        lam_base = float(lams[i_base])
        log.info(
            "p13b: base path over %d weights: best %d (lambda %.3g), "
            "chosen %d (lambda %.3g) by the one-standard-error rule",
            len(lams),
            best19,
            lams[best19],
            i_base,
            lam_base,
        )
        if i_base < n12:
            base_theta = np.asarray(a_doc["path"][i_base]["theta"], dtype=np.float64)
        else:
            base_theta = ext[i_base - n12]["theta"]
        T_base_u = np.zeros(ctx["L_u"])
        T_base_u[base_idx] = base_lay.tables(base_theta)

        # Steps 2 and 3: the cells and the arms at lambda_base.
        cell_recs, arm_recs = {}, {}
        others = [c for c in CELLS if c != BASE_CELL]
        fids = [fit_id(c, None, lam_base) for c in others] + [
            fit_id(BASE_CELL, a, lam_base) for a in ARMS
        ]
        with self._design_for(ctx, store, fids) as pool:
            for cell in others:
                lay, idx = self._cell_layout(ctx, cell)
                t0 = gam_arms.theta_from_tables(lay, T_base_u[idx])
                cell_recs[cell] = self._fit(
                    pool, ctx, store, cell, None, lam_base, t0, "factorial"
                )
            for arm in ARMS:
                arm_recs[arm] = self._fit(
                    pool, ctx, store, BASE_CELL, arm, lam_base, base_theta, "loss arm"
                )
        cell_recs[BASE_CELL] = {"theta": base_theta}
        tables_b = []
        for cell in CELLS:
            lay, idx = self._cell_layout(ctx, cell)
            tables_b.append(self._centered(ctx, lay, idx, cell_recs[cell]["theta"])[3])
        for arm in ARMS:
            tables_b.append(
                self._centered(ctx, base_lay, base_idx, arm_recs[arm]["theta"])[3]
            )
        ranks_b, scored_b, secs = self._pass(ctx, "cells_arms", tables_b, work, key)
        passes["cells_arms"] = (scored_b, secs)
        sc_b = ga.per_listener_scores(ranks_b)
        effects = main_effects(sc_b[: len(CELLS)], CELLS)
        base_score = float(sc_b[CELLS.index(BASE_CELL)].mean())
        chosen_cell, decisions = decide(effects, base_score)
        log.info(
            "p13b: main effects %s (SESOI %.5f). Chosen cell %s",
            ", ".join(f"{f} {effects[f]['mean']:+.5f}" for f in FACTORS),
            decisions["sesoi_abs"],
            cell_label(chosen_cell),
        )

        # Step 5: the final path of the chosen cell.
        if chosen_cell == BASE_CELL:
            final_recs = None
            ranks_f = ranks_19
        else:
            lay, idx = self._cell_layout(ctx, chosen_cell)
            theta = cell_recs[chosen_cell]["theta"]
            fids = [fit_id(chosen_cell, None, lam) for lam in lams]
            final_recs = []
            with self._design_for(ctx, store, fids) as pool:
                for lam in lams:
                    rec = self._fit(
                        pool, ctx, store, chosen_cell, None, lam, theta, "final path"
                    )
                    theta = rec["theta"]
                    final_recs.append(rec)
            tables_f = [
                self._centered(ctx, lay, idx, r["theta"])[3] for r in final_recs
            ]
            ranks_f, scored_f, secs = self._pass(ctx, "final", tables_f, work, key)
            passes["final"] = (scored_f, secs)
        s_f = ga.per_listener_scores(ranks_f)
        i_f, best_f, rows_f = ga.one_se_choice(s_f, list(range(len(lams))))
        log.info(
            "p13b: final path of cell %s: best %d (lambda %.3g), chosen "
            "%d (lambda %.3g)",
            cell_label(chosen_cell),
            best_f,
            lams[best_f],
            i_f,
            lams[i_f],
        )

        doc = self._document(
            ctx,
            ext,
            i_base,
            best19,
            rows19,
            lam_base,
            base_theta,
            cell_recs,
            arm_recs,
            sc_b,
            effects,
            decisions,
            chosen_cell,
            final_recs,
            i_f,
            best_f,
            rows_f,
            passes,
        )
        doc["seconds"] = time.time() - t_start
        atomic_json_write(doc, outputs["model"])
        arrays = {
            "sel_users": ctx["sel_users"],
            "ranks_ext": ranks_ext.astype(np.int32),
            "scored_ext": scored_ext.astype(np.int64),
            "ranks_cells_arms": ranks_b.astype(np.int32),
            "scored_cells_arms": scored_b.astype(np.int64),
        }
        if final_recs is not None:
            arrays["ranks_final"] = ranks_f.astype(np.int32)
            arrays["scored_final"] = passes["final"][0].astype(np.int64)
        s13._atomic_savez(outputs["arrays"], **arrays)
        log.info("p13b: done in %s", _clock(time.time() - t_start))
        return self._info(outputs)

    def _fit_json(self, ctx, rec, cell, arm, lam):
        lay, idx = self._cell_layout(ctx, cell)
        T, Tc, consts, _ = self._centered(ctx, lay, idx, rec["theta"])
        out = {
            "cell": cell_label(cell),
            "arm": arm_label(arm),
            "lambda": float(lam),
            "theta": rec["theta"].tolist(),
            "tables_raw": T.tolist(),
            "tables": Tc.tolist(),
            "center_consts": consts,
            "terms": [t.to_json() for t in lay.terms],
            "fingerprint": gam_arms.layout_fingerprint(lay),
        }
        if "objective" in rec:
            out["fit"] = {
                k: rec[k]
                for k in (
                    "objective",
                    "loss",
                    "smooth",
                    "ridge",
                    "proj_grad_max",
                    "proj_grad_before_newton",
                    "nit",
                    "nfev",
                    "newton_steps",
                    "method",
                    "message",
                    "seconds",
                )
            }
        if any(t.monotone for t in lay.terms):
            out["monotone"] = {
                "violation": gam_arms.monotone_violation(T, lay),
                "flat_share": gam_arms.active_share(rec["theta"], lay),
            }
        return out

    def _document(
        self,
        ctx,
        ext,
        i_base,
        best19,
        rows19,
        lam_base,
        base_theta,
        cell_recs,
        arm_recs,
        sc_b,
        effects,
        decisions,
        chosen_cell,
        final_recs,
        i_f,
        best_f,
        rows_f,
        passes,
    ):
        lams = ctx["path_lambdas"]
        n12 = len(s13.LAMBDAS)
        n_slots = int(self._id_slots()[0])
        k_base = CELLS.index(BASE_CELL)
        ref = sc_b[k_base]

        def row(k):
            d = ref - sc_b[k]
            return {
                "mean": float(sc_b[k].mean()),
                "diff_to_base": float(d.mean()),
                "se": (
                    float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else math.nan
                ),
            }

        cells = []
        for k, cell in enumerate(CELLS):
            j = self._fit_json(ctx, cell_recs[cell], cell, None, lam_base)
            j["factors"] = dict(zip(FACTORS, cell))
            j["selection"] = row(k)
            j["reuses_base_fit"] = cell == BASE_CELL
            cells.append(j)
        arms = []
        for k, arm in enumerate(ARMS):
            j = self._fit_json(ctx, arm_recs[arm], BASE_CELL, arm, lam_base)
            j["selection"] = row(len(CELLS) + k)
            if arm == ("p_pop", "B"):
                j["positives_without_weight"] = ctx["pos_no_pop"]
            arms.append(j)
        final = None
        if final_recs is not None:
            final = [
                dict(
                    self._fit_json(ctx, r, chosen_cell, None, lam), selection=rows_f[i]
                )
                for i, (r, lam) in enumerate(zip(final_recs, lams))
            ]
        return {
            "params": _params(),
            "code": self._code(),
            "upstream": self._upstream_outputs(),
            "p13a": {
                "chosen": ctx["a_doc"]["choice"]["chosen"],
                "fingerprint": ctx["a_doc"]["layout"]["fingerprint"],
            },
            "work_knots": list(ctx["work_knots"]),
            "shape_quantiles": ctx["shape_q"],
            "extension": {
                "lambdas": [float(x) for x in lams],
                "fits": [
                    dict(
                        self._fit_json(ctx, r, BASE_CELL, None, lam),
                        selection=rows19[n12 + i],
                    )
                    for i, (r, lam) in enumerate(zip(ext, EXT_LAMBDAS))
                ],
                "selection": rows19,
                "best": int(best19),
                "chosen": int(i_base),
                "lambda_base": lam_base,
                "base_fit": "p13a" if i_base < n12 else "extension",
            },
            "cells": cells,
            "effects": effects,
            "decisions": decisions,
            "base_score": float(ref.mean()),
            "arms": arms,
            "chosen_cell": cell_label(chosen_cell),
            "final": {
                "cell": cell_label(chosen_cell),
                "reuses_base_path": final is None,
                "path": final,
                "selection": rows_f,
                "best": int(best_f),
                "chosen": int(i_f),
                "lambda": float(lams[i_f]),
            },
            "selection_passes": {
                label: {
                    "configurations": int(
                        {
                            "ext": len(EXT_LAMBDAS),
                            "cells_arms": len(CELLS) + len(ARMS),
                            "final": len(lams),
                        }[label]
                    ),
                    "listeners": int(len(sc)),
                    "full_pass_share": float(np.mean(sc == n_slots)),
                    "mean_scored_share": float(np.mean(sc) / n_slots),
                    "seconds": float(secs),
                }
                for label, (sc, secs) in passes.items()
            },
        }

    # Verify

    def _load(self):
        doc = load_json(self.outputs()["model"])
        check(doc is not None, "p13b_model.json is missing or unreadable")
        z = np.load(self.outputs()["arrays"])
        return doc, {k: z[k] for k in z.files}

    def _all_fits(self, doc):
        """(cell, arm, lambda, stored fit json) of every refit fit."""
        out = []
        for j in doc["extension"]["fits"]:
            out.append((BASE_CELL, None, j["lambda"], j))
        for j in doc["cells"]:
            if not j["reuses_base_fit"]:
                out.append(
                    (tuple(j["factors"][f] for f in FACTORS), None, j["lambda"], j)
                )
        for j, arm in zip(doc["arms"], ARMS):
            out.append((BASE_CELL, arm, j["lambda"], j))
        for j in doc["final"]["path"] or []:
            cell = tuple(int(x) for x in doc["final"]["cell"][1::2])
            out.append((cell, None, j["lambda"], j))
        return out

    def verify(self, params):
        self._verified = None
        gaps = {}
        doc, arr = self._load()
        check(
            doc["params"] == _params(),
            "p13b_model.json was built with other parameters",
        )
        ctx = self._context()
        # Every fit: tables, stationarity, objective, order.
        worst_pg = worst_obj = worst_tab = worst_mono = 0.0
        fits = self._all_fits(doc)
        prog = Progress(
            "p13b verify: stationarity of every fit", len(fits), log.info, unit=" fits"
        )
        with self._design(ctx) as pool:
            for cell, arm, lam, j in fits:
                lay, idx = self._cell_layout(ctx, cell)
                check(
                    gam_arms.layout_fingerprint(lay) == j["fingerprint"],
                    f"{j['cell']} {j['arm']}: the layout differs",
                )
                theta = np.asarray(j["theta"], dtype=np.float64)
                T = lay.tables(theta)
                worst_tab = max(
                    worst_tab, float(np.max(np.abs(T - np.asarray(j["tables_raw"]))))
                )
                obj = self._objective(pool, ctx, lay, idx, arm, lam)
                loss, smooth, ridge, grad = obj.parts(theta)
                F = loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge
                rel = abs(F - j["fit"]["objective"]) / max(1.0, abs(F))
                pg = float(
                    np.max(np.abs(gam.projected_gradient(theta, grad, lay.bounds)))
                )
                worst_obj, worst_pg = max(worst_obj, rel), max(worst_pg, pg)
                check(
                    pg <= s13.KKT_TOL,
                    f"{j['cell']} {j['arm']} lambda {lam:g}: the largest "
                    f"projected gradient is {pg:.2e}",
                )
                check(
                    rel <= s13.OBJ_RTOL,
                    f"{j['cell']} {j['arm']} lambda {lam:g}: the "
                    f"objective differs from its recomputation by "
                    f"{rel:.2e}",
                )
                if any(t.monotone for t in lay.terms):
                    worst_mono = max(worst_mono, gam_arms.monotone_violation(T, lay))
                prog.advance(1)
        prog.finish()
        check(
            worst_tab == 0.0,
            f"stored raw tables differ from M theta by " f"{worst_tab:.2e}",
        )
        check(
            worst_mono <= MONOTONE_TOL,
            f"a monotone cell breaks its " f"order by {worst_mono:.2e}",
        )
        gaps.update(kkt=worst_pg, objective=worst_obj, monotone=worst_mono)
        # The arm offsets against P11's values.
        gaps["offsets"] = self._verify_offsets(ctx)
        un = {j["arm"]: np.asarray(j["tables"]) for j in doc["arms"]}
        gaps["uniform_b"] = float(np.max(np.abs(un["uniform-none"] - un["uniform-B"])))
        check(
            gaps["uniform_b"] <= s13.TWO_RUN_TOL,
            f"the uniform arm with form B differs from the uncorrected "
            f"one by {gaps['uniform_b']:.2e}",
        )
        # The choices from the stored ranks.
        self._verify_choices(doc, arr, ctx)
        gaps.update(self._verify_ranks(doc, arr, ctx))
        self._verified = gaps
        log.info(
            "p13b: verify passed. Largest gaps: %s",
            ", ".join(f"{k} {v:.1e}" for k, v in gaps.items()),
        )

    def _verify_offsets(self, ctx):
        """The arm offsets of sampled rows against a computation from
        P11's stored n_allowed, pop_allowed and P3's p_pop."""
        inp = self._inputs()
        p = inp["p"]
        item_ids, p_pop = self._proposals()
        pp = dict(zip(item_ids.tolist(), p_pop.tolist()))
        lens, n_pos = inp["lens"], inp["n_pos"]
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        rng = np.random.default_rng(s13.SEED + 11)
        users = np.sort(
            rng.choice(
                len(lens), size=min(VERIFY_OFFSET_USERS, len(lens)), replace=False
            )
        )
        worst = 0.0
        for u in users:
            r0 = int(starts[u])
            na = float(p["n_allowed"][inp["tr"][u]])
            pa = float(p["pop_allowed"][inp["tr"][u]])
            blk = {
                "q_u": ctx["q_u"][r0 : r0 + lens[u]],
                "q_p": ctx["q_p"][r0 : r0 + lens[u]],
                "pos_rows": np.arange(n_pos[u]),
                "pos_weight": np.ones(n_pos[u]),
                "neg_rows": (n_pos[u] + np.arange(2 * NEG_PER_KIND))[None, :],
            }
            items = inp["item"][r0 : r0 + lens[u]]
            for arm in ARMS:
                if arm[1] not in ("A", "B"):
                    continue
                got = arm_block(blk, arm)["offset"]
                m = len(neg_cols(arm[0]))
                for r, it in enumerate(items):
                    qu = 1.0 / na
                    qp = pp.get(int(it), 0.0) / pa
                    q = {"uniform": qu, "p_pop": qp, "mixture": 0.5 * qu + 0.5 * qp}[
                        arm[0]
                    ]
                    is_pos = r < n_pos[u]
                    if q <= 0.0 or (arm[1] == "A" and is_pos):
                        want = 0.0
                    else:
                        want = -math.log(m * q)
                    worst = max(worst, abs(got[r] - want))
        check(
            worst <= 1e-12,
            f"an arm offset differs from P11's values by " f"{worst:.2e}",
        )
        return worst

    def _verify_choices(self, doc, arr, ctx):
        a_arr = ctx["a_arr"]
        lams = ctx["path_lambdas"]
        r19 = np.concatenate(
            [a_arr["ranks"].astype(np.int64), arr["ranks_ext"].astype(np.int64)]
        )
        i_base, best, _ = ga.one_se_choice(
            ga.per_listener_scores(r19), list(range(len(lams)))
        )
        check(
            i_base == doc["extension"]["chosen"] and best == doc["extension"]["best"],
            "lambda_base does not follow from the stored ranks",
        )
        n12 = len(s13.LAMBDAS)
        k_base = CELLS.index(BASE_CELL)
        want = (
            ctx["a_doc"]["path"][i_base]["theta"]
            if i_base < n12
            else doc["extension"]["fits"][i_base - n12]["theta"]
        )
        check(
            doc["cells"][k_base]["theta"] == want,
            "the base cell does not hold the fit at lambda_base",
        )
        check(
            all(
                j["lambda"] == doc["extension"]["lambda_base"]
                for j in doc["cells"] + doc["arms"]
            ),
            "a cell or an arm was fit at another weight than lambda_base",
        )
        # Censored ranks do not depend on the other configurations of a
        # pass, so two passes must agree on a configuration they share:
        # the base cell is the base path at lambda_base, and (below) the
        # final path holds the chosen cell's factorial fit at lambda_base.
        check(
            np.array_equal(arr["ranks_cells_arms"][k_base], r19[i_base]),
            "the base cell's ranks differ from those of the base path at "
            "lambda_base",
        )
        sc = ga.per_listener_scores(arr["ranks_cells_arms"])
        eff = main_effects(sc[: len(CELLS)], CELLS)
        base_score = float(sc[k_base].mean())
        cell, dec = decide(eff, base_score)

        def same(a, b):  # NaN-safe exact comparison
            return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

        check(
            cell_label(cell) == doc["chosen_cell"] and same(dec, doc["decisions"]),
            "the chosen cell does not follow from the stored ranks",
        )
        check(
            same(eff, doc["effects"]) and base_score == doc["base_score"],
            "the stored effects differ from the stored ranks",
        )
        if "ranks_final" in arr:
            check(
                np.array_equal(
                    arr["ranks_final"][i_base],
                    arr["ranks_cells_arms"][CELLS.index(cell)],
                ),
                "the final path's ranks at lambda_base differ from the "
                "chosen cell's factorial ranks",
            )
        rf = arr["ranks_final"] if "ranks_final" in arr else r19
        i_f, best_f, _ = ga.one_se_choice(
            ga.per_listener_scores(rf), list(range(len(lams)))
        )
        check(
            i_f == doc["final"]["chosen"] and best_f == doc["final"]["best"],
            "the final choice does not follow from the stored ranks",
        )

    def _verify_ranks(self, doc, arr, ctx):
        """Sampled selection ranks of a few configurations of each pass
        against the dense reference on concepts.full_catalog()."""
        n_slots, _ = self._id_slots()
        src = self._sources(n_slots)
        space = self._space(src)
        u_indptr, u_indices, u_data = self._csr()
        users, picks = ctx["sel_users"], ctx["sel_picks"]
        L_u = ctx["L_u"]

        def union_of(j):
            cell = tuple(int(x) for x in j["cell"][1::2])
            _, idx = self._cell_layout(ctx, cell)
            Tu = np.zeros(L_u)
            Tu[idx] = np.asarray(j["tables"])
            return Tu

        checks = []
        for k in (0, len(doc["extension"]["fits"]) - 1):
            checks.append((union_of(doc["extension"]["fits"][k]), arr["ranks_ext"][k]))
        mono_k = next(k for k, c in enumerate(CELLS) if c[0] == 1)
        for k in (CELLS.index(BASE_CELL), mono_k):
            checks.append((union_of(doc["cells"][k]), arr["ranks_cells_arms"][k]))
        k_bpr = len(CELLS) + ARMS.index(("mix32", "bpr"))
        checks.append(
            (
                union_of(doc["arms"][ARMS.index(("mix32", "bpr"))]),
                arr["ranks_cells_arms"][k_bpr],
            )
        )
        if "ranks_final" in arr:
            i = doc["final"]["chosen"]
            checks.append((union_of(doc["final"]["path"][i]), arr["ranks_final"][i]))
        tables = np.stack([t for t, _ in checks])
        stored = np.stack([r for _, r in checks])
        cut1 = s13.RANK_CUT + 1
        rng = np.random.default_rng(s13.SEED + 12)
        take = np.sort(
            rng.choice(
                len(users), size=min(s13.VERIFY_USERS, len(users)), replace=False
            )
        )
        near = 0
        prog = Progress(
            "p13b verify: selection ranks against the dense " "reference",
            len(take),
            log.info,
            unit=" listeners",
        )
        for j in take:
            u = int(users[j])
            items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
            state = cn.build_user_state(items, weights, space)
            full = cn.full_catalog(state, space)
            ref = ct.reference_scores(
                full, ctx["union"], tables, float(ctx["m_sel"][j]), space.tier
            )
            del full
            for c in range(len(tables)):
                for p in range(2):
                    lo, hi = ct.reference_rank(
                        ref[c], picks[j, p], items, s13.SCORE_TOL
                    )
                    near += int(hi > lo)
                    got = int(stored[c, j, p])
                    check(
                        min(lo, cut1) <= got <= min(hi, cut1),
                        f"listener {u}, check {c}, pick {p}: the stored "
                        f"rank {got} is outside the reference range "
                        f"[{lo}, {hi}] censored at {cut1}",
                    )
            prog.advance(1)
        prog.finish()
        return {"near_ties": float(near)}

    def report(self, params):
        doc, arr = self._load()
        ctx = self._context()
        lines = []
        add = lines.append
        add("# P13b report, part 1: the arms on the CPU")
        add("")
        add(f"- parameters: {json.dumps(doc['params'])}")
        if self._verified:
            add(
                "- verify(): "
                + ", ".join(f"{k} {v:.1e}" for k, v in self._verified.items())
                + ". kkt is the largest projected gradient over all fits, "
                "objective the largest relative gap to a recomputation, "
                "monotone the largest break of a monotone order, "
                "offsets the largest gap of an arm offset, uniform_b the "
                "largest table gap between the uniform arms with and "
                "without form B, near_ties the checked ranks with a score "
                "within the tolerance of the pick's."
            )
        add(f"- run time: {_clock(doc['seconds'])}.")
        for label in ("ext", "cells_arms", "final"):
            sp = doc["selection_passes"].get(label)
            if sp is None:
                continue
            add(
                f"- selection pass '{label}': {sp['configurations']} "
                f"configurations, {sp['listeners']:,} listeners, "
                f"{100 * sp['full_pass_share']:.1f} percent of them over the "
                f"whole catalog, {100 * sp['mean_scored_share']:.1f} percent "
                f"of all slots scored, {_clock(sp['seconds'])}. Ranks are "
                f"exact up to {s13.RANK_CUT} and stored as "
                f"{s13.RANK_CUT + 1} beyond."
            )
        add("")
        self._report_path(doc, add)
        self._report_factorial(doc, ctx, add)
        self._report_arms(doc, ctx, add)
        self._report_final(doc, ctx, add)
        text = "\n".join(lines) + "\n"
        check(";" not in text, "the report holds a semicolon")
        atomic_text_write(text, self.outputs()["report"])
        log.info("p13b report written")

    def _report_path(self, doc, add):
        e = doc["extension"]
        add("## 1. The base path, extended")
        add("")
        n12 = len(s13.LAMBDAS)
        add(
            f"P13a's best weight was its weakest, so the path continues "
            f"below it. The first {n12} rows are P13a's fits and ranks, the "
            f"last {len(e['lambdas']) - n12} are new. Difference: the best "
            f"score minus the row's score, with the standard error of the "
            f"paired per-listener differences."
        )
        add("")
        add("| lambda | selection score | difference to best | choice |")
        add("|---:|---:|---|---|")
        for i, (lam, r) in enumerate(zip(e["lambdas"], e["selection"])):
            mark = []
            if i == e["best"]:
                mark.append("best")
            if i == e["chosen"]:
                mark.append("chosen")
            add(
                f"| {lam:.3g} | {r['mean']:.5f} | {r['diff']:.5f} +- "
                f"{r['se']:.5f} | {', '.join(mark)} |"
            )
        add("")
        add(
            f"- lambda_base = {e['lambda_base']:.3g}. The factorial and the "
            f"loss arms use it."
        )
        add("")

    def _report_factorial(self, doc, ctx, add):
        add("## 2. The factorial")
        add("")
        add(
            "Every cell at lambda_base, with the base loss (mixture of 16 "
            "uniform and 16 p_pop negatives, form A). Factors: monotone "
            "bounds on the affinity terms, work_affinity, the fuzzy_dur "
            "deviation curve, the popularity tilt. Difference: the base "
            "cell's score minus the cell's score."
        )
        add("")
        add(
            "| cell | monotone | work | curve | tilt | score | difference "
            "to base | fit |"
        )
        add("|---|---|---|---|---|---:|---|---|")
        for j in doc["cells"]:
            fct = j["factors"]
            s = j["selection"]
            fit = (
                "base fit"
                if j["reuses_base_fit"]
                else f"{j['fit']['message']}, {j['fit']['proj_grad_max']:.1e}"
            )
            add(
                f"| {j['cell']} | {fct['monotone']} | "
                f"{fct['work_affinity']} | {fct['tier_curve']} | "
                f"{fct['tilt']} | {s['mean']:.5f} | "
                f"{s['diff_to_base']:+.5f} +- {s['se']:.5f} | {fit} |"
            )
        add("")
        add(
            "Effects: the mean over listeners of the average score of the "
            "cells with a factor minus that of the cells without it, and "
            "for two factors the matching interaction contrast. Relative: "
            "the effect as a share of the base cell's score."
        )
        add("")
        add("| effect | estimate | standard error | relative |")
        add("|---|---:|---:|---:|")
        bs = doc["base_score"]
        for name, e in doc["effects"].items():
            add(
                f"| {name} | {e['mean']:+.5f} | {e['se']:.5f} | "
                f"{100 * e['mean'] / bs:+.1f}% |"
            )
        add("")
        d = doc["decisions"]
        add(
            f"Decisions (rules declared in the stage, SESOI "
            f"{d['sesoi_abs']:.5f}, 2 percent of the base score "
            f"{bs:.5f}):"
        )
        add("")
        add(
            f"- monotone bounds: {'kept' if d['monotone'] else 'dropped'} "
            f"(kept unless they cost more than SESOI)."
        )
        for f, label in (
            ("work_affinity", "work_affinity"),
            ("tier_curve", "fuzzy_dur deviation curve"),
            ("tilt", "popularity tilt"),
        ):
            add(
                f"- {label}: {'kept' if d[f] else 'dropped'} (kept only "
                f"on a gain above SESOI)."
            )
        add(f"- chosen cell: {doc['chosen_cell']}.")
        add("")
        mono = next(
            j
            for j in doc["cells"]
            if j["factors"]["monotone"] == 1
            and j["factors"]["work_affinity"] == 0
            and j["factors"]["tier_curve"] == 1
            and j["factors"]["tilt"] == 1
        )
        base = next(j for j in doc["cells"] if j["reuses_base_fit"])
        add("### 2a. The affinity shapes with and without the bounds")
        add("")
        add(
            "Values in log-odds points, centered on the p_pop negatives, "
            "at percentiles of the positives' values, for the base cell "
            "(free) and the same cell with bounds (monotone). flat: the "
            "share of the term's knot segments held flat by a bound."
        )
        add("")
        add("| term | shape | p5 | p25 | p50 | p75 | p95 | zero | missing " "| flat |")
        add("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for name in s13.BASE_CONCEPTS:
            for label, j in (("free", base), ("monotone", mono)):
                vals, zero, miss = self._shape_values(ctx, j, name)
                flat = j.get("monotone", {}).get("flat_share", {}).get(name)
                add(
                    f"| {name} | {label} | "
                    + " | ".join(f"{v:+.3f}" for v in vals)
                    + f" | {zero} | {miss} | "
                    f"{'' if flat is None else f'{100 * flat:.0f}%'} |"
                )
        add("")

    def _shape_values(self, ctx, j, name):
        terms = [gam.Term.from_json(t) for t in j["terms"]]
        T = np.asarray(j["tables"])
        offsets = np.cumsum([0] + [t.size for t in terms])[:-1]
        at = [k for k, t in enumerate(terms) if t.name == name]
        if not at:
            raise KeyError(f"{j['cell']} {j['arm']} has no term {name}")
        t, off = terms[at[0]], int(offsets[at[0]])
        xs = ctx["shape_q"].get(name) or []
        kn = t.knot_array
        tab = T[off : off + t.size]
        vals = [float(np.interp(np.clip(x, kn[0], kn[-1]), kn, tab[: t.K])) for x in xs]
        zero = f"{tab[t.K + gam.ZERO]:+.3f}" if t.zero_state else ""
        miss = f"{tab[t.K + gam.MISS]:+.3f}" if t.missing_state else ""
        return vals, zero, miss

    def _report_arms(self, doc, ctx, add):
        add("## 3. The loss and sampler arms")
        add("")
        add(
            "Every arm at lambda_base on the base terms. uniform, p_pop and "
            "mixture use 16 negatives per positive (the mixture the first "
            "8 of each kind). none: no correction. A: -log(m q) on each "
            "negative. B: -log(m q) on every candidate, the positive "
            "included. mix32-bpr: BPR on all 32 negatives. Difference: the "
            "base cell's score minus the arm's score. The loss decision "
            "waits for the large-m reference of part 2."
        )
        add("")
        add("| arm | score | difference to base | fit |")
        add("|---|---:|---|---|")
        for j in doc["arms"]:
            s = j["selection"]
            extra = ""
            if j.get("positives_without_weight"):
                extra = (
                    f", {j['positives_without_weight']:,} positives " f"without weight"
                )
            add(
                f"| {j['arm']} | {s['mean']:.5f} | "
                f"{s['diff_to_base']:+.5f} +- {s['se']:.5f} | "
                f"{j['fit']['message']}{extra} |"
            )
        add("")
        add(
            "Popularity shape and missing constants per arm (log-odds "
            "points, centered on the p_pop negatives). The popularity "
            "columns give f_0 at percentiles of the positives' "
            "popularity."
        )
        add("")
        miss_terms = [t.name for t in ctx["base"] if t.missing_state]
        add(
            "| arm | pop p5 | pop p25 | pop p50 | pop p75 | pop p95 | "
            + " | ".join(f"missing {n}" for n in miss_terms)
            + " |"
        )
        add("|---|" + "---:|" * (5 + len(miss_terms)))
        base = next(j for j in doc["cells"] if j["reuses_base_fit"])
        for label, j in [("base", base)] + [(a["arm"], a) for a in doc["arms"]]:
            vals, _, _ = self._shape_values(ctx, j, "popularity")
            misses = [self._shape_values(ctx, j, n)[2] for n in miss_terms]
            add(
                f"| {label} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + " | "
                + " | ".join(misses)
                + " |"
            )
        add("")

    def _final_model(self, doc, ctx):
        """The chosen model of the final path as a fit record with terms
        and centered tables."""
        fin = doc["final"]
        i = fin["chosen"]
        if not fin["reuses_base_path"]:
            return fin["path"][i]
        n12 = len(s13.LAMBDAS)
        if i >= n12:
            return doc["extension"]["fits"][i - n12]
        a = ctx["a_doc"]
        return {
            "cell": cell_label(BASE_CELL),
            "arm": "base",
            "lambda": a["path"][i]["lambda"],
            "terms": a["terms"],
            "tables": a["path"][i]["tables"],
        }

    def _report_final(self, doc, ctx, add):
        fin = doc["final"]
        add("## 4. The final path and the chosen model")
        add("")
        if fin["reuses_base_path"]:
            add(
                f"The chosen cell is the base cell, so the base path of "
                f"section 1 is the final path. Chosen: lambda "
                f"{fin['lambda']:.3g}."
            )
        else:
            add(
                f"The chosen cell {fin['cell']} over the {len(fin['path'])} "
                f"weights, strongest penalty first, warm-started from its "
                f"factorial fit. Difference: the best score minus the row's "
                f"score, with the standard error of the paired per-listener "
                f"differences."
            )
            add("")
            add("| lambda | selection score | difference to best | fit | " "choice |")
            add("|---:|---:|---|---|---|")
            for i, j in enumerate(fin["path"]):
                r = j["selection"]
                mark = []
                if i == fin["best"]:
                    mark.append("best")
                if i == fin["chosen"]:
                    mark.append("chosen")
                add(
                    f"| {j['lambda']:.3g} | {r['mean']:.5f} | "
                    f"{r['diff']:.5f} +- {r['se']:.5f} | "
                    f"{j['fit']['message']} | {', '.join(mark)} |"
                )
        add("")
        j = self._final_model(doc, ctx)
        add(
            f"### 4a. The chosen model (cell {fin['cell']}, lambda "
            f"{fin['lambda']:.3g})"
        )
        add("")
        add(
            "Values in log-odds points, centered on the p_pop negatives, at "
            "percentiles of the positives' values. zero: data, but no "
            "match. missing: no data."
        )
        add("")
        add("| term | p5 | p25 | p50 | p75 | p95 | zero | missing |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in j["terms"]:
            vals, zero, miss = self._shape_values(ctx, j, t["name"])
            add(
                f"| {t['name']} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + f" | {zero} | {miss} |"
            )
        add("")
        add("The loss stays form A until the large-m reference of part 2.")
        add("")
