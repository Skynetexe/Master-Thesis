"""Stage P13b, part 2: the loss form and the number of negatives.

Part 1 found that the loss form decides the popularity and the missing-
data lines of the explanations. Under form A the popularity shape flips
with the sampler, under form B it does not. Part 2 answers two questions
with the pairs of P11b and the features of P12b:
1. Which form gives the shapes of the full softmax? The large-m
   reference on REF_USERS listeners answers it.
2. How many negatives does the chosen form need? The sweep on all
   training listeners answers it.

The model stays part 1's final model: its cell (the chosen factorial
cell), its knots, its layout constants and its penalty weight. Part 2
changes only the loss and the negatives. The twin cell is the chosen
cell with the monotone factor flipped, for the check of the bounds under
the new loss. The layouts are rebuilt from P13a's stored constants, and
their fingerprints must equal part 1's.

Data levels. A level k keeps each listener's positives and the first k
draws of each kind, so the levels of one file are nested. Arms, as in
part 1 (stage_p13b.arm_block): ("mix32", form) uses all 2k negatives of
the level with the mixture density, ("p_pop", form) the k p_pop draws.
Form A subtracts log(m q) from each negative, form B from every
candidate, the positive included.
- ref, levels REF_LEVELS: the chosen cell with the mixture in both forms
  at every level, and p_pop in both forms at the smallest and the
  largest level.
- wide, levels FULL_LEVELS: the chosen and the twin cell with mixture
  form A, mixture form B and p_pop form B at every level.
At the smallest wide level the rows are P11's rows, and mixture form A
is part 1's loss. The chosen cell's fit there must equal part 1's final
fit (verify() checks the objective and the parameters). The fit blocks
grow with the rows (fit_blocks), and at that level they equal part 1's
blocks on the real data, so the loss sums in the same order.

Explanation points. For each term: its value at the positives'
percentiles of P13a (5, 25, 50, 75, 95), its zero and its missing
constant, each minus its value at the median. They are what an
explanation line reads, and they do not depend on centering. The gap
between two fits is the largest absolute difference of their points.

Rules, declared before the first run:
- D1, reported: the largest gap between the four fits at the largest
  reference level (both forms, both samplers). All four estimate the
  full softmax, so a small gap shows the reference has converged. p_pop
  form B gives no weight to a positive outside P3's core, so part of its
  gap can come from those positives.
- D2, the loss form. R is mixture form A at the largest reference level:
  the sampled softmax, whose bias vanishes as m grows (Blanc & Rendle
  2018). Each form's bias is its gap to R at the smallest reference
  level (m = 32). The form with the smaller bias wins. If both biases lie
  within TAU, the higher selection score of the chosen cell at m = 32
  (mixture) decides. Ties go to form A, part 1's loss.
- D3, the number of negatives. Among the chosen form's full-data fits of
  the chosen cell, the best selection score s*. The fit with the fewest
  negatives whose score lies within SESOI (2 percent of s*) of s* wins.
  Ties in m go to the higher score.
- D4, the bounds. At the chosen arm and level, the chosen cell stays
  unless the twin scores more than SESOI (2 percent of the chosen cell's
  score) above it.
The final model is the chosen or the twin cell (D4) with the arm and
level of D2 and D3, at part 1's penalty weight.

Selection: one pass over the full-data fits, full-catalog NDCG@100 of the
20,000 selection listeners, as in part 1. The reference fits get no pass:
they use 5,000 listeners, and they answer a question about shapes.

Warm starts. A mixture fit starts from the same arm one level below (at
the first level: part 1's fit of the cell, or the level's form A fit for
form B). A p_pop fit starts from the level's mixture fit of its form.
The objective is strictly convex (the ridge term), so the start changes
the path of the solver, not the optimum.

Resume: every fit, every level's background means and every selection
block is saved under SCRATCH_DIR/p13b2, keyed by the parameters, the
upstream outputs and the code (this module included). Progress lines
give the share of the fit work done, by rows, and a line at most every
PROGRESS_SECONDS tells which Newton step a long fit has reached.

verify():
- P13a's knots and terms, and the layout fingerprints of both cells
  against part 1.
- level by level, from the level's own rows: every fit is a stationary
  point of its objective (projected gradient, objective, tables, monotone
  order), the stored centered tables follow from the level's p_pop
  negatives, and the stored explanation points follow from the tables.
- level by level, the negative columns and offsets of each arm on
  sampled listeners, against a computation from the stored n_allowed and
  pop_allowed and P3's p_pop.
- part 1's final fit is the chosen cell's fit at the smallest wide level
  with mixture form A (same objective, same parameters).
- the scores, the gaps, D1 to D4 and the final model from the stored
  tables and ranks.
- sampled selection ranks against the dense reference.
"""

import contextlib
import hashlib
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
from . import evaluation as ev
from . import fusion_pairs as fp
from . import gam
from . import gam_arms
from . import gam_audit as ga
from . import selection as sel
from . import stage_p11b as s11b
from . import stage_p13a as s13
from . import stage_p13b as s13b
from .als import PROGRESS_SECONDS, Progress, _clock
from .common import (
    atomic_json_write,
    atomic_text_write,
    check,
    file_sha256,
    load_json,
    log,
)
from .config import MODEL_DIR, SCRATCH_DIR

VERSION = "p13b2-v1"
FULL_LEVELS = (16, 32, 64)  # draws per kind, all training listeners
REF_LEVELS = (16, 64, 256, 1024, 4096)
FULL_ARMS = (("mix32", "A"), ("mix32", "B"), ("p_pop", "B"))
REF_MIX_ARMS = (("mix32", "A"), ("mix32", "B"))
REF_POP_ARMS = (("p_pop", "A"), ("p_pop", "B"))
TAU = 0.5  # log-odds points
SESOI = 0.02
ROWS_PER_BLOCK = 300_000  # fit blocks scale with the rows
MIN_BLOCKS = 64  # part 1's block count at 18.7M rows
VERIFY_RANK_USERS = 24
VERIFY_OFFSET_USERS = 300
CENTER_TOL = 1e-12


def _params():
    return {
        "version": VERSION,
        "full_levels": list(FULL_LEVELS),
        "ref_levels": list(REF_LEVELS),
        "full_arms": [list(a) for a in FULL_ARMS],
        "ref_arms": {
            "mixture": [list(a) for a in REF_MIX_ARMS],
            "p_pop": [list(a) for a in REF_POP_ARMS],
        },
        "tau": TAU,
        "sesoi": SESOI,
        "rows_per_block": ROWS_PER_BLOCK,
        "min_blocks": MIN_BLOCKS,
        "solver": s13b.SOLVER,
        "kkt_tol": s13.KKT_TOL,
    }


def ref_arms(level):
    """The arms of the reference at one level."""
    if level in (min(REF_LEVELS), max(REF_LEVELS)):
        return REF_MIX_ARMS + REF_POP_ARMS
    return REF_MIX_ARMS


def n_negatives(arm, level):
    """The negatives per listener of an arm at a level."""
    return 2 * int(level) if arm[0] == "mix32" else int(level)


def fit_blocks(n_rows):
    return max(MIN_BLOCKS, int(math.ceil(n_rows / ROWS_PER_BLOCK)))


def fid2(which, level, cell, arm, lam):
    return (
        f"{which}|{int(level)}|{s13b.cell_label(cell)}|"
        f"{s13b.arm_label(arm)}|{float(lam)!r}"
    )


def flip_monotone(cell):
    return (1 - int(cell[0]),) + tuple(int(x) for x in cell[1:])


def warm_start(prev, cur, cell, arm, first):
    """The start of a fit. prev: {(cell, arm): theta} one level below,
    cur: the fits of this level so far, first: the cell's start at the
    first level."""
    if arm[0] == "mix32":
        for src, key in ((prev, (cell, arm)), (cur, (cell, ("mix32", "A")))):
            if key in src:
                return src[key]
        return first
    key = (cell, ("mix32", arm[1]))
    if key in cur:
        return cur[key]
    return prev.get((cell, arm), first)


def shape_points(terms_json, tables, xq):
    """{term: its values at the x positions xq[term] (the positives'
    percentiles), then its zero and its missing constant, each minus its
    value at the middle position}. Invariant to a shift of the term."""
    T = np.asarray(tables, dtype=np.float64)
    out = {}
    off = 0
    for t in terms_json:
        kn = np.asarray(t["knots"], dtype=np.float64)
        K = len(kn)
        size = K + 4
        tab = T[off : off + size]
        off += size
        xs = xq.get(t["name"]) or []
        if not xs:
            continue
        vals = [
            float(np.interp(min(max(float(x), kn[0]), kn[-1]), kn, tab[:K])) for x in xs
        ]
        mid = vals[len(vals) // 2]
        pts = [v - mid for v in vals]
        if t["zero_state"]:
            pts.append(float(tab[K + gam.ZERO]) - mid)
        if t["missing_state"]:
            pts.append(float(tab[K + gam.MISS]) - mid)
        out[t["name"]] = pts
    return out


def points_gap(a, b):
    """The largest absolute difference of two fits' explanation points
    over the terms they share."""
    worst = 0.0
    for name in set(a) & set(b):
        worst = max(
            worst, float(np.max(np.abs(np.asarray(a[name]) - np.asarray(b[name]))))
        )
    return worst


def decide2(ref_pts, full_means, chosen, twin, tau=TAU, sesoi=SESOI):
    """D1 to D4 of the module docstring. ref_pts: {(level, arm): points},
    full_means: {(cell, arm, level): mean selection score}."""
    top, low = max(REF_LEVELS), min(REF_LEVELS)
    tops = [ref_pts[(top, a)] for a in ref_arms(top)]
    g_ref = max(points_gap(x, y) for i, x in enumerate(tops) for y in tops[i + 1 :])
    R = ref_pts[(top, ("mix32", "A"))]
    bias = {f: points_gap(ref_pts[(low, ("mix32", f))], R) for f in "AB"}
    lvl0 = min(FULL_LEVELS)
    s32 = {f: full_means[(chosen, ("mix32", f), lvl0)] for f in "AB"}
    if bias["A"] <= tau and bias["B"] <= tau:
        form = "B" if s32["B"] > s32["A"] else "A"
        by = "selection score (both biases within tau)"
    else:
        form = "B" if bias["B"] < bias["A"] else "A"
        by = "bias"
    cands = [(a, k) for a in FULL_ARMS if a[1] == form for k in FULL_LEVELS]
    means = {c: full_means[(chosen, c[0], c[1])] for c in cands}
    best = max(means.values())
    thr = sesoi * best
    ok = sorted(
        (c for c in cands if means[c] >= best - thr),
        key=lambda c: (n_negatives(*c), -means[c]),
    )
    arm, level = ok[0]
    s_ch = means[(arm, level)]
    s_tw = full_means[(twin, arm, level)]
    keep = not (s_tw - s_ch > sesoi * s_ch)
    return {
        "g_ref": g_ref,
        "bias": bias,
        "tau": tau,
        "form": form,
        "form_by": by,
        "scores_m32": s32,
        "best": best,
        "sesoi_abs": thr,
        "arm": list(arm),
        "level": int(level),
        "m": n_negatives(arm, level),
        "chosen_score": s_ch,
        "twin_score": s_tw,
        "twin_kept_out": bool(keep),
        "cell": s13b.cell_label(chosen if keep else twin),
    }


class _Relabel(logging.Filter):
    """Renames the log lines that P13a's and P13b's code writes while this
    stage runs a selection pass, and points the advice of the swap warning
    at this stage."""

    def filter(self, record):
        msg = record.msg
        if isinstance(msg, str):
            if msg.startswith("p13a"):
                msg = "p13b2" + msg[4:]
            elif msg.startswith("p13b:"):
                msg = "p13b2:" + msg[5:]
            record.msg = msg.replace("--only p13a_fit", "--only p13b2_loss")
        return True


@contextlib.contextmanager
def _relabel_logs():
    flt = _Relabel()
    log.addFilter(flt)
    try:
        yield
    finally:
        log.removeFilter(flt)


class _Store(s13b._FitStore):
    """Part 1's fit store plus each level's background means."""

    def means(self, level_id):
        m = (self.state.get("means") or {}).get(level_id)
        if m is None:
            return None
        return {k: np.asarray(v, dtype=np.float64) for k, v in m.items()}

    def put_means(self, level_id, means):
        self.state.setdefault("means", {})[level_id] = {
            k: np.asarray(v, dtype=np.float64).tolist() for k, v in means.items()
        }
        atomic_json_write(self.state, self.path)


class StageP13b2(s13b.StageP13b):
    name = "p13b2_loss"
    upstream = s13b.StageP13b.upstream + [
        "p13b_arms",
        "p11b_more_pairs",
        "p12b_more_features",
    ]

    def __init__(self):
        super().__init__()
        self._level = None

    def outputs(self):
        return {
            "model": MODEL_DIR / "p13b2_model.json",
            "arrays": MODEL_DIR / "p13b2_arrays.npz",
            "report": MODEL_DIR / "reports" / "p13b2_report.md",
        }

    def _modules(self):
        return (gam, gam_arms, ga, ct, cn, cd, ar, sel, s13, s13b, s11b)

    def _code(self):
        return s13b.StageP13b._code(self) | {
            Path(__file__).name: file_sha256(Path(__file__))
        }

    def _key(self):
        return {
            "params": _params(),
            "p13a_params": s13._params(),
            "p13b_params": s13b._params(),
            "p11b_params": s11b._params(),
            "upstream": self._upstream_outputs(),
            "code": self._code(),
        }

    # Data

    def _data(self, which):
        """P11b's pairs and P12b's features of one file, checked."""
        key = f"data_{which}"
        if key not in self._cache:
            pp = MODEL_DIR / f"fusion_pairs_{which}.npz"
            pf = MODEL_DIR / f"fusion_features_{which}.npz"
            zp, zf = np.load(pp), np.load(pf)
            p_sha = ev.npz_hash(pp)
            check(
                str(zf["pairs_sha256"]) == p_sha,
                f"fusion_features_{which}.npz was built from other pairs",
            )
            params = json.loads(str(zp["params"]))
            check(
                params == s11b._params(),
                f"fusion_pairs_{which}.npz was built with other P11b " f"parameters",
            )
            K = int(params["wide_per_kind" if which == "wide" else "ref_per_kind"])
            p = {
                k: np.asarray(zp[k])
                for k in (
                    "user_id",
                    "split",
                    "indptr",
                    "item",
                    "kind",
                    "n_allowed",
                    "pop_allowed",
                )
            }
            f = {k: zf[k] for k in ("values", "has", "match_tier")}
            n_pos = np.add.reduceat(
                (p["kind"] == fp.KIND_POSITIVE).astype(np.int64), p["indptr"][:-1]
            )
            kind, _ = s11b.layout_arrays(p["indptr"], n_pos, K)
            check(
                np.array_equal(p["kind"], kind),
                f"fusion_pairs_{which}.npz does not hold each listener's "
                f"positives, then {K} uniform, then {K} p_pop draws",
            )
            self._cache[key] = {
                "p": p,
                "f": f,
                "K": K,
                "sha": p_sha,
                "f_params": json.loads(str(zf["params"])),
                "n_pos": n_pos,
            }
        return self._cache[key]

    def _level_rows(self, which):
        """(listeners, positives) of one file, for the progress lines."""
        z = np.load(MODEL_DIR / f"fusion_pairs_{which}.npz")
        return (
            int(len(z["user_id"])),
            int(np.count_nonzero(z["kind"] == fp.KIND_POSITIVE)),
        )

    def _set_level(self, which, k):
        """Makes the level the one that _pairs() and _features() give. The
        rows of the level before are freed first."""
        self._cache.pop("inputs", None)
        self._level = None
        d = self._data(which)
        K = d["K"]
        check(
            k <= K,
            f"level {k} needs more than the {K} draws per kind " f"of the {which} file",
        )
        p0, f0 = d["p"], d["f"]
        if k == K:
            p, f = dict(p0), dict(f0)
        else:
            rows = s11b.prefix_rows(p0["indptr"], d["n_pos"], K, k)
            p = dict(
                p0,
                item=p0["item"][rows],
                kind=p0["kind"][rows],
                indptr=np.concatenate(([0], np.cumsum(d["n_pos"] + 2 * k))).astype(
                    np.int64
                ),
            )
            f = {name: arr[rows] for name, arr in f0.items()}
        sha = hashlib.sha256(f"{d['sha']}|{which}|{k}".encode("utf-8")).hexdigest()
        self._level = {
            "id": f"{which}:{k}",
            "pairs": (p, {"neg_per_kind": int(k)}, sha),
            "features": (f, d["f_params"], sha, sha),
        }

    def _pairs(self):
        check(self._level is not None, "no data level is set")
        return self._level["pairs"]

    def _features(self):
        check(self._level is not None, "no data level is set")
        return self._level["features"]

    def _drop_inputs(self):
        super()._drop_inputs()
        for k in ("data_wide", "data_ref"):
            self._cache.pop(k, None)
        self._level = None

    def _release(self, which):
        self._cache.pop(f"data_{which}", None)
        self._cache.pop("inputs", None)
        self._level = None

    # Context

    def _p13b(self):
        if "p13b" not in self._cache:
            doc = load_json(MODEL_DIR / "p13b_model.json")
            check(doc is not None, "p13b_model.json is missing: run " "p13b_arms first")
            self._cache["p13b"] = doc
        return self._cache["p13b"]

    def _part1_final(self, a_doc, b_doc):
        """(cell, lambda, theta, fingerprint, objective, selection score) of
        part 1's final model."""
        fin = b_doc["final"]
        i = int(fin["chosen"])
        cell = tuple(int(x) for x in fin["cell"][1::2])
        n12 = len(s13.LAMBDAS)
        score = float(fin["selection"][i]["mean"])
        if not fin["reuses_base_path"]:
            j = fin["path"][i]
            return (
                cell,
                float(j["lambda"]),
                np.asarray(j["theta"]),
                j["fingerprint"],
                float(j["fit"]["objective"]),
                score,
            )
        base = next(j for j in b_doc["cells"] if j["reuses_base_fit"])
        if i >= n12:
            j = b_doc["extension"]["fits"][i - n12]
            return (
                cell,
                float(j["lambda"]),
                np.asarray(j["theta"]),
                j["fingerprint"],
                float(j["fit"]["objective"]),
                score,
            )
        j = a_doc["path"][i]
        return (
            cell,
            float(j["lambda"]),
            np.asarray(j["theta"]),
            base["fingerprint"],
            float(j["fit"]["objective"]),
            score,
        )

    def _base_context(self):
        """The constants that every level shares: P13a's terms and layout
        constants, part 1's final model, the selection listeners."""
        if "base2" in self._cache:
            return self._cache["base2"]
        a_doc, a_arr = self._p13a()
        b_doc = self._p13b()
        check(
            a_doc["params"] == s13._params(),
            "p13a_model.json was built with other P13a parameters",
        )
        check(
            b_doc["params"] == s13b._params(),
            "p13b_model.json was built with other P13b parameters",
        )
        knots = {k: tuple(v) for k, v in a_doc["knots"].items()}
        base = s13.base_terms(knots)
        check(
            [t.to_json() for t in base] == a_doc["terms"],
            "the base terms differ from P13a's",
        )
        chosen, lam, theta1, fp1, obj1, score1 = self._part1_final(a_doc, b_doc)
        check(
            chosen[1] == 0,
            "part 2 supports a chosen cell without " "work_affinity only",
        )
        twin = flip_monotone(chosen)
        union = [
            t
            for t in base
            if not (t.name == "popularity_tilt" and not chosen[3])
            and not (t.name == "audio_fit_fuzzy_dur" and not chosen[2])
        ]
        lay_meta = a_doc["layout"]
        refm = a_doc["reference_means"]
        names = [t.name for t in union]
        n_zero = {n: int(lay_meta["n_zero"][n]) for n in names}
        n_miss = {n: int(lay_meta["n_miss"][n]) for n in names}
        neutral = [n for n in lay_meta["neutral"] if n in names]
        data_means = {n: np.asarray(v) for n, v in refm["data"].items() if n in names}
        dev_means = {n: np.asarray(v) for n, v in refm["dev"].items() if n in names}
        union_lay = gam.Layout(union, n_zero, n_miss, neutral, data_means, dev_means)
        ctx = {
            "a_doc": a_doc,
            "a_arr": a_arr,
            "b_doc": b_doc,
            "union": union,
            "union_layout": union_lay,
            "L_u": union_lay.L,
            "union_off": {t.name: int(o) for t, o in zip(union, union_lay.offsets)},
            "n_zero": n_zero,
            "n_miss": n_miss,
            "neutral": neutral,
            "data_means": data_means,
            "dev_means": dev_means,
            "shape_q": a_doc["shape_quantiles"],
            "chosen": chosen,
            "twin": twin,
            "lambda": lam,
            "theta1": theta1,
            "objective1": obj1,
            "score1": score1,
        }
        lay_c, _ = self._cell_layout(ctx, chosen)
        check(
            gam_arms.layout_fingerprint(lay_c) == fp1,
            "the chosen cell's layout differs from part 1's final model",
        )
        twin_j = next(
            (j for j in b_doc["cells"] if j["cell"] == s13b.cell_label(twin)), None
        )
        check(twin_j is not None, "part 1 holds no fit of the twin cell")
        lay_t, _ = self._cell_layout(ctx, twin)
        check(
            gam_arms.layout_fingerprint(lay_t) == twin_j["fingerprint"],
            "the twin cell's layout differs from part 1's",
        )
        ctx["theta_twin1"] = np.asarray(twin_j["theta"])
        m_consts = a_doc["mainstreaminess"]
        m_users, m_raw = a_arr["m_users"], a_arr["m_raw"]
        sel_users, sel_picks = a_arr["sel_users"], a_arr["sel_picks"]
        ctx.update(
            m_consts=m_consts,
            m_users=m_users,
            m_raw=m_raw,
            sel_users=sel_users,
            sel_picks=sel_picks,
            m_sel=gam.mainstreaminess(
                m_raw[np.searchsorted(m_users, sel_users)], m_consts
            ),
        )
        self._cache["base2"] = ctx
        return ctx

    def _level_context(self, base, which, k, store):
        """The base context plus the rows of one level: the proposal
        densities, mainstreaminess and the stored background means (None
        when the store holds none)."""
        self._set_level(which, k)
        inp = self._inputs()
        tr, ru = inp["tr"], inp["row_user"]
        users = inp["p"]["user_id"][tr]
        m_users = base["m_users"]
        at = np.minimum(np.searchsorted(m_users, users), len(m_users) - 1)
        check(
            np.array_equal(m_users[at], users),
            "a training listener has no mainstreaminess in P13a",
        )
        n_slots = inp["n_slots"]
        item_ids, p_pop = self._proposals()
        p_pop_slot = np.zeros(n_slots, dtype=np.float64)
        p_pop_slot[item_ids] = p_pop
        q_u = 1.0 / inp["p"]["n_allowed"][tr][ru].astype(np.float64)
        q_p = p_pop_slot[inp["item"]] / inp["p"]["pop_allowed"][tr][ru]
        pos = inp["kind"] == fp.KIND_POSITIVE
        ctx = dict(base)
        ctx.update(
            level=(which, int(k)),
            level_id=self._level["id"],
            rows=int(len(inp["rows"])),
            listeners=int(len(tr)),
            q_u=q_u,
            q_p=q_p,
            m_row=gam.mainstreaminess(base["m_raw"][at], base["m_consts"])[ru],
            pos_no_pop=int(np.count_nonzero(q_p[pos] == 0.0)),
            means_ref=(store.means(self._level["id"]) if store is not None else None),
        )
        return ctx

    def _level_means(self, ctx, slot, w, store):
        """The level's background means over its p_pop negatives, from the
        encoded rows, saved in the store."""
        inp = self._inputs()
        means, _, _ = s13.reference_means(
            ctx["union"], slot, w, inp["kind"] == fp.KIND_POP
        )
        ctx["means_ref"] = means
        if store is not None:
            store.put_means(ctx["level_id"], means)

    def _ensure_means(self, ctx, store):
        """The background means when neither the store nor a design gave
        them (a resumed level whose fits are all saved)."""
        if ctx["means_ref"] is not None:
            return
        inp = self._inputs()
        slot, w = s13.encode_rows(ctx["union"], inp["f"], inp["rows"])
        self._level_means(ctx, slot, w, store)
        del slot, w

    # Design

    @contextlib.contextmanager
    def _design(self, ctx, store=None):
        """Part 1's design with a block count that grows with the rows,
        so each worker's share of a Hessian stays near part 1's. The
        level's background means come from the same encoding when the
        store holds none."""
        for k in ("space", "sources"):
            self._cache.pop(k, None)
        inp = self._inputs()
        slot, w = s13.encode_rows(ctx["union"], inp["f"], inp["rows"])
        if ctx.get("means_ref") is None:
            self._level_means(ctx, slot, w, store)
        blocks = s13.make_blocks(
            ctx["union"],
            ctx["union_layout"],
            slot,
            w,
            ctx["m_row"],
            inp["offset"],
            inp["lens"],
            inp["n_pos"],
            fit_blocks(len(inp["rows"])),
            inp["m_neg"],
        )
        lens = inp["lens"]
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        for blk in blocks:
            lo, hi = blk["users"]
            r0, r1 = int(starts[lo]), int(starts[hi - 1] + lens[hi - 1])
            blk["q_u"] = ctx["q_u"][r0:r1]
            blk["q_p"] = ctx["q_p"][r0:r1]
        s13b._CTX.update(blocks=blocks, L=ctx["L_u"])
        try:
            with s13._Pool(s13.FIT_WORKERS) as pool:
                yield pool
        finally:
            s13b._CTX.clear()
            del blocks, slot, w

    @contextlib.contextmanager
    def _design_for(self, ctx, store, fids):
        if all(fid in store for fid in fids):
            yield None
            return
        with self._design(ctx, store) as pool:
            yield pool

    def _fit2(self, pool, ctx, store, cell, arm, theta0):
        which, k = ctx["level"]
        fid = fid2(which, k, cell, arm, ctx["lambda"])
        if fid in store:
            return store.get(fid)
        check(pool is not None, f"fit {fid} is not stored and no design " f"is loaded")
        lay, idx = self._cell_layout(ctx, cell)
        obj = self._objective(pool, ctx, lay, idx, arm, ctx["lambda"])
        label = (
            f"{which} level {k} (m = {n_negatives(arm, k)}), cell "
            f"{s13b.cell_label(cell)}, arm {s13b.arm_label(arm)}"
        )
        t0 = time.time()
        obj.hess_fn = _step_lines(obj.hess_fn, label, t0)
        if any(lo is not None or hi is not None for lo, hi in lay.bounds):
            res = gam_arms.fit_bounded(obj, theta0, s13b.SOLVER)
        else:
            res = gam.fit(obj, theta0, s13b.SOLVER)
        res["seconds"] = time.time() - t0
        store.put(fid, res)
        log.info(
            "p13b2: %s: objective %.8f, largest projected gradient " "%.1e, %s, %s",
            label,
            res["objective"],
            res["proj_grad_max"],
            res["message"],
            _clock(res["seconds"]),
        )
        return store.get(fid)

    # Run

    def _run_level(self, base, which, k, cells, arms, first, store, prev, prog, out):
        """The fits of one level, their records in out, and the level's
        row counts. prev holds the fits one level below and receives this
        level's."""
        ctx = self._level_context(base, which, k, store)
        fids = [fid2(which, k, c, a, base["lambda"]) for c in cells for a in arms]
        cur, recs = {}, []
        with self._design_for(ctx, store, fids) as pool:
            for cell in cells:
                for arm in arms:
                    start = warm_start(prev, cur, cell, arm, first[cell])
                    rec = self._fit2(pool, ctx, store, cell, arm, start)
                    cur[(cell, arm)] = rec["theta"]
                    recs.append((cell, arm, rec))
                    prog.advance(ctx["rows"])
        self._ensure_means(ctx, store)
        for cell, arm, rec in recs:
            out.append(self._fit_doc(ctx, rec, cell, arm))
        prev.clear()
        prev.update(cur)
        log.info(
            "p13b2: %s level %d done: %s rows, %s listeners",
            which,
            k,
            f"{ctx['rows']:,}",
            f"{ctx['listeners']:,}",
        )
        return {
            "rows": ctx["rows"],
            "listeners": ctx["listeners"],
            "pos_no_pop": ctx["pos_no_pop"],
        }

    def _union_tables(self, base, j):
        cell = tuple(int(x) for x in j["cell"][1::2])
        _, idx = self._cell_layout(base, cell)
        Tu = np.zeros(base["L_u"], dtype=np.float64)
        Tu[idx] = np.asarray(j["tables"])
        return Tu

    def _pass(self, ctx, label, tables_u, work, key):
        """A selection pass over configurations given as union tables:
        (ranks, slots scored per listener, seconds)."""
        log.info(
            "p13b2: selection pass '%s' over %d configurations", label, len(tables_u)
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

    def run(self, params, files):
        t_start = time.time()
        check(s13.RANK_CUT >= 100, "RANK_CUT must be at least 100")
        outputs = self.outputs()
        work = SCRATCH_DIR / "p13b2"
        work.mkdir(parents=True, exist_ok=True)
        key = self._key()
        base = self._base_context()
        store = _Store(work / "fits.json", s13._sha(key))
        chosen, twin = base["chosen"], base["twin"]
        log.info(
            "p13b2: part 1's final model: cell %s, lambda %.3g. Twin " "cell %s",
            s13b.cell_label(chosen),
            base["lambda"],
            s13b.cell_label(twin),
        )
        plan = [("ref", k, (chosen,), ref_arms(k)) for k in REF_LEVELS] + [
            ("wide", k, (chosen, twin), FULL_ARMS) for k in FULL_LEVELS
        ]
        sizes = {w: self._level_rows(w) for w in ("ref", "wide")}
        work_rows = sum(
            (sizes[w][1] + 2 * k * sizes[w][0]) * len(c) * len(a) for w, k, c, a in plan
        )
        n_fits = sum(len(c) * len(a) for _, _, c, a in plan)
        log.info(
            "p13b2: %d fits over %d levels, %s rows of fit work in all",
            n_fits,
            len(plan),
            f"{work_rows:,}",
        )
        prog = Progress(
            f"p13b2: fits ({n_fits})", work_rows, log.info, unit=" M rows", scale=10**6
        )
        data_rows = {}
        fits = {"ref": [], "wide": []}
        firsts = {chosen: base["theta1"], twin: base["theta_twin1"]}
        prev = {}
        for i, (which, k, cells, arms) in enumerate(plan):
            if i > 0 and plan[i - 1][0] != which:
                self._release(plan[i - 1][0])
                prev = {}
            data_rows[f"{which}:{k}"] = self._run_level(
                base, which, k, cells, arms, firsts, store, prev, prog, fits[which]
            )
        prog.finish()
        wide_fits = fits["wide"]
        tables_u = [self._union_tables(base, j) for j in wide_fits]
        ranks, scored, secs = self._pass(base, "part2", tables_u, work, key)
        scores = ga.per_listener_scores(ranks)
        doc = self._document(
            base, fits["ref"], wide_fits, scores, data_rows, scored, secs
        )
        doc["seconds"] = time.time() - t_start
        atomic_json_write(doc, outputs["model"])
        s13._atomic_savez(
            outputs["arrays"],
            sel_users=base["sel_users"],
            ranks=ranks.astype(np.int32),
            scored=scored.astype(np.int64),
        )
        d = doc["decisions"]
        log.info(
            "p13b2: bias of form A %.3f, form B %.3f (tau %.2f). "
            "Chosen: form %s, arm %s, level %d (m = %d), cell %s. Done "
            "in %s",
            d["bias"]["A"],
            d["bias"]["B"],
            TAU,
            d["form"],
            "-".join(d["arm"]),
            d["level"],
            d["m"],
            d["cell"],
            _clock(time.time() - t_start),
        )
        return self._info(outputs)

    def _fit_doc(self, ctx, rec, cell, arm):
        which, k = ctx["level"]
        j = self._fit_json(ctx, rec, cell, arm, ctx["lambda"])
        j["data"] = which
        j["level"] = int(k)
        j["m"] = n_negatives(arm, k)
        j["points"] = shape_points(j["terms"], j["tables_raw"], ctx["shape_q"])
        return j

    def _document(self, base, ref_fits, wide_fits, scores, data_rows, scored, secs):
        chosen, twin = base["chosen"], base["twin"]
        ref_pts = {
            (j["level"], tuple(j["arm"].split("-"))): j["points"] for j in ref_fits
        }
        full_means = {
            (
                tuple(int(x) for x in j["cell"][1::2]),
                tuple(j["arm"].split("-")),
                j["level"],
            ): float(s.mean())
            for j, s in zip(wide_fits, scores)
        }
        dec = decide2(ref_pts, full_means, chosen, twin)
        R = ref_pts[(max(REF_LEVELS), ("mix32", "A"))]
        for j in ref_fits:
            j["gap_to_reference"] = points_gap(j["points"], R)
        best_k = int(np.argmax(scores.mean(axis=1)))
        n = scores.shape[1]
        for i, j in enumerate(wide_fits):
            d = scores[best_k] - scores[i]
            j["selection"] = {
                "mean": float(scores[i].mean()),
                "diff_to_best": float(d.mean()),
                "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
            }
        final = next(
            j
            for j in wide_fits
            if j["cell"] == dec["cell"]
            and j["arm"] == "-".join(dec["arm"])
            and j["level"] == dec["level"]
        )
        n_slots = int(self._id_slots()[0])
        return {
            "params": _params(),
            "code": self._code(),
            "upstream": self._upstream_outputs(),
            "part1": {
                "cell": s13b.cell_label(chosen),
                "twin": s13b.cell_label(twin),
                "lambda": base["lambda"],
                "objective": base["objective1"],
                "selection_mean": base["score1"],
            },
            "data": data_rows,
            "ref_fits": ref_fits,
            "wide_fits": wide_fits,
            "decisions": dec,
            "final": {
                "index": wide_fits.index(final),
                "cell": final["cell"],
                "arm": final["arm"],
                "level": final["level"],
                "m": final["m"],
                "lambda": base["lambda"],
            },
            "best_index": best_k,
            "selection_pass": {
                "configurations": len(wide_fits),
                "listeners": int(len(scored)),
                "full_pass_share": float(np.mean(scored == n_slots)),
                "mean_scored_share": float(np.mean(scored) / n_slots),
                "seconds": float(secs),
            },
        }

    def verify(self, params):
        self._verified = None
        doc, arr = self._load()
        check(
            doc["params"] == _params(),
            "p13b2_model.json was built with other parameters",
        )
        base = self._base_context()
        gaps = {
            "kkt": 0.0,
            "objective": 0.0,
            "tables": 0.0,
            "centering": 0.0,
            "monotone": 0.0,
            "offsets": 0.0,
        }
        plan = [
            ("ref", k, [j for j in doc["ref_fits"] if j["level"] == k])
            for k in REF_LEVELS
        ] + [
            ("wide", k, [j for j in doc["wide_fits"] if j["level"] == k])
            for k in FULL_LEVELS
        ]
        check(
            sum(len(js) for _, _, js in plan)
            == len(doc["ref_fits"]) + len(doc["wide_fits"])
            and all(
                len(js) == (len(ref_arms(k)) if w == "ref" else 2 * len(FULL_ARMS))
                for w, k, js in plan
            ),
            "the stored fits do not match the levels and arms of the " "stage",
        )
        prog = Progress(
            "p13b2 verify: every fit on its level's rows",
            len(plan),
            log.info,
            unit=" levels",
        )
        for i, (which, k, js) in enumerate(plan):
            ctx = self._level_context(base, which, k, None)
            arms = ref_arms(k) if which == "ref" else FULL_ARMS
            gaps["offsets"] = max(gaps["offsets"], self._verify_offsets2(ctx, arms))
            with self._design(ctx) as pool:
                for j in js:
                    self._verify_fit(pool, ctx, j, gaps)
                if (which, k) == ("wide", min(FULL_LEVELS)):
                    gaps["part1"] = self._verify_part1(pool, ctx, doc)
            if i + 1 == len(plan) or plan[i + 1][0] != which:
                self._release(which)
            del ctx
            prog.advance(1)
        prog.finish()
        self._verify_decisions(doc, arr, base)
        gaps.update(self._verify_ranks2(doc, arr, base))
        self._verified = gaps
        log.info(
            "p13b2: verify passed. Largest gaps: %s",
            ", ".join(f"{k} {v:.1e}" for k, v in gaps.items()),
        )

    def _verify_fit(self, pool, ctx, j, gaps):
        cell = tuple(int(x) for x in j["cell"][1::2])
        arm = tuple(j["arm"].split("-"))
        lay, idx = self._cell_layout(ctx, cell)
        label = f"{j['data']} level {j['level']} {j['cell']} {j['arm']}"
        check(
            gam_arms.layout_fingerprint(lay) == j["fingerprint"],
            f"{label}: the layout differs",
        )
        check(
            j["lambda"] == ctx["lambda"] and j["m"] == n_negatives(arm, j["level"]),
            f"{label}: the weight or the number of "
            f"negatives differs from the stage's",
        )
        theta = np.asarray(j["theta"], dtype=np.float64)
        obj = self._objective(pool, ctx, lay, idx, arm, j["lambda"])
        loss, smooth, ridge, grad = obj.parts(theta)
        F = loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge
        rel = abs(F - j["fit"]["objective"]) / max(1.0, abs(F))
        pg = float(np.max(np.abs(gam.projected_gradient(theta, grad, lay.bounds))))
        gaps["objective"] = max(gaps["objective"], rel)
        gaps["kkt"] = max(gaps["kkt"], pg)
        check(
            pg <= s13.KKT_TOL,
            f"{label}: the largest projected gradient " f"is {pg:.2e}",
        )
        check(
            rel <= s13.OBJ_RTOL,
            f"{label}: the objective differs from " f"its recomputation by {rel:.2e}",
        )
        T, Tc, _, _ = self._centered(ctx, lay, idx, theta)
        gaps["tables"] = max(
            gaps["tables"], float(np.max(np.abs(T - np.asarray(j["tables_raw"]))))
        )
        check(
            gaps["tables"] == 0.0,
            f"{label}: the stored raw tables " f"differ from M theta",
        )
        gaps["centering"] = max(
            gaps["centering"], float(np.max(np.abs(Tc - np.asarray(j["tables"]))))
        )
        check(
            gaps["centering"] <= CENTER_TOL,
            f"{label}: the stored centered tables differ from the tables "
            f"centered on the level's p_pop negatives by "
            f"{gaps['centering']:.2e}",
        )
        if any(t.monotone for t in lay.terms):
            gaps["monotone"] = max(
                gaps["monotone"], gam_arms.monotone_violation(T, lay)
            )
            check(
                gaps["monotone"] <= s13b.MONOTONE_TOL,
                f"{label}: the monotone order breaks by " f"{gaps['monotone']:.2e}",
            )
        pts = shape_points(j["terms"], j["tables_raw"], ctx["shape_q"])
        check(
            json.dumps(pts, sort_keys=True) == json.dumps(j["points"], sort_keys=True),
            f"{label}: the stored explanation points differ from the " f"tables",
        )

    def _verify_offsets2(self, ctx, arms):
        """The negative columns, offsets and positive weights of each arm
        of the level on sampled listeners, against a computation from the
        stored n_allowed and pop_allowed and P3's p_pop (read through a
        dictionary). Returns the largest offset gap."""
        inp = self._inputs()
        p = inp["p"]
        which, k = ctx["level"]
        item_ids, p_pop = self._proposals()
        pp = dict(zip(item_ids.tolist(), p_pop.tolist()))
        lens, n_pos = inp["lens"], inp["n_pos"]
        starts = np.concatenate(([0], np.cumsum(lens)[:-1]))
        rng = np.random.default_rng(s13.SEED + 21)
        users = np.sort(
            rng.choice(
                len(lens), size=min(VERIFY_OFFSET_USERS, len(lens)), replace=False
            )
        )
        want_kinds = {
            "mix32": np.repeat([fp.KIND_UNIFORM, fp.KIND_POP], k),
            "p_pop": np.full(k, fp.KIND_POP),
        }
        worst = 0.0
        for u in users:
            r0, n, npu = int(starts[u]), int(lens[u]), int(n_pos[u])
            check(
                n == npu + 2 * k,
                f"{which} level {k}: a listener does "
                f"not hold its positives and 2k draws",
            )
            na = float(p["n_allowed"][inp["tr"][u]])
            pa = float(p["pop_allowed"][inp["tr"][u]])
            items = inp["item"][r0 : r0 + n]
            kinds = inp["kind"][r0 : r0 + n]
            qu = np.full(n, 1.0 / na)
            qp = np.array([pp.get(int(i), 0.0) for i in items]) / pa
            blk = {
                "q_u": ctx["q_u"][r0 : r0 + n],
                "q_p": ctx["q_p"][r0 : r0 + n],
                "pos_rows": np.arange(npu),
                "pos_weight": np.ones(npu),
                "neg_rows": (npu + np.arange(2 * k))[None, :],
            }
            for arm in arms:
                got = s13b.arm_block(blk, arm)
                m = n_negatives(arm, k)
                check(
                    got["neg_rows"].shape == (1, m)
                    and np.array_equal(kinds[got["neg_rows"][0]], want_kinds[arm[0]]),
                    f"{which} level {k}, arm {s13b.arm_label(arm)}: the "
                    f"arm takes other negatives than its {m} draws",
                )
                q = qp if arm[0] == "p_pop" else 0.5 * qu + 0.5 * qp
                with np.errstate(divide="ignore"):
                    off = -np.log(m * q)
                keep = q > 0.0
                if arm[1] == "A":
                    keep &= np.arange(n) >= npu
                want = np.where(keep, off, 0.0)
                worst = max(worst, float(np.max(np.abs(got["offset"] - want))))
                want_w = (
                    np.where(q[:npu] > 0.0, 1.0, 0.0) if arm[1] == "B" else np.ones(npu)
                )
                check(
                    np.array_equal(got["pos_weight"], want_w),
                    f"{which} level {k}, arm {s13b.arm_label(arm)}: a "
                    f"positive has the wrong weight",
                )
        check(
            worst <= 1e-12,
            f"{which} level {k}: an arm offset differs "
            f"from the stored values by {worst:.2e}",
        )
        return worst

    def _verify_part1(self, pool, ctx, doc):
        """Part 1's final fit is the chosen cell's fit at the smallest
        wide level with mixture form A."""
        chosen = ctx["chosen"]
        j = next(
            j
            for j in doc["wide_fits"]
            if j["level"] == min(FULL_LEVELS)
            and j["cell"] == s13b.cell_label(chosen)
            and j["arm"] == "mix32-A"
        )
        lay, idx = self._cell_layout(ctx, chosen)
        obj = self._objective(pool, ctx, lay, idx, ("mix32", "A"), ctx["lambda"])
        loss, smooth, ridge, grad = obj.parts(ctx["theta1"])
        F = loss + obj.lam_smooth * smooth + obj.lam_ridge * ridge
        rel = abs(F - ctx["objective1"]) / max(1.0, abs(F))
        check(
            rel <= s13.OBJ_RTOL,
            f"part 1's final model has objective {ctx['objective1']!r} "
            f"in part 1 and {F!r} on the smallest wide level",
        )
        d_theta = float(np.max(np.abs(np.asarray(j["theta"]) - ctx["theta1"])))
        check(
            d_theta <= s13.TWO_RUN_TOL,
            f"the chosen cell's fit at the smallest wide level differs "
            f"from part 1's final fit by {d_theta:.2e}",
        )
        return max(rel, d_theta)

    def _verify_decisions(self, doc, arr, base):
        ranks = arr["ranks"]
        check(
            ranks.shape == (len(doc["wide_fits"]), len(base["sel_users"]), 2)
            and np.array_equal(arr["sel_users"], base["sel_users"]),
            "the stored ranks do not match the fits and the selection " "listeners",
        )
        scores = ga.per_listener_scores(ranks)
        best_k = int(np.argmax(scores.mean(axis=1)))
        check(
            best_k == doc["best_index"],
            "the stored best configuration does not follow from the ranks",
        )
        n = scores.shape[1]
        for i, (j, s) in enumerate(zip(doc["wide_fits"], scores)):
            d = scores[best_k] - s
            want = {
                "mean": float(s.mean()),
                "diff_to_best": float(d.mean()),
                "se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
            }
            check(
                json.dumps(j["selection"], sort_keys=True)
                == json.dumps(want, sort_keys=True),
                f"{j['cell']} {j['arm']} level {j['level']}: the stored "
                f"score does not follow from the ranks",
            )
        ref_pts = {
            (j["level"], tuple(j["arm"].split("-"))): j["points"]
            for j in doc["ref_fits"]
        }
        R = ref_pts[(max(REF_LEVELS), ("mix32", "A"))]
        for j in doc["ref_fits"]:
            check(
                j["gap_to_reference"] == points_gap(j["points"], R),
                f"ref level {j['level']} {j['arm']}: the stored gap to the "
                f"reference does not follow from the points",
            )
        full_means = {
            (
                tuple(int(x) for x in j["cell"][1::2]),
                tuple(j["arm"].split("-")),
                j["level"],
            ): float(s.mean())
            for j, s in zip(doc["wide_fits"], scores)
        }
        dec = decide2(ref_pts, full_means, base["chosen"], base["twin"])
        check(
            json.dumps(dec, sort_keys=True)
            == json.dumps(doc["decisions"], sort_keys=True),
            "the stored decisions do not follow from the stored tables and " "ranks",
        )
        fin = doc["wide_fits"][doc["final"]["index"]]
        check(
            fin["cell"] == dec["cell"]
            and fin["arm"] == "-".join(dec["arm"])
            and fin["level"] == dec["level"]
            and doc["final"]["cell"] == fin["cell"]
            and doc["final"]["arm"] == fin["arm"]
            and doc["final"]["level"] == fin["level"]
            and doc["final"]["m"] == fin["m"]
            and doc["final"]["lambda"] == fin["lambda"],
            "the final model is not the one the decisions name",
        )

    def _verify_ranks2(self, doc, arr, base):
        """Sampled ranks of three configurations against the dense
        reference: the final model, the best one and the last one."""
        n_slots, _ = self._id_slots()
        src = self._sources(n_slots)
        space = self._space(src)
        u_indptr, u_indices, u_data = self._csr()
        users, picks = base["sel_users"], base["sel_picks"]
        pick_k = sorted(
            {
                int(doc["final"]["index"]),
                int(doc["best_index"]),
                len(doc["wide_fits"]) - 1,
            }
        )
        tables = np.stack(
            [self._union_tables(base, doc["wide_fits"][i]) for i in pick_k]
        )
        cut1 = s13.RANK_CUT + 1
        rng = np.random.default_rng(s13.SEED + 22)
        take = np.sort(
            rng.choice(
                len(users), size=min(VERIFY_RANK_USERS, len(users)), replace=False
            )
        )
        near = 0
        prog = Progress(
            "p13b2 verify: selection ranks against the dense " "reference",
            len(take),
            log.info,
            unit=" listeners",
        )
        for jj in take:
            u = int(users[jj])
            items, weights = cd.history_of(u_indptr, u_indices, u_data, u)
            state = cn.build_user_state(items, weights, space)
            full = cn.full_catalog(state, space)
            ref = ct.reference_scores(
                full, base["union"], tables, float(base["m_sel"][jj]), space.tier
            )
            del full
            for c, i in enumerate(pick_k):
                for p in range(2):
                    lo, hi = ct.reference_rank(
                        ref[c], picks[jj, p], items, s13.SCORE_TOL
                    )
                    near += int(hi > lo)
                    got = int(arr["ranks"][i, jj, p])
                    check(
                        min(lo, cut1) <= got <= min(hi, cut1),
                        f"listener {u}, fit {i}, pick {p}: the stored rank "
                        f"{got} is outside the reference range [{lo}, "
                        f"{hi}] censored at {cut1}",
                    )
            prog.advance(1)
        prog.finish()
        return {"near_ties": float(near)}

    def report(self, params):
        doc, arr = self._load()
        base = self._base_context()
        lines = []
        add = lines.append
        add("# P13b report, part 2: the loss form and the number of " "negatives")
        add("")
        add(f"- parameters: {json.dumps(doc['params'])}")
        p1 = doc["part1"]
        add(
            f"- part 1's final model: cell {p1['cell']}, lambda "
            f"{p1['lambda']:.3g}, selection score "
            f"{p1['selection_mean']:.5f}. Twin cell: {p1['twin']}."
        )
        if self._verified:
            add(
                "- verify(): "
                + ", ".join(f"{k} {v:.1e}" for k, v in self._verified.items())
                + ". kkt is the largest projected gradient over all fits, "
                "objective the largest relative gap to a recomputation, "
                "tables the gap between the stored tables and M theta, "
                "centering the gap between the stored centered tables and "
                "a centering on the level's rows, monotone the largest "
                "break of a monotone order, offsets the largest gap of an "
                "arm offset, part1 the gap between part 1's final fit and "
                "its refit here, near_ties the checked ranks with a score "
                "within the tolerance of the pick's."
            )
        sp = doc["selection_pass"]
        add(
            f"- run time: {_clock(doc['seconds'])}. Selection pass: "
            f"{sp['configurations']} configurations, {sp['listeners']:,} "
            f"listeners, {100 * sp['full_pass_share']:.1f} percent of them "
            f"over the whole catalog, {_clock(sp['seconds'])}."
        )
        add("")
        self._report_data(doc, add)
        self._report_reference(doc, base, add)
        self._report_full(doc, add)
        self._report_decisions(doc, add)
        self._report_final(doc, base, add)
        text = "\n".join(lines) + "\n"
        check(";" not in text, "the report holds a semicolon")
        atomic_text_write(text, self.outputs()["report"])
        log.info("p13b2 report written")

    def _report_data(self, doc, add):
        add("## 1. Data")
        add("")
        add(
            "A level keeps each listener's positives and the first k draws "
            "of each kind. The mixture arms use all 2k negatives of a "
            "level, the p_pop arms its k p_pop draws."
        )
        add("")
        add("| file | level k | listeners | rows | positives outside P3's " "core |")
        add("|---|---:|---:|---:|---:|")
        order = sorted(
            doc["data"],
            key=lambda key: (key.split(":")[0] != "ref", int(key.split(":")[1])),
        )
        for key in order:
            d = doc["data"][key]
            which, k = key.split(":")
            add(
                f"| {which} | {int(k):,} | {d['listeners']:,} | "
                f"{d['rows']:,} | {d['pos_no_pop']:,} |"
            )
        add("")
        add(
            "A positive outside P3's core cannot be a p_pop draw, so it "
            "gets weight 0 under p_pop form B."
        )
        add("")

    def _report_reference(self, doc, base, add):
        dec = doc["decisions"]
        add("## 2. The reference")
        add("")
        add(
            "The chosen cell on the reference listeners. Gap: the largest "
            "absolute difference of the explanation points (each term's "
            "values at the positives' percentiles, its zero and its "
            "missing constant, each minus its median value) to the "
            "reference R, mixture form A at the largest level. Popularity "
            "and missing constants in log-odds points, centered on the "
            "level's p_pop negatives."
        )
        add("")
        miss_terms = [t.name for t in base["union"] if t.missing_state]
        add(
            "| arm | m | gap to R | pop p5 | pop p25 | pop p50 | pop p75 | "
            "pop p95 | " + " | ".join(f"missing {n}" for n in miss_terms) + " |"
        )
        add("|---|---:|---:|" + "---:|" * (5 + len(miss_terms)))
        for j in doc["ref_fits"]:
            vals, _, _ = self._shape_values(base, j, "popularity")
            misses = [self._shape_values(base, j, n)[2] for n in miss_terms]
            add(
                f"| {j['arm']} | {j['m']:,} | {j['gap_to_reference']:.3f} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + " | "
                + " | ".join(misses)
                + " |"
            )
        add("")
        add(
            f"- D1: the four fits at the largest level differ by at most "
            f"{dec['g_ref']:.3f} log-odds points. p_pop form B gives no "
            f"weight to a positive outside P3's core, so part of this gap "
            f"can come from those positives."
        )
        add(
            f"- D2: bias at m = 32 (gap to R): form A {dec['bias']['A']:.3f},"
            f" form B {dec['bias']['B']:.3f}, tau {dec['tau']:.2f}."
        )
        add("")

    def _report_full(self, doc, add):
        add("## 3. The full data")
        add("")
        p1 = doc["part1"]
        j16 = next(
            j
            for j in doc["wide_fits"]
            if j["level"] == min(FULL_LEVELS)
            and j["cell"] == p1["cell"]
            and j["arm"] == "mix32-A"
        )
        add(
            f"Every training listener. Difference: the best score minus the "
            f"row's score, with the standard error of the paired "
            f"per-listener differences. The first row is part 1's loss on "
            f"part 1's rows: it scores {j16['selection']['mean']:.5f}, part "
            f"1's final model {p1['selection_mean']:.5f}."
        )
        add("")
        add("| cell | arm | m | score | difference to best | fit |")
        add("|---|---|---:|---:|---|---|")
        for j in doc["wide_fits"]:
            s = j["selection"]
            add(
                f"| {j['cell']} | {j['arm']} | {j['m']:,} | "
                f"{s['mean']:.5f} | {s['diff_to_best']:.5f} +- "
                f"{s['se']:.5f} | {j['fit']['message']} |"
            )
        add("")

    def _report_decisions(self, doc, add):
        d = doc["decisions"]
        add("## 4. Decisions")
        add("")
        add(
            f"- D2, the loss form: form {d['form']}, by {d['form_by']}. "
            f"Selection score of the chosen cell at m = 32: form A "
            f"{d['scores_m32']['A']:.5f}, form B {d['scores_m32']['B']:.5f}."
        )
        add(
            f"- D3, the negatives: arm {'-'.join(d['arm'])} at level "
            f"{d['level']} (m = {d['m']}), score {d['chosen_score']:.5f}. "
            f"Best score of the form {d['best']:.5f}, SESOI "
            f"{d['sesoi_abs']:.5f}."
        )
        add(
            f"- D4, the bounds: the twin scores {d['twin_score']:.5f} at the "
            f"same arm and level. Final cell: {d['cell']}."
        )
        add("")

    def _report_final(self, doc, base, add):
        fin = doc["final"]
        j = doc["wide_fits"][fin["index"]]
        add(
            f"## 5. The final model (cell {fin['cell']}, arm {fin['arm']}, "
            f"m = {fin['m']}, lambda {fin['lambda']:.3g})"
        )
        add("")
        add(
            "Values in log-odds points, centered on the p_pop negatives of "
            "its level, at percentiles of the positives' values. zero: data, "
            "but no match. missing: no data."
        )
        add("")
        add("| term | p5 | p25 | p50 | p75 | p95 | zero | missing |")
        add("|---|---:|---:|---:|---:|---:|---:|---:|")
        for t in j["terms"]:
            vals, zero, miss = self._shape_values(base, j, t["name"])
            add(
                f"| {t['name']} | "
                + " | ".join(f"{v:+.3f}" for v in vals)
                + f" | {zero} | {miss} |"
            )
        add("")


def _step_lines(hess_fn, label, t0):
    """hess_fn with a log line at most every PROGRESS_SECONDS: one Hessian
    is one Newton step, so the line tells how far a long fit has come. It
    only counts, so it cannot change a result."""
    state = {"steps": 0, "last": time.time()}

    def wrapped(T, D):
        H = hess_fn(T, D)
        state["steps"] += 1
        now = time.time()
        if now - state["last"] >= PROGRESS_SECONDS:
            state["last"] = now
            log.info(
                "p13b2: %s: Newton step %d done, %s in this fit",
                label,
                state["steps"],
                _clock(now - t0),
            )
        return H

    return wrapped
