"""Checks for pipeline/stage_p13b.py on the synthetic world of
p13_world.py (the world of test_p13a.py). Every path points into a
temporary folder and the loaders are replaced, so nothing touches the
real model folder. P13a runs first, in the same folder, as on the real
machine.

1. The helpers: the negative columns and the proposal density of each
   arm, the offsets and weights of forms A and B (a positive that the
   proposal cannot draw gets weight 0), the factorial contrasts, the
   decision rules, the fit store and the log relabelling.
2. run() writes the model, the arrays and a report, verify() passes, and
   the report holds every section. The pass log lines name P13b.
3. Every selection rank of every configuration of the three passes
   equals a direct count under the P5 rule on a dense scorer written
   here, from each fit's own terms and tables, censored at RANK_CUT + 1.
4. Recomputed here without the stage's code: the loss of every fit from
   the raw pairs (each arm's negatives, offsets and weights, BPR by its
   pairs), the KKT conditions of a monotone fit and of the BPR fit by
   finite differences of that loss, the orders of every monotone fit,
   lambda_base by the one-standard-error rule, the effects, the
   decisions and the final choice.
5. One worker and three workers give the same outputs.
6. Resume: a finished run fits nothing and scores nothing, and a lost fit
   and a lost selection block are computed again with the same result.
7. verify() fails for each planted fault in the outputs, among them
   two passes that disagree on a configuration they share.
8. The branch where the chosen cell is the base cell: the base path is
   the final path, and verify() passes.

Usage, from the repository root:
    python3 test_p13b.py
"""

import json
import logging
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import p13_world as W
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp
from pipeline import gam
from pipeline import gam_arms
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline.common import VerificationError

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)
        if os.environ.get("FAIL_FAST"):  # FAIL_FAST=1: stop at the first failed check
            raise SystemExit(1)


def raises(fn, text):
    try:
        fn()
    except VerificationError as exc:
        if text in str(exc):
            return True
        print(f"      other message: {exc}")
        return False
    except Exception as exc:  # reported, then False
        print(f"      unexpected {type(exc).__name__}: {exc}")
        return False
    return False


print(W.summary())
CUT1 = s13.RANK_CUT + 1
# Positives outside P3's core, as in the real data: the items that some
# positive holds but no p_pop draw holds leave the core. The p_pop draws
# keep their densities, so P11's pairs stay valid.
OUT_CORE = np.setdiff1d(
    np.unique(W.P["item"][W.P["kind"] == fp.KIND_POSITIVE]),
    np.unique(W.P["item"][W.P["kind"] == fp.KIND_POP]),
)
CORE = np.setdiff1d(W.core, OUT_CORE).astype(np.int64)
PP_SLOT = W.p_pop_slot.copy()
PP_SLOT[OUT_CORE] = 0.0
PP = PP_SLOT[CORE]

# 1: helpers
check(
    s13b.neg_cols("uniform", 16).tolist() == list(range(16))
    and s13b.neg_cols("p_pop", 16).tolist() == list(range(16, 32))
    and s13b.neg_cols("mixture", 16).tolist() == list(range(8)) + list(range(16, 24))
    and s13b.neg_cols("mix32", 16).tolist() == list(range(32)),
    "neg_cols: 16 uniform draws, 16 p_pop draws, the first 8 of each "
    "kind for the mixture, all 32 for BPR",
)
qa, qb = np.array([0.1, 0.2]), np.array([0.3, 0.0])
check(
    np.array_equal(s13b.arm_q(qa, qb, "uniform"), qa)
    and np.array_equal(s13b.arm_q(qa, qb, "p_pop"), qb)
    and np.allclose(s13b.arm_q(qa, qb, "mixture"), [0.2, 0.1], rtol=0, atol=1e-15),
    "arm_q: the proposal density of each arm, the mixture with equal " "weights",
)
# A toy block: listener 0 holds 2 positives (the second outside P3's
# core, q_p = 0) and listener 1 one positive, 4 negatives each (2 + 2).
toy = {
    "q_u": np.full(11, 0.01),
    "q_p": np.array([0.02, 0.0, 0.05, 0.05, 0.01, 0.02, 0.03, 0.04, 0.02, 0.06, 0.01]),
    "pos_rows": np.array([0, 1, 6]),
    "pos_user": np.array([0, 0, 1]),
    "pos_weight": np.array([0.25, 0.25, 0.5]),
    "neg_rows": np.array([[2, 3, 4, 5], [7, 8, 9, 10]]),
    "offset": np.zeros(11),
}
toy["q_u"][6:] = 0.02
got_a = s13b.arm_block(toy, ("p_pop", "A"))
got_b = s13b.arm_block(toy, ("p_pop", "B"))
got_n = s13b.arm_block(toy, ("uniform", "none"))
got_bpr = s13b.arm_block(toy, ("mix32", "bpr"))
want_a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
for r in (4, 5, 9, 10):  # the p_pop columns of each listener
    want_a[r] = -math.log(2 * toy["q_p"][r])
for r in (2, 3, 7, 8):  # uniform rows: not this arm's
    want_a[r] = -math.log(2 * toy["q_p"][r])
want_b = np.array([-math.log(2 * q) if q > 0 else 0.0 for q in toy["q_p"]])
check(
    got_a["neg_rows"].tolist() == [[4, 5], [9, 10]]
    and np.allclose(got_a["offset"], want_a, rtol=0, atol=1e-15)
    and np.array_equal(got_a["pos_weight"], toy["pos_weight"]),
    "arm_block, form A: the arm's own negatives, -log(m q) on every "
    "row that is not a positive, 0 on the positives",
)
check(
    np.allclose(got_b["offset"], want_b, rtol=0, atol=1e-15)
    and got_b["pos_weight"].tolist() == [0.25, 0.0, 0.5],
    "arm_block, form B: -log(m q) on every row, the positives included, "
    "and weight 0 for the positive that the proposal cannot draw",
)
check(
    not got_n["offset"].any()
    and got_n["neg_rows"].tolist() == [[2, 3], [7, 8]]
    and "offset" in got_bpr
    and got_bpr["neg_rows"].shape == (2, 4)
    and s13b.arm_block(toy, None) is toy,
    "arm_block: no correction gives zero offsets, BPR keeps all "
    "negatives, the base arm is the block itself",
)
sc_toy = np.random.default_rng(5).random((16, 40))
eff = s13b.main_effects(sc_toy, s13b.CELLS)
ok_eff = True
for i, f in enumerate(s13b.FACTORS):
    on = [k for k, c in enumerate(s13b.CELLS) if c[i]]
    off = [k for k, c in enumerate(s13b.CELLS) if not c[i]]
    d = sc_toy[on].mean(axis=0) - sc_toy[off].mean(axis=0)
    ok_eff &= (
        abs(eff[f]["mean"] - d.mean()) < 1e-15
        and abs(eff[f]["se"] - d.std(ddof=1) / math.sqrt(40)) < 1e-15
    )
    for j in range(i + 1, 4):
        g = s13b.FACTORS[j]
        same = [k for k, c in enumerate(s13b.CELLS) if c[i] == c[j]]
        diff = [k for k, c in enumerate(s13b.CELLS) if c[i] != c[j]]
        d2 = sc_toy[same].mean(axis=0) - sc_toy[diff].mean(axis=0)
        ok_eff &= abs(eff[f"{f} x {g}"]["mean"] - d2.mean()) < 1e-15
check(
    ok_eff and len(eff) == 10,
    "main_effects: each main effect is the mean over listeners of the "
    "cells with the factor minus the cells without it, each interaction "
    "the cells where two factors agree minus those where they differ",
)


def fake(m, w, t, p):
    return {
        "monotone": {"mean": m},
        "work_affinity": {"mean": w},
        "tier_curve": {"mean": t},
        "tilt": {"mean": p},
    }


thr = 0.02 * 0.05
check(
    s13b.decide(fake(-0.9 * thr, 1.1 * thr, 0.9 * thr, -1.0), 0.05)[0] == (1, 1, 0, 0)
    and s13b.decide(fake(-1.1 * thr, 0.0, 2 * thr, 1.01 * thr), 0.05)[0]
    == (0, 0, 1, 1),
    "decide: the bounds stay unless they cost more than 2 percent of the "
    "base score, the other factors stay only on a gain above it",
)
ids = (
    {s13b.fit_id(c, None, 0.1) for c in s13b.CELLS}
    | {s13b.fit_id(s13b.BASE_CELL, a, 0.1) for a in s13b.ARMS}
    | {s13b.fit_id(s13b.BASE_CELL, None, lam) for lam in (1e-5, 10**-5.5)}
)
check(
    len(ids) == 16 + 10 + 2,
    "fit_id is unique over the cells, the arms " "and the weights",
)
tmp_store = Path(tempfile.mkdtemp(prefix="test_p13b_store_"))
st_a = s13b._FitStore(tmp_store / "fits.json", "key-a")
st_a.put("x", {"theta": np.array([1.5, -2.0]), "objective": 3.0})
st_a2 = s13b._FitStore(tmp_store / "fits.json", "key-a")
st_b = s13b._FitStore(tmp_store / "fits.json", "key-b")
check(
    "x" in st_a2
    and st_a2.get("x")["theta"].tolist() == [1.5, -2.0]
    and "x" not in st_b,
    "the fit store keeps a fit under the same key and starts empty "
    "under another key",
)
rec = logging.LogRecord(
    "pipeline",
    logging.INFO,
    "",
    0,
    "p13a: swap. Start again with P13A_WORKERS=8 python3 "
    "-m pipeline.runner --only p13a_fit.",
    (),
    None,
)
s13b._Relabel().filter(rec)
check(
    rec.msg == "p13b: swap. Start again with P13A_WORKERS=8 python3 -m "
    "pipeline.runner --only p13b_arms.",
    "the log filter renames a P13a pass line and its advice",
)

# Stage harness
TMP = Path(tempfile.mkdtemp(prefix="test_p13b_"))


def configure(workers=1, sub="run"):
    root = TMP / sub
    for m in (s13, s13b):
        m.MODEL_DIR = root / "model"
        m.SCRATCH_DIR = root / "scratch"
    s13.LAMBDAS = (1.0, 0.1, 0.01, 1e-3)
    s13.N_BOOT = 30
    s13.FIT_BLOCKS = 5
    s13.FIT_WORKERS = workers
    s13.SELECTION_WORKERS = workers
    s13.TASKS_PER_WORKER = 2
    s13.VERIFY_USERS = 6
    s13.VERIFY_ROWS = 4000
    s13.VERIFY_MAINSTREAM = 10_000
    s13.NEUTRAL_MARGIN = 0.02
    s13.SELECTION_BLOCK = 7
    s13.SELECTION_CHUNK = 64
    s13.PROBE_USERS = 3
    s13b.EXT_LAMBDAS = (1e-4, 0.0)
    s13b.VERIFY_OFFSET_USERS = 60
    return root


def loaders(st):
    st._pairs = lambda: (W.P, W.P_PARAMS, "pairs-sha")
    st._features = lambda: (W.F, W.F_PARAMS, "pairs-sha", "features-sha")
    st._proposals = lambda: (CORE.copy(), PP.copy())
    st._id_slots = lambda: (W.NI, 1)
    st._csr = lambda: (W.u_indptr, W.u_indices, W.u_data)
    st._popularity = lambda n: W.popularity.copy()
    st._selection = lambda: (W.SEL.copy(), W.SEL_PICKS.copy())
    st._sources = lambda n: W.SRC
    st._space = lambda src: W.SPACE
    st._als_reference = lambda: {
        "popularity": 0.003,
        "als": 0.017,
        "als_label": "alpha 1, reg 100",
    }
    st._upstream_outputs = lambda: {"p11_fusion_pairs": {"pairs": "x"}}
    return st


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


def run_b():
    """A fresh P13b stage: run, verify, report, and the log lines."""
    cap = Capture()
    s13.log.addHandler(cap)
    s13.log.setLevel(logging.INFO)
    try:
        st = loaders(s13b.StageP13b())
        st.run({}, [])
        st.verify({})
        st.report({})
    finally:
        s13.log.removeHandler(cap)
    return st, cap.lines


def load(root):
    doc = json.loads((root / "model" / "p13b_model.json").read_text())
    z = np.load(root / "model" / "p13b_arrays.npz")
    return doc, {k: z[k] for k in z.files}


def strip_seconds(obj):
    if isinstance(obj, dict):
        return {k: strip_seconds(v) for k, v in obj.items() if k != "seconds"}
    if isinstance(obj, list):
        return [strip_seconds(v) for v in obj]
    return obj


def same_outputs(root_a, root_b):
    da, aa = load(root_a)
    db, ab = load(root_b)
    return (
        strip_seconds(da) == strip_seconds(db)
        and set(aa) == set(ab)
        and all(np.array_equal(aa[k], ab[k]) for k in aa)
    )


# 2: the first run
root1 = configure(workers=1)
st_a = loaders(s13.StageP13a())
st_a.run({}, [])
st_a.verify({})
st_a.report({})
st1, lines1 = run_b()
doc, arr = load(root1)
a_doc = json.loads((root1 / "model" / "p13a_model.json").read_text())
z = np.load(root1 / "model" / "p13a_arrays.npz")
a_arr = {k: z[k] for k in z.files}
rep = (root1 / "model" / "reports" / "p13b_report.md").read_text()
gaps = st1._verified
check(
    gaps is not None
    and gaps["kkt"] <= s13.KKT_TOL
    and gaps["monotone"] <= s13b.MONOTONE_TOL,
    f"run() and verify() pass on the synthetic world (gaps: "
    f"{ {k: f'{v:.1e}' for k, v in gaps.items()} })",
)
check(
    all(
        h in rep
        for h in (
            "## 1. The base path, extended",
            "## 2. The factorial",
            "### 2a.",
            "## 3. The loss and sampler arms",
            "## 4. The final path and the chosen model",
            "### 4a.",
            "selection pass 'ext'",
            "selection pass 'cells_arms'",
        )
    )
    and ";" not in rep,
    "report() writes every section and the pass lines, without a " "semicolon",
)
pass_lines = [
    x
    for x in lines1
    if "selection pass" in x
    or "probe" in x
    or "item side" in x
    or "catalog scorer" in x
]
check(
    pass_lines
    and not any(x.startswith("p13a") for x in lines1)
    and any(x.startswith("p13b: probe mean") for x in lines1),
    f"while P13b runs, the pass log lines name P13b ({len(pass_lines)} "
    f"lines, none with the prefix p13a)",
)
n12 = len(s13.LAMBDAS)
lams = [float(x) for x in s13.LAMBDAS] + [float(x) for x in s13b.EXT_LAMBDAS]
check(
    doc["extension"]["lambdas"] == lams
    and arr["ranks_ext"].shape == (len(s13b.EXT_LAMBDAS), len(W.SEL), 2)
    and arr["ranks_cells_arms"].shape == (26, len(W.SEL), 2),
    "the arrays hold the ranks of the extension (2 weights here) and of "
    "the 16 cells and 10 arms",
)
print(
    f"      lambda_base {doc['extension']['lambda_base']:g}, effects "
    + ", ".join(f"{f} {doc['effects'][f]['mean']:+.5f}" for f in s13b.FACTORS)
    + f", chosen cell {doc['chosen_cell']}, final lambda "
    f"{doc['final']['lambda']:g}"
)

# 3: every rank, dense and here
ZERO, MISS = gam.ZERO, gam.MISS


def term_values(t, tab, v, h, tier, m):
    """One term's value on rows with source values v and flags h: linear
    between the knots, flat beyond, the state constants, the tiers and the
    multiplier. Written here from the Term fields alone."""
    kn = np.asarray(t["knots"], dtype=np.float64)
    K = len(kn)
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t["zero_state"]:
        val = np.where(h & (v == 0.0), tab[K + ZERO], val)
    miss = tab[K + MISS] if t["missing_state"] else 0.0
    val = np.where(h, val, miss)
    if t["tiers"] is not None:
        val = np.where(np.isin(tier, t["tiers"]), val, 0.0)
    if t["multiplier"] is not None:
        val = m * val
    return val


def scores_of(j_terms, tables, cols, tier, m):
    out = np.zeros(len(tier))
    off = 0
    for t in j_terms:
        size = len(t["knots"]) + 4
        v, h = cols[t["source"]]
        out += term_values(t, np.asarray(tables[off : off + size]), v, h, tier, m)
        off += size
    return out


def rank_bounds(s, pick, exclude, tol=1e-9):
    s = np.array(s, dtype=np.float64)
    s[0] = -np.inf
    s[exclude] = -np.inf
    sp_ = s[pick]
    s[pick] = -np.inf
    exact = 1 + int(np.count_nonzero(s > sp_)) + int(np.count_nonzero(s[:pick] == sp_))
    near = int(np.count_nonzero(np.abs(s - sp_) <= tol))
    if near == int(np.count_nonzero(s == sp_)):
        return exact, exact
    g = int(np.count_nonzero(s > sp_ + tol))
    return 1 + g, 1 + g + near


mc = a_doc["mainstreaminess"]


def mainstream(items, weights):
    raw = float(np.sum(weights * W.popularity[items]) / np.sum(weights))
    return float(np.clip((raw - mc["mean"]) / mc["sd"], mc["lo"], mc["hi"]))


configs = {"ext": doc["extension"]["fits"], "cells_arms": doc["cells"] + doc["arms"]}
if doc["final"]["path"] is not None:
    configs["final"] = doc["final"]["path"]
key_of = {"ext": "ranks_ext", "cells_arms": "ranks_cells_arms", "final": "ranks_final"}
all_ok, n_exact, n_beyond, n_near = True, 0, 0, 0
for j, u in enumerate(W.SEL):
    items_h, w_h = W.cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    full = cn.full_catalog(cn.build_user_state(items_h, w_h, W.SPACE), W.SPACE)
    m_u = mainstream(items_h, w_h.astype(np.float64))
    for label, fits in configs.items():
        stored = arr[key_of[label]]
        for c, fj in enumerate(fits):
            s = scores_of(fj["terms"], fj["tables"], full, W.SPACE.tier, m_u)
            for p_ in range(2):
                lo, hi = rank_bounds(s, int(W.SEL_PICKS[j, p_]), items_h)
                got = int(stored[c, j, p_])
                all_ok &= min(lo, CUT1) <= got <= min(hi, CUT1)
                n_near += int(hi > lo)
                n_exact += int(lo == hi and lo <= s13.RANK_CUT)
                n_beyond += int(lo > s13.RANK_CUT)
check(
    all_ok and n_exact > 0 and n_beyond > 0,
    f"every stored rank of the {sum(len(v) for v in configs.values())} "
    f"configurations and {len(W.SEL)} selection listeners equals the P5 "
    f"count on a dense scorer written here, censored at {CUT1} "
    f"({n_exact} exact ranks, {n_beyond} beyond the cut, {n_near} near "
    f"ties)",
)

# 4: independent recomputations
tr_idx = np.flatnonzero(W.P["split"] == fp.SPLIT_TRAIN)
lens_tr = np.diff(W.P["indptr"])[tr_idx]
rows_i = np.concatenate(
    [np.arange(W.P["indptr"][j], W.P["indptr"][j + 1]) for j in tr_idx]
)
kind_i = W.P["kind"][rows_i]
item_i = W.P["item"][rows_i].astype(np.int64)
tier_i = W.F["match_tier"][rows_i].astype(np.int64)
user_i = np.repeat(np.arange(len(tr_idx)), lens_tr)
npos_tr = np.array(
    [
        int(
            np.sum(
                W.P["kind"][W.P["indptr"][j] : W.P["indptr"][j + 1]] == fp.KIND_POSITIVE
            )
        )
        for j in tr_idx
    ]
)
U_tr = len(tr_idx)
cols_i = {
    c: (W.F["values"][rows_i, k].astype(np.float64), W.F["has"][rows_i, k].astype(bool))
    for k, c in enumerate(cn.CONTINUOUS)
}
m_tr = np.array(
    [
        mainstream(
            *[
                np.asarray(x)
                for x in W.cd.history_of(
                    W.u_indptr, W.u_indices, W.u_data, int(W.P["user_id"][j])
                )
            ]
        )
        for j in tr_idx
    ]
)
m_i = m_tr[user_i]
q_u_i = 1.0 / W.P["n_allowed"][tr_idx][user_i]
q_p_i = PP_SLOT[item_i] / W.P["pop_allowed"][tr_idx][user_i]
starts_tr = np.concatenate(([0], np.cumsum(lens_tr)[:-1]))
M16 = 16


def arm_loss(s, arm):
    """The loss of an arm from the row scores, one listener at a time."""
    total = []
    for u in range(U_tr):
        r0, npu = starts_tr[u], npos_tr[u]
        negs = (
            r0
            + npu
            + {
                "uniform": np.arange(16),
                "p_pop": np.arange(16, 32),
                "mixture": np.r_[0:8, 16:24],
                "mix32": np.arange(32),
                "base": np.arange(32),
            }[arm[0]]
        )
        m = len(negs)
        q = {
            "uniform": q_u_i,
            "p_pop": q_p_i,
            "mixture": 0.5 * q_u_i + 0.5 * q_p_i,
            "base": 0.5 * q_u_i + 0.5 * q_p_i,
        }.get(arm[0])
        for p in range(r0, r0 + npu):
            w = 1.0 / (npu * U_tr)
            if arm[1] == "bpr":
                total.append(w / m * float(np.sum(np.logaddexp(0.0, s[negs] - s[p]))))
                continue
            zn = s[negs].copy()
            zp = s[p]
            if arm[1] in ("A", "B"):
                zn = zn - np.log(m * q[negs])
            if arm[1] == "B":
                if q[p] <= 0.0:
                    continue  # weight 0
                zp = zp - math.log(m * q[p])
            total.append(
                w * (float(np.logaddexp.reduce(np.concatenate(([zp], zn)))) - zp)
            )
    return math.fsum(total)


arm_of = {a: a for a in s13b.ARMS}
worst_loss, n_loss = 0.0, 0
for fj in (
    doc["extension"]["fits"]
    + [c for c in doc["cells"] if not c["reuses_base_fit"]]
    + doc["arms"]
    + (doc["final"]["path"] or [])
):
    arm = ("base", "A") if fj["arm"] == "base" else tuple(fj["arm"].split("-"))
    s = scores_of(fj["terms"], fj["tables_raw"], cols_i, tier_i, m_i)
    lo_ = arm_loss(s, arm)
    worst_loss = max(worst_loss, abs(lo_ - fj["fit"]["loss"]) / max(1.0, abs(lo_)))
    n_loss += 1
check(
    worst_loss < 1e-12,
    f"the stored loss of each of the {n_loss} fits equals the loss "
    f"recomputed here from the raw pairs, with each arm's negatives, "
    f"offsets and weights (largest relative gap {worst_loss:.1e})",
)


def knots_direct(v, pos_mask):
    parts = [
        np.array([0.0]),
        np.quantile(v[pos_mask], s13.KNOT_POS_LEVELS),
        np.quantile(v[~pos_mask], s13.KNOT_NEG_LEVELS),
    ]
    k = np.unique(np.concatenate(parts))
    gap = 1e-6 * (k[-1] - k[0])
    out = [k[0]]
    for x_ in k[1:]:
        if x_ - out[-1] > gap:
            out.append(x_)
    return [float(x_) for x_ in out]


v_w, h_w = cols_i["work_affinity"]
on_w = h_w & (v_w != 0.0)
check(
    doc["work_knots"] == knots_direct(v_w[on_w], (kind_i == fp.KIND_POSITIVE)[on_w]),
    "the work_affinity knots follow P13a's knot rule on the training "
    "rows with a match, recomputed here",
)
ref_i = kind_i == fp.KIND_POP
worst_c, n_c = 0.0, 0
for fj in (
    doc["extension"]["fits"] + doc["cells"] + doc["arms"] + (doc["final"]["path"] or [])
):
    off = 0
    for t in fj["terms"]:
        size = len(t["knots"]) + 4
        if not t["centered"]:
            v_, h_ = cols_i[t["source"]]
            val = term_values(
                dict(t, multiplier=None),
                np.asarray(fj["tables"][off : off + size]),
                v_[ref_i],
                h_[ref_i],
                tier_i[ref_i],
                1.0,
            )
            worst_c = max(worst_c, abs(float(np.mean(val))))
            n_c += 1
        off += size
check(
    worst_c < 1e-12,
    f"every stored table is centered: each of the {n_c} term tables has "
    f"mean 0 over the p_pop negatives, recomputed here (largest "
    f"{worst_c:.1e})",
)
ctx_b = st1._context()
bounds_ok = True
for cell in s13b.CELLS:
    lay_c, _ = st1._cell_layout(ctx_b, cell)
    for lab, bnd in zip(lay_c.labels, lay_c.bounds):
        name, part = lab.rsplit(":", 1)
        t = next(t_ for t_ in lay_c.terms if t_.name == name)
        if not cell[0]:
            bounds_ok &= bnd == (None, None)
        elif part == "zero" and t.monotone:
            bounds_ok &= bnd == (None, 0.0)
        elif part.startswith("inc"):
            bounds_ok &= t.monotone == 1 and bnd == (0.0, None)
        else:
            bounds_ok &= bnd == (None, None)
check(
    bounds_ok,
    "a monotone cell bounds every increment of its affinity terms below "
    "at 0 and every free zero state above at the first knot height, a "
    "free cell bounds nothing",
)
work_signal = not a_doc["audit"]["missing"]["concepts"]["work_affinity"]["no_signal"]
lay_w, _ = st1._cell_layout(ctx_b, (0, 1, 1, 1))
check(
    ("work_affinity:miss" in lay_w.labels) == work_signal
    and "work_affinity" not in [t["name"] for t in a_doc["terms"]],
    f"work_affinity's missing constant is learned exactly when P13a's "
    f"audit found a signal in its data flag (signal: {work_signal})",
)


# The KKT conditions of a monotone fit with work_affinity and of the BPR
# fit, by finite differences of the loss written here.
ctx1 = st1._context()


def kkt_fd(fj, arm, lam):
    cell = tuple(int(x) for x in fj["cell"][1::2])
    lay, _ = st1._cell_layout(ctx1, cell)
    theta = np.asarray(fj["theta"], dtype=np.float64)

    def F(th):
        T = lay.tables(th)
        s_ = scores_of(fj["terms"], T, cols_i, tier_i, m_i)
        return (
            arm_loss(s_, arm)
            + lam * float(th @ lay.Q @ th)
            + s13.RIDGE * float(th @ th)
        )

    eps = 1e-6
    g = np.zeros(len(theta))
    for k in range(len(theta)):
        e = np.zeros(len(theta))
        e[k] = eps
        g[k] = (F(theta + e) - F(theta - e)) / (2 * eps)
    worst = 0.0
    for x_, g_, (lo, hi) in zip(theta, g, lay.bounds):
        if lo is not None and x_ <= lo:
            worst = max(worst, max(-g_, 0.0))
        elif hi is not None and x_ >= hi:
            worst = max(worst, max(g_, 0.0))
        else:
            worst = max(worst, abs(g_))
    return worst, int(
        sum(
            1
            for x_, (lo, hi) in zip(theta, lay.bounds)
            if (lo is not None and x_ <= lo) or (hi is not None and x_ >= hi)
        )
    )


cell_mw = next(
    c
    for c in doc["cells"]
    if c["factors"]["monotone"] and c["factors"]["work_affinity"]
)
k1, active1 = kkt_fd(cell_mw, ("base", "A"), doc["extension"]["lambda_base"])
bpr = doc["arms"][s13b.ARMS.index(("mix32", "bpr"))]
k2, _ = kkt_fd(bpr, ("mix32", "bpr"), doc["extension"]["lambda_base"])
check(
    k1 < 1e-6 and k2 < 1e-6 and active1 > 0,
    f"finite differences of the loss written here meet the KKT "
    f"conditions at the stored monotone fit {cell_mw['cell']} ({active1} "
    f"parameters at a bound, worst {k1:.1e}) and at the BPR fit (worst "
    f"{k2:.1e})",
)

# Exact Newton needs the exact Hessian: through the stage's objective, the
# Hessian of a monotone cell, the BPR arm and a form-B arm equals central
# finite differences of the gradient, on 25 random parameter columns.
rng_h = np.random.default_rng(77)
lam_b = doc["extension"]["lambda_base"]
worst_h = {}
with st1._design(ctx1) as pool_h:
    for label, fj, cell, arm in (
        ("monotone cell", cell_mw, (1, 1, 0, 0), None),
        ("BPR arm", bpr, s13b.BASE_CELL, ("mix32", "bpr")),
        (
            "p_pop form B arm",
            doc["arms"][s13b.ARMS.index(("p_pop", "B"))],
            s13b.BASE_CELL,
            ("p_pop", "B"),
        ),
    ):
        lay_h, idx_h = st1._cell_layout(ctx1, cell)
        obj_h = st1._objective(pool_h, ctx1, lay_h, idx_h, arm, lam_b)
        th_h = np.asarray(fj["theta"], dtype=np.float64)
        H_h = obj_h.hessian(th_h)
        eps_h = 1e-6
        worst = 0.0
        for k in rng_h.choice(lay_h.P, size=25, replace=False):
            e = np.zeros(lay_h.P)
            e[k] = eps_h
            col = (obj_h(th_h + e)[1] - obj_h(th_h - e)[1]) / (2 * eps_h)
            worst = max(worst, float(np.max(np.abs(col - H_h[:, k]))))
        worst_h[label] = worst / max(1.0, float(np.max(np.abs(H_h))))
check(
    max(worst_h.values()) < 1e-6,
    f"the Hessian that the stage's Newton steps use equals finite "
    f"differences of its gradient ("
    + ", ".join(f"{k} {v:.1e}" for k, v in worst_h.items())
    + ")",
)

mono_ok, n_mono = True, 0
for fj in doc["cells"] + (doc["final"]["path"] or []):
    for t, off in zip(
        fj["terms"], np.cumsum([0] + [len(t["knots"]) + 4 for t in fj["terms"]])[:-1]
    ):
        if not t["monotone"]:
            continue
        n_mono += 1
        K = len(t["knots"])
        hgt = np.asarray(fj["tables_raw"][off : off + K]) * t["monotone"]
        mono_ok &= bool(np.all(np.diff(hgt) >= -1e-12))
        if t["zero_state"]:
            mono_ok &= (
                fj["tables_raw"][off + K + ZERO] * t["monotone"] <= hgt[0] + 1e-12
            )
aff = set(s13b.AFFINITY)
mono_terms_ok = all(
    (
        {t["name"] for t in fj["terms"] if t["monotone"]}
        == {
            t["name"]
            for t in fj["terms"]
            if t["source"] in aff
            and t["multiplier"] is None
            and not t["centered"]
            and t["tiers"] is None
        }
        if fj["factors"]["monotone"]
        else not any(t["monotone"] for t in fj["terms"])
    )
    for fj in doc["cells"]
)
check(
    mono_ok and n_mono > 0 and mono_terms_ok,
    f"every monotone cell bounds exactly the affinity terms, and each of "
    f"its {n_mono} bounded shapes never falls and keeps its zero state "
    f"at or below the first knot",
)
free_breaks = [
    t["name"]
    for t, off in zip(
        doc["cells"][s13b.CELLS.index(s13b.BASE_CELL)]["terms"],
        np.cumsum(
            [0]
            + [
                len(t["knots"]) + 4
                for t in doc["cells"][s13b.CELLS.index(s13b.BASE_CELL)]["terms"]
            ]
        )[:-1],
    )
    if t["source"] in aff
    and t["multiplier"] is None
    and not t["centered"]
    and t["tiers"] is None
    and np.any(
        np.diff(
            np.asarray(
                doc["cells"][s13b.CELLS.index(s13b.BASE_CELL)]["tables_raw"][
                    off : off + len(t["knots"])
                ]
            )
        )
        < -1e-9
    )
]
check(
    len(free_breaks) > 0,
    f"the free base cell breaks the order for some affinity terms "
    f"({', '.join(free_breaks)}), so the bounds bind",
)

# lambda_base, the effects, the decisions and the final choice.


def ndcg(r):
    r = np.asarray(r, dtype=np.float64)
    return np.where(r <= 100, 1.0 / np.log2(r + 1.0), 0.0).mean(axis=-1)


def one_se(sc):
    means = sc.mean(axis=1)
    best = 0
    for c in range(len(sc)):
        if means[c] > means[best]:
            best = c
    for c in range(len(sc)):
        d = sc[best] - sc[c]
        if c == best or d.mean() <= d.std(ddof=1) / math.sqrt(sc.shape[1]):
            return c, best


sc19 = ndcg(np.concatenate([a_arr["ranks"], arr["ranks_ext"]]))
c19, b19 = one_se(sc19)
check(
    c19 == doc["extension"]["chosen"]
    and b19 == doc["extension"]["best"]
    and doc["extension"]["lambda_base"] == lams[c19],
    f"lambda_base follows from P13a's and the extension's ranks by the "
    f"one-standard-error rule written here (weight {c19})",
)
base_theta = (
    a_doc["path"][c19]["theta"]
    if c19 < n12
    else doc["extension"]["fits"][c19 - n12]["theta"]
)
check(
    doc["cells"][s13b.CELLS.index(s13b.BASE_CELL)]["theta"] == base_theta
    and all(j["lambda"] == lams[c19] for j in doc["cells"] + doc["arms"]),
    "the base cell holds the base fit at lambda_base, and every cell and "
    "arm is fit at lambda_base",
)
scb = ndcg(arr["ranks_cells_arms"])
kb = s13b.CELLS.index(s13b.BASE_CELL)
ok_e = True
eff_here = {}
for i, f in enumerate(s13b.FACTORS):
    on = [k for k, c in enumerate(s13b.CELLS) if c[i]]
    off = [k for k, c in enumerate(s13b.CELLS) if not c[i]]
    d = scb[on].mean(axis=0) - scb[off].mean(axis=0)
    eff_here[f] = d.mean()
    ok_e &= (
        abs(doc["effects"][f]["mean"] - d.mean()) < 1e-14
        and abs(doc["effects"][f]["se"] - d.std(ddof=1) / math.sqrt(len(d))) < 1e-14
    )
thr_ = 0.02 * scb[kb].mean()
want_cell = (
    int(eff_here["monotone"] > -thr_),
    int(eff_here["work_affinity"] > thr_),
    int(eff_here["tier_curve"] > thr_),
    int(eff_here["tilt"] > thr_),
)
check(
    ok_e
    and doc["chosen_cell"] == s13b.cell_label(want_cell)
    and abs(doc["decisions"]["sesoi_abs"] - thr_) < 1e-16,
    f"the effects and the decisions follow from the stored ranks by the "
    f"declared rules written here (cell {s13b.cell_label(want_cell)})",
)
sel_ok = True
for k, fj in enumerate(doc["cells"] + doc["arms"]):
    d = scb[kb] - scb[k]
    se = d.std(ddof=1) / math.sqrt(len(d))
    sel_ok &= (
        abs(fj["selection"]["mean"] - scb[k].mean()) < 1e-15
        and abs(fj["selection"]["diff_to_base"] - d.mean()) < 1e-15
        and abs(fj["selection"]["se"] - se) < 1e-15
    )
check(
    sel_ok,
    "each cell's and arm's score, difference to the base cell and "
    "standard error follow from the stored ranks",
)
rf = (
    arr["ranks_final"]
    if "ranks_final" in arr
    else np.concatenate([a_arr["ranks"], arr["ranks_ext"]])
)
cf, bf = one_se(ndcg(rf))
check(
    cf == doc["final"]["chosen"]
    and bf == doc["final"]["best"]
    and doc["final"]["lambda"] == lams[cf],
    "the final choice follows from the final ranks",
)
un = {j["arm"]: np.asarray(j["tables"]) for j in doc["arms"]}
check(
    np.max(np.abs(un["uniform-none"] - un["uniform-B"])) <= 1e-9
    and np.max(np.abs(un["uniform-none"] - un["uniform-A"])) > 1e-3,
    "the uniform arm with form B equals the uncorrected one, and form A "
    "differs from both",
)
n_q0 = int(np.count_nonzero(q_p_i[kind_i == fp.KIND_POSITIVE] == 0.0))
check(
    n_q0 > 0
    and doc["arms"][s13b.ARMS.index(("p_pop", "B"))]["positives_without_weight"]
    == n_q0,
    f"the p_pop arm with form B reports the positives outside P3's core, "
    f"which get weight 0 ({n_q0} in this world)",
)
sp_ = doc["selection_passes"]
check(
    set(sp_) == set(configs)
    and all(sp_[k]["listeners"] == len(W.SEL) for k in sp_)
    and abs(sp_["ext"]["full_pass_share"] - float(np.mean(arr["scored_ext"] == W.NI)))
    < 1e-15
    and abs(
        sp_["cells_arms"]["mean_scored_share"]
        - float(np.mean(arr["scored_cells_arms"]) / W.NI)
    )
    < 1e-15,
    "the pass records hold the share of whole passes and of scored slots",
)

# 5: three workers
root3 = configure(workers=3, sub="w3")
(root3 / "model").mkdir(parents=True)
for name in ("p13a_model.json", "p13a_arrays.npz"):
    shutil.copy(root1 / "model" / name, root3 / "model" / name)
run_b()
check(
    same_outputs(root1, root3),
    "one worker and three workers give the same model and arrays",
)

# 6: resume
root1 = configure(workers=1)
fit_calls = []
real_fit, real_fb = gam.fit, gam_arms.fit_bounded


def count_fit(*a, **k):
    fit_calls.append("fit")
    return real_fit(*a, **k)


def count_fb(*a, **k):
    fit_calls.append("bounded")
    return real_fb(*a, **k)


gam.fit, gam_arms.fit_bounded = count_fit, count_fb
try:
    shutil.copytree(root1 / "model", TMP / "first" / "model")
    _, lines_r = run_b()
    check(
        not fit_calls
        and not any("catalog scorer ready" in x for x in lines_r)
        and sum("selection pass resume" in x for x in lines_r) == len(configs)
        and same_outputs(root1, TMP / "first"),
        "a finished run fits nothing, scores no listener and writes the "
        "same outputs",
    )
    work = root1 / "scratch" / "p13b"
    st_json = json.loads((work / "fits.json").read_text())
    lost = s13b.fit_id((1, 1, 0, 1), None, doc["extension"]["lambda_base"])
    del st_json["fits"][lost]
    (work / "fits.json").write_text(json.dumps(st_json))
    parts = sorted((work / "pass_cells_arms" / "selection").glob("part_*.npz"))
    parts[3].unlink()
    fit_calls.clear()
    _, lines_r2 = run_b()
    check(
        fit_calls == ["bounded"]
        and sum("catalog scorer ready" in x for x in lines_r2) == 1
        and same_outputs(root1, TMP / "first"),
        "a lost fit and a lost selection block are computed again, and "
        "the outputs stay the same",
    )
finally:
    gam.fit, gam_arms.fit_bounded = real_fit, real_fb

# 7: planted faults
st_v = loaders(s13b.StageP13b())
mpath = root1 / "model" / "p13b_model.json"
apath = root1 / "model" / "p13b_arrays.npz"
good_doc = mpath.read_text()
good_arr = dict(np.load(apath))


def with_doc(change, text):
    d = json.loads(good_doc)
    change(d)
    mpath.write_text(json.dumps(d))
    try:
        return raises(lambda: st_v.verify({}), text)
    finally:
        mpath.write_text(good_doc)


def with_arr(change, text):
    a = {k: v.copy() for k, v in good_arr.items()}
    change(a)
    np.savez(apath, **a)
    try:
        return raises(lambda: st_v.verify({}), text)
    finally:
        np.savez(apath, **good_arr)


def cell_k(d, cell):
    return d["cells"][s13b.CELLS.index(cell)]


def bump_theta(d):
    j = cell_k(d, (1, 0, 0, 0))
    j["theta"][5] += 1e-3


def bump_tables(d):
    cell_k(d, (0, 1, 0, 0))["tables_raw"][3] += 1e-9


def bump_objective(d):
    d["arms"][0]["fit"]["objective"] += 1e-6


def bump_uniform_b(d):
    j = d["arms"][s13b.ARMS.index(("uniform", "B"))]
    j["tables"][2] += 1e-3


def other_chosen(d):
    d["chosen_cell"] = "m0w0t0p0" if d["chosen_cell"] != "m0w0t0p0" else "m1w1t1p1"


def other_base(d):
    d["extension"]["chosen"] = (d["extension"]["chosen"] + 1) % len(lams)


def other_base_theta(d):
    cell_k(d, s13b.BASE_CELL)["theta"][0] += 0.5


def other_effect(d):
    d["effects"]["tilt"]["mean"] += 1e-4


def other_params(d):
    d["params"]["sesoi"] = 0.03


def bpr_ranks(a):
    a["ranks_cells_arms"][16 + s13b.ARMS.index(("mix32", "bpr"))] = 1


def ext_ranks(a):
    a["ranks_ext"][-1] = np.where(a["ranks_ext"][-1] > 1, 1, CUT1)


check(
    with_doc(bump_theta, "projected gradient"),
    "verify() fails when a stored fit is moved off its optimum",
)
check(
    with_doc(bump_tables, "stored raw tables differ"),
    "verify() fails when stored tables differ from M theta",
)
check(
    with_doc(bump_objective, "objective differs"),
    "verify() fails when a stored objective differs from its " "recomputation",
)
check(
    with_doc(bump_uniform_b, "form B differs"),
    "verify() fails when the uniform arm with form B leaves the " "uncorrected arm",
)
check(
    with_doc(other_chosen, "chosen cell does not follow"),
    "verify() fails when the stored chosen cell does not follow from the " "ranks",
)
check(
    with_doc(other_base, "lambda_base does not follow"),
    "verify() fails when lambda_base does not follow from the ranks",
)
check(
    with_doc(other_base_theta, "base cell does not hold"),
    "verify() fails when the base cell holds another fit",
)
check(
    with_doc(other_effect, "stored effects differ"),
    "verify() fails when a stored effect differs from the ranks",
)
check(
    with_doc(other_params, "other parameters"),
    "verify() fails for outputs built with other parameters",
)
check(
    with_arr(bpr_ranks, "outside the reference range"),
    "verify() fails when the stored ranks of an arm leave the dense " "reference",
)
check(
    with_arr(ext_ranks, "does not follow")
    or with_arr(ext_ranks, "outside the reference range"),
    "verify() fails when the extension's stored ranks change",
)


def base_cell_ranks(a):
    r = a["ranks_cells_arms"][s13b.CELLS.index(s13b.BASE_CELL)]
    r[0] = np.where(r[0] == 1, 2, 1)


def final_ranks(a):
    r = a["ranks_final"][doc["extension"]["chosen"]]
    r[0] = np.where(r[0] == 1, 2, 1)


check(
    with_arr(base_cell_ranks, "base cell's ranks differ"),
    "verify() fails when the base cell's ranks leave the base path's "
    "ranks at lambda_base",
)
check(
    with_arr(final_ranks, "final path's ranks at lambda_base differ"),
    "verify() fails when the final path's ranks at lambda_base leave the "
    "chosen cell's factorial ranks",
)
check(
    raises(lambda: st_v.verify({}), "no such text") is False
    and st_v._verified is not None,
    "verify() passes again on the restored outputs",
)

# 8: the base-cell branch
real_decide = s13b.decide


def decide_base(effects, base_score, sesoi=s13b.SESOI):
    _, rec_ = real_decide(effects, base_score, sesoi)
    return s13b.BASE_CELL, rec_


s13b.decide = decide_base
try:
    st_base, _ = run_b()
    doc_b, arr_b = load(root1)
    rep_b = (root1 / "model" / "reports" / "p13b_report.md").read_text()
    check(
        doc_b["final"]["reuses_base_path"]
        and doc_b["final"]["path"] is None
        and "ranks_final" not in arr_b
        and doc_b["final"]["chosen"] == doc_b["extension"]["chosen"]
        and st_base._verified is not None
        and "The chosen cell is the base cell" in rep_b
        and "### 4a." in rep_b,
        "when the chosen cell is the base cell, the base path is the "
        "final path, no final pass runs, and verify() passes",
    )
finally:
    s13b.decide = real_decide

# Summary
shutil.rmtree(TMP, ignore_errors=True)
shutil.rmtree(tmp_store, ignore_errors=True)
print()
if failures:
    print(f"{len(failures)} check(s) FAILED:")
    for f in failures:
        print("  - " + f)
    raise SystemExit(1)
print("all checks passed")
