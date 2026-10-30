"""Smoke test of pipeline/beyond.py (the runner stage beyond_accuracy) on
the P13 chain of p13_chain.py. Every path points into a temporary folder,
and nothing touches the real model folder.

1. Units without the chain: the Gini index of known vectors, the Jaccard
   index, the cap walk (hand-made rows and a loop on random rows),
   first_new with ties (Scan against hand-made answers), Scan against a
   full sort written here on random scores with ties, non-candidates and
   small prune limits, the known tracks against a loop over the world's
   credits, and the npz writer (equal arrays give equal bytes).
2. The chain with its markers, p14_cold_gam and p14_eval through the
   runner on synthetic inputs, as test_p14.py makes them.
3. beyond_accuracy through the runner with 3 workers and blocks of 16
   listeners: every list, score, known flag, first_new and pick rank of
   every eval listener and arm against a full sort written here (a dense
   sum in term order for the GAM arms, ItemKNN's scores with a loop over
   P9b's lists as a second route, the factors' product for ALS, P5's
   scores for popularity), every metric against loops written here, P14's
   ranks, NDCG@10 and R@10, a second call that skips, a refused change of
   the results, a changed P14 rank that stops the run, the report and the
   tables.
4. One worker against three: a run stopped inside the pass and resumed
   with one worker writes outputs byte-identical to the run with three.

Usage, from the repository root:
    python3 test_beyond.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import csv
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

import p13_chain as C
import p13_world as W
from pipeline import als
from pipeline import beyond as B
from pipeline import common
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import config
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import itemknn as knn
from pipeline import itemknn_select as IK
from pipeline import p13c as PC
from pipeline import p13d as D
from pipeline import p14 as P
from pipeline import p14_cold as PK
from pipeline import runner as RN
from pipeline import stage_p13b2 as s13b2

failures = []
T0 = time.time()
PARAMS = dict(config.DEFAULT_PARAMS)
FAKED = ("p6_als", "p9b_colisten", "p13b2_loss", "p5_eval_baseline", "als_phase_d")


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def quiet(msg):
    pass


def clock():
    return f"{time.time() - T0:.0f} s"


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


CAP = Capture()
common.log.addHandler(CAP)
common.log.setLevel(logging.INFO)


def raises(fn, text):
    try:
        fn()
    except (ValueError, RuntimeError) as exc:
        return text in str(exc)
    return False


def sorted_full(s, picks, new, keep):
    """A full sort written here: (top ids, first_new, pick ranks)."""
    cand = [i for i in range(len(s)) if s[i] > -np.inf]
    order = sorted(cand, key=lambda i: (-float(s[i]), i))
    pos = {i: p + 1 for p, i in enumerate(order)}
    first = next((p + 1 for p, i in enumerate(order) if new[i]), -1)
    return order[:keep], first, [pos[int(t)] for t in picks]


# 1: units without the chain
check(
    B.gini([0, 0, 0, 10]) == 0.75
    and B.gini([5, 5, 5, 5]) == 0.0
    and B.gini([0, 1]) == 0.5
    and B.gini([0, 0]) is None,
    "1: the Gini index of hand-made vectors (0.75, 0, 0.5, none)",
)
rng = np.random.default_rng(15)
ok = True
for _ in range(20):
    v = rng.integers(0, 6, size=int(rng.integers(2, 40)))
    if v.sum() == 0:
        continue
    pair = np.abs(v[:, None] - v[None, :]).sum() / (2 * len(v) ** 2 * v.mean())
    ok &= abs(B.gini(v) - pair) < 1e-12
check(
    ok,
    "1: the Gini index equals the mean absolute difference over 2 "
    "times the mean, on 20 random vectors",
)
A_ = np.array([[1, 2, 3, -1], [1, 2, 3, 4], [5, 6, -1, -1]])
B_ = np.array([[2, 3, 4, 5], [4, 3, 2, 1], [7, 8, 9, -1]])
check(
    np.allclose(B.jaccard(A_, B_), [0.4, 1.0, 0.0], rtol=0, atol=1e-15),
    "1: the Jaccard index of hand-made sets (0.4, 1, 0)",
)

# the cap walk: artists A A B A C - B B D E F G H I (- has no credit)
ids_ = np.arange(10, 24).reshape(1, -1)
prim_ = np.array([[1, 1, 2, 1, 3, -1, 2, 2, 4, 5, 6, 7, 8, 9]])
want = {
    1: [0, 2, 4, 5, 8, 9, 10, 11, 12, 13],
    2: [0, 1, 2, 4, 5, 6, 8, 9, 10, 11],
    None: list(range(10)),
}
ok = all(B.cap_positions(ids_, prim_, c)[0].tolist() == w for c, w in want.items())
one = np.array([[1] * 12 + [-1, -1]])
ids1 = np.where(np.arange(14) < 12, np.arange(14), -1).reshape(1, -1)
p1 = B.cap_positions(ids1, one, 1)[0]
p2 = B.cap_positions(ids1, one, 2)[0]
pn = B.cap_positions(ids1, one, None)[0]
check(
    ok
    and p1.tolist() == [0] + [-1] * 9
    and p2.tolist() == [0, 1] + [-1] * 8
    and pn.tolist() == list(range(10)),
    "1: the cap walk keeps the first tracks of each primary artist up to "
    "the cap, never caps a track without a credit, and leaves a list "
    "unfilled when the artists run out",
)
ok = True
for _ in range(200):
    k_ = int(rng.integers(1, 30))
    pr = rng.integers(-1, 5, size=(1, k_))
    ids_r = np.where(rng.random((1, k_)) < 0.9, np.arange(k_), -1)
    ids_r = np.where(np.cumsum(ids_r < 0, axis=1) > 0, -1, ids_r)
    cap = [1, 2, 3, None][int(rng.integers(0, 4))]
    kept, cnt = [], Counter()
    for j in range(k_):
        if ids_r[0, j] < 0 or len(kept) == B.TOP:
            break
        a = int(pr[0, j])
        if a >= 0 and cap is not None and cnt[a] >= cap:
            continue
        kept.append(j)
        if a >= 0:
            cnt[a] += 1
    got = B.cap_positions(ids_r, pr, cap)[0]
    ok &= got[got >= 0].tolist() == kept
check(ok, "1: the cap walk equals a loop written here on 200 random rows")

cases = [
    (
        np.array([-np.inf, 5, 3, 5, 3, 1]),
        [0, 0, 1, 0, 1, 1],
        [3, 4],
        3,
        [1, 3, 2],
        3,
        [2, 4],
    ),
    (np.array([-np.inf, 3, 3, 2]), [0, 0, 1, 0], [1, 3], 2, [1, 2], 2, [1, 3]),
    (np.array([-np.inf, 4, 2, 4]), [0, 1, 0, 1], [2, 3], 3, [1, 3, 2], 1, [3, 2]),
    (np.array([-np.inf, 4, 4, 4]), [0, 1, 0, 1], [2, 3], 3, [1, 2, 3], 1, [2, 3]),
    (np.array([-np.inf, 2, 1, 1]), [0, 0, 0, 0], [2, 3], 2, [1, 2], -1, [2, 3]),
    (np.array([-np.inf, -np.inf, 1, 1]), [0, 1, 0, 0], [2, 3], 5, [2, 3], -1, [1, 2]),
]
ok = True
for s_, nw, pk, keep, top_w, fn_w, rk_w in cases:
    nw = np.asarray(nw, dtype=bool)
    for step in (1, 2, len(s_)):
        sc = B.Scan(keep, pk, s_[pk])
        for a in range(0, len(s_), step):
            sc.feed(s_[a : a + step], a, nw[a : a + step])
        r = sc.result()
        ok &= (
            r["top"].tolist() == top_w
            and r["first_new"] == fn_w
            and r["ranks"].tolist() == rk_w
        )
check(
    ok,
    "1: Scan on hand-made cases: first_new counts the candidates "
    "with a higher score and the equal scores with a smaller id, an "
    "earlier new artist wins a tie, -1 without a new candidate, and "
    "every range size gives the same answer",
)
ok, real_prune = True, B.PRUNE
try:
    for trial in range(400):
        B.PRUNE = [1 << 16, 3, 1][trial % 3]
        n = int(rng.integers(5, 300))
        s_ = np.round(rng.normal(size=n), 1)
        if trial % 2:
            s_ = s_.astype(np.float32)
        s_[rng.random(n) < 0.2] = -np.inf
        s_[0] = -np.inf
        nw = rng.random(n) < rng.random()
        keep = int(rng.integers(1, 30))
        cand = np.flatnonzero(s_ > -np.inf)
        if not len(cand):
            continue
        pk = rng.choice(cand, size=2)
        top_w, fn_w, rk_w = sorted_full(s_, pk, nw, keep)
        step = int(rng.integers(1, 60))
        sc = B.Scan(keep, pk, s_[pk])
        for a in range(0, n, step):
            sc.feed(s_[a : a + step], a, nw[a : a + step])
        r = sc.result()
        ok &= (
            r["top"].tolist() == top_w
            and r["first_new"] == fn_w
            and r["ranks"].tolist() == rk_w
            and np.array_equal(r["score"], s_[top_w].astype(np.float64))
        )
finally:
    B.PRUNE = real_prune
check(
    ok,
    "1: Scan equals a full sort written here in 400 random cases "
    "(float64 and float32, ties, non-candidates, prune limits 1, 3 "
    "and 65536)",
)

credit = W.SPACE.credit_mats["artists"]
credit_t = credit.T.tocsr()
ip_a, cols_a = W.SRC.credits["artists"][0], W.SRC.credits["artists"][1]
row_cols = [set(cols_a[ip_a[i] : ip_a[i + 1]].tolist()) for i in range(W.NI)]
ok = True
for u in range(1, 40):
    hist_ = W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]]
    arts = set().union(*[row_cols[int(h)] for h in hist_])
    want_k = [i for i in range(W.NI) if row_cols[i] & arts]
    ok &= B.known_slots(credit, credit_t, np.unique(hist_)).tolist() == want_k
prim_w = np.array([min(c) if c and i else -1 for i, c in enumerate(row_cols)])
check(
    ok and np.array_equal(B.primary_artists(credit), prim_w),
    "1: the known tracks of 39 listeners and the primary artists equal "
    "loops over the world's credit rows",
)
TMP = Path(tempfile.mkdtemp(prefix="test_beyond_"))
arr_ = {"b": np.arange(5, dtype=np.int32), "a": np.array([[1.5, -np.inf]])}
B.write_npz(TMP / "x1.npz", arr_)
time.sleep(1.1)
B.write_npz(TMP / "x2.npz", arr_)
z_ = np.load(TMP / "x2.npz")
check(
    (TMP / "x1.npz").read_bytes() == (TMP / "x2.npz").read_bytes()
    and all(np.array_equal(z_[k], v) for k, v in arr_.items()),
    "1: the npz writer gives equal bytes for equal arrays a second later, "
    "and np.load reads them back",
)

# 2: the chain and P14
print(W.summary())
ROOT = TMP / "chain"
CHAIN = C.chain_into(ROOT)
SCR = ROOT / "scratch"
common.MODEL_DIR = CHAIN
NBOOT = {"listener": 6, "twoway": 3}


def stage():
    return C.p13_loaders(s13b2.StageP13b2())


def fake_marker(name):
    st = RN.STAGE_BY_NAME[name]
    common.write_marker(
        name,
        RN.stage_params_digest(st, PARAMS),
        {"test": name},
        {},
        source_sha=RN.stage_source_sha(st),
    )


def stages(model, scratch):
    ik = IK.StageItemKNN(model, scratch)
    ik._load = lambda: (
        W.NI,
        C.U_ARRAYS,
        W.SEL.copy(),
        W.SEL_PICKS.copy(),
        C.POP,
        C.LISTS,
    )
    sl = G.StageGamKnnSlices(model, scratch)
    sl._load = lambda: (W.NI, C.U_ARRAYS, C.LISTS, C.pop_centiles())
    out = [
        ik,
        sl,
        GR.StageGamRefit(model, scratch),
        PC.StageP13c("gam_refit", model, scratch),
        D.StageP13d(model, scratch),
        PC.StageP13c("p13d", model, scratch),
    ]
    for s in out[2:]:
        s._stage = stage
    out[3].n_boot, out[5].n_boot = NBOOT, {"listener": 4, "twoway": 2}
    out[3].n_verify = out[5].n_verify = 6
    return out


def execute(sts, **kw):
    by = dict(RN.STAGE_BY_NAME)
    by.update({s.name: s for s in sts})
    return RN.execute(sts, PARAMS, [], by_name=by, **kw)


for name in FAKED:
    fake_marker(name)
IK.run(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    W.SEL.copy(),
    W.SEL_PICKS.copy(),
    C.POP,
    C.LISTS,
    workers=1,
    say=quiet,
    n_check=12,
    inputs=IK.provenance(CHAIN),
)
G.run_slices(
    CHAIN,
    W.NI,
    C.U_ARRAYS,
    C.LISTS,
    C.pop_centiles(),
    workers=1,
    say=quiet,
    n_check=5,
    inputs=G.provenance(CHAIN),
)
GR.run(
    stage(),
    CHAIN,
    SCR / "gam_refit",
    say=quiet,
    n_verify=6,
    inputs=GR.provenance(CHAIN),
)
PC.run(
    stage(),
    CHAIN,
    SCR / "p13c",
    say=quiet,
    n_verify=6,
    n_boot=NBOOT,
    inputs=PC.provenance(CHAIN),
)
STS = stages(CHAIN, SCR)
real_decide = D.decide


def forced(scores, names_, model_arm):
    res = real_decide(scores, names_, model_arm)
    res["s2"] = dict(res["s2"], chosen=D.PILOT)
    return res


D.decide = forced
try:
    D.run(
        stage(), CHAIN, SCR / "p13d", say=quiet, n_verify=6, inputs=D.provenance(CHAIN)
    )
    rc_a = execute(STS, adopt=True)
finally:
    D.decide = real_decide
COLD = PK.StageP14Cold(CHAIN, SCR)
COLD._stage = stage

users_all = np.arange(1, len(W.u_indptr) - 1)
rest = np.setdiff1d(users_all, W.SEL)
EU = np.sort(np.random.default_rng(1404).choice(rest, size=60, replace=False))
NE = len(EU)
r4 = np.random.default_rng(1405)
y2011, y2014 = 1293840000, 1388534400
a_last = 120 * r4.integers((y2011 + 3600 * 24 * 30) // 120, y2014 // 120 - 10, NE) + 75
a_first = a_last - 120 * r4.integers(0, 40 * 24 * 30, NE)
FRAME = {
    "user": EU,
    "decile": EU % 10,
    "picks": np.array([[W.held[u][0], W.held[u][2]] for u in EU], dtype=np.int64),
    "ts_first": np.stack([a_last - 120 * r4.integers(10, 99999, NE), a_first], axis=1),
    "ts_last": np.stack([a_last, a_first + (a_last - a_first) // 2], axis=1),
    "dropped": 0,
}
tr_u, tr_f, tr_l, tr_w = [], [], [], []
pos_e = {int(u): i for i, u in enumerate(EU)}
for u in users_all:
    s, e = int(W.u_indptr[u]), int(W.u_indptr[u + 1])
    n = e - s
    lis = np.maximum(1, np.round(np.expm1(W.u_data[s:e] / 1000.0))).astype(np.int64)
    if int(u) in pos_e:
        i = pos_e[int(u)]
        lo_t = a_first[i] - 120 * 3000 * 24 * 30
        f_ = r4.integers(lo_t, a_first[i] - 120, n)
        l_ = np.minimum(f_ + r4.integers(0, 120 * 24 * 900, n), a_last[i] - 120)
    else:
        f_ = r4.integers(1104537600, y2014 - 200 * 24 * 3600, n)
        l_ = f_ + r4.integers(0, 120 * 24 * 900, n)
    f_ = (f_ // 120) * 120 + 30
    l_ = np.maximum((l_ // 120) * 120 + 30, f_)
    tr_u.append(np.full(n, u))
    tr_f.append(f_)
    tr_l.append(l_)
    tr_w.append(lis)
TR = [np.concatenate(x) for x in (tr_u, tr_f, tr_l, tr_w)]
TRAIN_PAIRS = np.bincount(W.u_indices, minlength=W.NI).astype(np.int64)
TRAIN = {int(u): W.u_indices[W.u_indptr[u] : W.u_indptr[u + 1]] for u in EU}
r5 = np.random.default_rng(1406)
XD = r5.normal(size=(len(W.u_indptr) - 1, 8)).astype(np.float32)
YD = r5.normal(size=(W.NI, 8)).astype(np.float32)
X6 = r5.normal(size=(len(W.u_indptr) - 1, 4)).astype(np.float32)
Y6 = r5.normal(size=(W.NI, 4)).astype(np.float32)
rows_ = P.TrainRows(W.u_indptr, W.u_indices)
g_pop = ev.global_ranks(C.POP)
STORED = {
    "p5": (
        EU.copy(),
        np.stack(
            [ev.user_ranks(g_pop, FRAME["picks"][:, p], EU, TRAIN) for p in range(2)],
            axis=1,
        ),
    ),
    "p6": (EU.copy(), als.rank_items(X6, Y6, rows_, EU, FRAME["picks"])),
}
ALS_D_SCORE = float(
    np.where(
        (rk := als.rank_items(XD, YD, rows_, W.SEL, W.SEL_PICKS)) <= 100,
        1 / np.log2(rk + 1.0),
        0,
    )
    .mean(axis=1)
    .mean()
)


class FakeInputs:
    def __init__(self):
        self.d = CHAIN

    def key(self):
        return {"fake": "test_beyond"}

    def frame(self):
        return dict(FRAME), (
            FRAME["user"],
            FRAME["decile"],
            FRAME["picks"][:, 0],
            FRAME["picks"][:, 1],
            0,
        )

    def covariates(self, n_slots):
        return C.pop_centiles(), TRAIN_PAIRS.copy(), W.tier.copy()

    def popularity(self, n_slots, id_min):
        return C.POP.copy()

    def leakage(self, frame, say):
        return P.leakage_counts(*TR, frame["user"], P.anchor_of(frame))

    def lists(self, n_slots):
        return C.LISTS

    def als(self, which):
        return (XD, YD, {"d": "x"}) if which == "d" else (X6, Y6, {"p6": "y"})

    def als_reference(self, which):
        return ALS_D_SCORE if which == "d" else None

    def stored_ranks(self, which):
        return STORED[which]


class FakeBeyondInputs(FakeInputs):
    def item_listeners(self, n_slots):
        return TRAIN_PAIRS.copy()

    def train_listeners(self):
        return int(np.count_nonzero(np.diff(W.u_indptr)))


P.EVAL_CHUNK, P.EVAL_BLOCK, P.PROBE_USERS = 64, 7, 3
P.N_PERM = 500
EVS = P.StageP14(CHAIN, SCR)
EVS._stage = stage
EVS._io = lambda: FakeInputs()
EVS.workers = 1
rc = execute(STS + [COLD, EVS])
check(
    rc_a == 0
    and rc == 0
    and common.marker_path("p14_eval").exists()
    and common.marker_path("p14_cold_gam").exists(),
    f"2: the chain, p14_cold_gam and p14_eval run through the runner " f"({clock()})",
)
base = stage()._base_context()
G14 = P.load_models(CHAIN, base)
terms = G14["terms"]
tables = G14["tables"]
offs = P.offsets_of(terms)
Z14 = dict(np.load(CHAIN / P.OUT_FILES[1]))
DOC14 = json.loads((CHAIN / P.OUT_FILES[0]).read_text())


def term_value(t, tab, v, h):
    K = t.K
    kn = t.knot_array
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t.zero_state:
        val = np.where(h & (v == 0.0), tab[K + gam.ZERO], val)
    return np.where(h, val, tab[K + gam.MISS] if t.missing_state else 0.0)


def dense_score(full, T):
    """The sum of the term values in term order, from 0."""
    s = np.zeros(W.NI)
    for t, o in zip(terms, offs):
        s += term_value(t, T[o : o + t.size], full[t.source][0], full[t.source][1])
    return s


def knn_loop(items, weights, kinds):
    w_ = np.zeros(W.NI)
    w_[items] = weights
    s = np.zeros(W.NI)
    for kk in kinds:
        ip_, nb, cs = C.LISTS[kk]
        for i in range(W.NI):
            a, b = ip_[i], ip_[i + 1]
            s[i] += float(np.dot(cs[a:b], w_[nb[a:b]]))
    return s


# 3: the stage
B.CHUNK, B.BLOCK, B.PROBE_USERS = 64, 16, 3
BS = B.StageBeyond(CHAIN, SCR)
BS._stage = stage
BS._io = lambda: FakeBeyondInputs()
BS.workers = 3
CAP.lines.clear()
rc = execute(STS + [COLD, EVS, BS])
OUT = {k: Path(v) for k, v in BS.outputs().items()}
doc = json.loads(OUT["results"].read_text()) if OUT["results"].exists() else None
z = dict(np.load(OUT["arrays"])) if OUT["arrays"].exists() else {}
real_stage = RN.STAGE_BY_NAME.get("beyond_accuracy")
check(
    isinstance(real_stage, B.StageBeyond)
    and [s.name for s in RN.SCRIPT_STAGES][-3:-1]
    == ["cold_start_curve", "beyond_accuracy"]
    and real_stage.upstream == ["p14_eval", "p14_cold_gam"]
    and real_stage.guarded,
    "3: the runner holds beyond_accuracy after cold_start_curve, guarded, "
    "with p14_eval and p14_cold_gam upstream",
)
check(
    rc == 0
    and doc is not None
    and common.marker_path("beyond_accuracy").exists()
    and any("probe listener" in ln for ln in CAP.lines)
    and any("pass, 7 arms, 3 workers" in ln for ln in CAP.lines)
    and any("bit for bit on the first chunk" in ln for ln in CAP.lines),
    f"3: beyond_accuracy runs through the runner with 3 workers, its "
    f"checks pass and it writes its marker ({clock()})",
)
CAP.lines.clear()
check(
    execute(STS + [COLD, EVS, BS]) == 0
    and any("skip beyond_accuracy" in ln for ln in CAP.lines),
    "3: a second runner call skips beyond_accuracy",
)

# Every list against a full sort written here
picks = FRAME["picks"]
ok_l = {a: True for a in B.ARMS}
knn_gap = 0.0
cand_k, known_k = {}, {}
for k, u in enumerate(EU):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    cand = np.ones(W.NI, dtype=bool)
    cand[0] = False
    cand[items] = False
    arts = set().union(*[row_cols[int(h)] for h in items])
    known = np.array([bool(row_cols[i] & arts) and i > 0 for i in range(W.NI)])
    new = np.array([bool(row_cols[i]) and i > 0 for i in range(W.NI)]) & ~known
    cand_k[k], known_k[k] = cand, known
    full = cn.full_catalog(cn.build_user_state(items, weights, W.SPACE), W.SPACE)
    scores = {}
    for c_, arm in enumerate(B.GAM_ARMS):
        scores[arm] = np.where(cand, dense_score(full, tables[c_]), -np.inf)
    hi_, hw = knn.history(W.u_indptr, W.u_indices, W.u_data, int(u))
    fast = knn.user_scores({kk: W.SPACE.colisten_t[kk] for kk in knn.KINDS}, hi_, hw)
    for kind, arm, kinds in (
        ("same_artist", "knn_same", ["same_artist"]),
        ("both", "knn_both", ["same_artist", "other_artists"]),
    ):
        loop = knn_loop(items, weights, kinds)
        knn_gap = max(knn_gap, float(np.max(np.abs(fast[kind] - loop))))
        scores[arm] = np.where(cand, fast[kind], -np.inf)
    c0 = (k // B.ALS_CHUNK) * B.ALS_CHUNK
    S_ = XD[EU[c0 : c0 + B.ALS_CHUNK]] @ YD.T
    scores["als_d"] = np.where(cand, S_[k - c0], -np.inf)
    scores["popularity"] = np.where(cand, C.POP.astype(np.float64), -np.inf)
    for arm in B.ARMS:
        top_w, fn_w, rk_w = sorted_full(scores[arm], picks[k], new, B.K)
        t = len(top_w)
        got = z[f"top_{arm}"][k].astype(np.int64)
        ok_l[arm] &= (
            got[:t].tolist() == top_w
            and np.all(got[t:] == -1)
            and np.array_equal(
                z[f"score_{arm}"][k][:t], scores[arm][top_w].astype(np.float64)
            )
            and z[f"known_{arm}"][k][:t].tolist() == known[top_w].tolist()
            and int(z[f"first_new_{arm}"][k]) == fn_w
            and z[f"rank_{arm}"][k].tolist() == rk_w
            and z[f"rank_{arm}"][k].tolist() == Z14[f"rank_{arm}"][k].tolist()
        )
check(
    all(ok_l.values()) and knn_gap < 1e-12,
    f"3: for all {NE} eval listeners and {len(B.ARMS)} arms the top "
    f"{B.K}, its scores and known flags, first_new and the pick ranks "
    f"equal a full sort written here, and the pick ranks equal P14's "
    f"(ItemKNN's scores against a loop over P9b's lists: largest gap "
    f"{knn_gap:.1e}). Failing arms: "
    f"{[a for a, v in ok_l.items() if not v] or 'none'}",
)
gk = Z14["top_has"][:, :10, [t.name for t in terms].index("artist_affinity")] & (
    np.nan_to_num(
        Z14["top_val"][:, :10, [t.name for t in terms].index("artist_affinity")]
    )
    > 0
)
check(
    np.array_equal(z["known_gam"][:, :10], gk)
    and doc["check"]["gam_known"]["known_share"] == float(gk.mean())
    and np.array_equal(z["top_gam"][:, :20], Z14["top_id"][:, 0, :20]),
    f"3: the GAM's top-10 known flags equal P14's artist affinity above 0 "
    f"(share {gk.mean():.3f}, P14 shows the artist reason on "
    f"{Z14['candidate'][0].mean():.3f}), and its top 20 equal P14's",
)

# The metrics by loops written here
prim_of = B.primary_artists(credit)
N_ = int(np.count_nonzero(np.diff(W.u_indptr)))
si_of = [-math.log2((int(TRAIN_PAIRS[i]) + 1) / (N_ + 1)) for i in range(W.NI)]
pc_of = C.pop_centiles()
pi_, pj_ = B.pairs_of(NE)


def mean_lo_hi(x):
    x = np.asarray(x, dtype=float)
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / math.sqrt(len(x)))
    return m, m - B.Z95 * se, m + B.Z95 * se


def close(iv, x, tol=1e-12):
    m, lo, hi = mean_lo_hi(x)
    return (
        abs(iv["mean"] - m) <= tol
        and abs(iv["lo"] - lo) <= tol
        and abs(iv["hi"] - hi) <= tol
    )


def ndcg_r(lst, pk):
    nd = rc_ = 0.0
    for t in pk:
        if int(t) in lst:
            p = lst.index(int(t)) + 1
            nd += 1.0 / math.log2(p + 1.0)
            rc_ += 1.0
    return nd / len(pk), rc_ / len(pk)


ok_m = {}
for arm in B.ARMS:
    r = doc["results"]["arms"][arm]
    tops = [z[f"top_{arm}"][k][:10].tolist() for k in range(NE)]
    kn = [z[f"known_{arm}"][k][:10].tolist() for k in range(NE)]
    known_sh = [sum(x) / 10 for x in kn]
    new_any = [
        float(any((prim_of[t] >= 0) and not f for t, f in zip(tt, ff)))
        for tt, ff in zip(tops, kn)
    ]
    dist = [len({int(prim_of[t]) for t in tt if prim_of[t] >= 0}) for tt in tops]
    multi = [sum(len(row_cols[t]) > 1 for t in tt) / 10 for tt in tops]
    si = [sum(si_of[t] for t in tt) / 10 for tt in tops]
    pcs = [sum(float(pc_of[t]) for t in tt) / 10 for tt in tops]
    union = set().union(*map(set, tops))
    cnt = Counter(t for tt in tops for t in tt)
    vec = np.array([cnt.get(i, 0) for i in range(1, W.NI)])
    g_pair = np.abs(vec[:, None] - vec[None, :]).sum() / (
        2 * len(vec) ** 2 * vec.mean()
    )
    jac = [
        len(set(tops[i]) & set(tops[j])) / len(set(tops[i]) | set(tops[j]))
        for i, j in zip(pi_, pj_)
    ]
    fn = [int(x) for x in z[f"first_new_{arm}"]]
    have = sorted(x for x in fn if x >= 1)
    qs = {
        f"p{q}": have[max(0, math.ceil(q / 100 * len(have)) - 1)] for q in B.QUANTILES
    }
    within = {c: [float(1 <= x <= c) for x in fn] for c in B.CUTS}
    caps_ok = True
    base_nd, base_r = None, None
    for cap in reversed(B.CAPS):
        nds, rcs, dis, kns, unf = [], [], [], [], []
        for k in range(NE):
            lst, ks, cnt_a = [], [], Counter()
            for j, t in enumerate(z[f"top_{arm}"][k].tolist()):
                if t < 0 or len(lst) == 10:
                    break
                a = int(prim_of[t])
                if a >= 0 and cap is not None and cnt_a[a] >= cap:
                    continue
                lst.append(t)
                ks.append(bool(z[f"known_{arm}"][k][j]))
                if a >= 0:
                    cnt_a[a] += 1
            nd, rr = ndcg_r(lst, picks[k])
            nds.append(nd)
            rcs.append(rr)
            dis.append(len({int(prim_of[t]) for t in lst if prim_of[t] >= 0}))
            kns.append(sum(ks) / max(len(lst), 1))
            unf.append(float(len(lst) < 10))
        if cap is None:
            base_nd, base_r = nds, rcs
        rr_ = r["caps"]["none" if cap is None else str(cap)]
        d_nd = np.array(nds) - np.array(base_nd)
        caps_ok &= (
            close(rr_["ndcg_at_10"], nds)
            and close(rr_["recall_at_10"], rcs)
            and abs(rr_["ndcg_minus_none"]["mean"] - d_nd.mean()) < 1e-12
            and close(rr_["distinct_artists"], dis)
            and close(rr_["known_share"], kns)
            and abs(rr_["unfilled"]["mean"] - np.mean(unf)) < 1e-12
        )
    ok_m[arm] = (
        close(r["known_share_top10"], known_sh)
        and close(r["new_artist_listeners"], new_any)
        and close(r["distinct_artists_top10"], dist)
        and close(r["multi_artist_share_top10"], multi)
        and close(r["novelty"]["self_information"], si)
        and close(r["novelty"]["pop_centile"], pcs)
        and r["coverage"]["tracks"] == len(union)
        and abs(r["coverage"]["share"] - len(union) / (W.NI - 1)) < 1e-15
        and r["coverage"]["artists"]
        == len({int(prim_of[t]) for t in union if prim_of[t] >= 0})
        and abs(r["gini"] - g_pair) < 1e-12
        and abs(r["overlap"]["jaccard"] - np.mean(jac)) < 1e-12
        and r["first_new"]["quantiles"] == qs
        and all(close(r["first_new"]["within"][str(c)], within[c]) for c in B.CUTS)
        and caps_ok
        and r["uncapped"]["ndcg_at_10"] == DOC14["arms"][arm]["ndcg_at_10"]
        and r["uncapped"]["recall_at_10"] == DOC14["arms"][arm]["recall_at_10"]
    )
ref_si = [float(np.mean([si_of[t] for t in TRAIN[int(u)]])) for u in EU]
ref_pc = [float(np.mean([pc_of[t] for t in TRAIN[int(u)]])) for u in EU]
check(
    all(ok_m.values())
    and close(doc["results"]["reference"]["self_information"], ref_si)
    and close(doc["results"]["reference"]["pop_centile"], ref_pc),
    f"3: every metric of every arm (known share, new artists, distinct "
    f"artists, tracks with two or more artists, coverage, Gini by pairs, "
    f"novelty and its reference row, "
    f"Jaccard on the stored pairs, first_new percentiles and shares, the "
    f"caps by a walk written here) equals a loop written here, and the "
    f"uncapped NDCG@10 and R@10 equal P14's exactly. Failing arms: "
    f"{[a for a, v in ok_m.items() if not v] or 'none'}",
)
caps_seen = {
    a: doc["results"]["arms"][a]["caps"]["1"]["ndcg_minus_none"]["mean"] for a in B.ARMS
}
check(
    any(v != 0 for v in caps_seen.values()),
    f"3: a cap of 1 changes NDCG@10 of some arms in the small world "
    f"({ {a: round(v, 5) for a, v in caps_seen.items()} })",
)

# Lists cut to 12 entries: the caps cannot fill every list
A12 = {
    k: v
    for k, v in z.items()
    if k.startswith(("top_", "known_", "first_new_", "n_known_cand", "n_new_cand"))
}
for arm in B.ARMS:
    t_ = A12[f"top_{arm}"].copy()
    t_[:, 12:] = -1
    A12[f"top_{arm}"] = t_
glob12 = {
    "picks": picks,
    "primary": z["track_primary"],
    "credits": z["track_credits"],
    "n_i": z["track_listeners"],
    "pop_centile": z["track_pop_centile"],
    "N": int(z["train_listeners"]),
    "ref_selfinfo": z["ref_selfinfo"],
    "ref_pop_centile": z["ref_pop_centile"],
}
res12, _ = B.summarize(A12, glob12)
ok12, unf_seen = True, 0.0
for arm in B.ARMS:
    for cap in (1, 2):
        kept_n = []
        for k in range(NE):
            cnt_a, n_k = Counter(), 0
            for t in A12[f"top_{arm}"][k].tolist():
                if t < 0 or n_k == 10:
                    break
                a = int(prim_of[t])
                if a >= 0 and cnt_a[a] >= cap:
                    continue
                n_k += 1
                if a >= 0:
                    cnt_a[a] += 1
            kept_n.append(n_k)
        u_ = res12["arms"][arm]["caps"][str(cap)]["unfilled"]["mean"]
        unf_seen = max(unf_seen, u_)
        ok12 &= abs(u_ - np.mean([x < 10 for x in kept_n])) < 1e-12
check(
    ok12 and unf_seen > 0,
    f"3: with lists cut to 12 entries the share of unfilled capped lists "
    f"equals a walk written here (largest {unf_seen:.3f})",
)
flip = {k: v.copy() for k, v in z.items() if k.startswith("known_")}
flip["known_gam"][3, 2] = ~flip["known_gam"][3, 2]
check(
    raises(lambda: B.check_gam_known(flip, Z14, terms), "P14's stored")
    and B.check_gam_known({"known_gam": z["known_gam"]}, Z14, terms)["known_share"]
    == float(gk.mean()),
    "3: check_gam_known refuses a flipped known flag of the GAM's top 10",
)

# The read-back and a changed P14 rank
saved = OUT["results"].read_bytes()
bad = json.loads(saved)
bad["results"]["arms"]["gam"]["known_share_top10"]["mean"] += 1e-6
OUT["results"].write_text(json.dumps(bad))
refused = raises(BS.check, "results")
OUT["results"].write_bytes(saved)
BS.check()
real_load = B.load_p14


def bad_load(d):
    doc_, zz = real_load(d)
    zz = dict(zz)
    r_ = zz["rank_als_d"].copy()
    r_[5, 1] += 1
    zz["rank_als_d"] = r_
    return doc_, zz


B.load_p14 = bad_load
try:
    stopped = raises(
        lambda: B.run(
            stage(),
            FakeBeyondInputs(),
            CHAIN,
            SCR / "beyond_bad",
            say=quiet,
            n_verify=2,
            inputs=BS.provenance(),
            workers=1,
        ),
        "differ from P14's",
    )
finally:
    B.load_p14 = real_load
check(
    refused and stopped,
    "3: the read-back refuses a changed result, and a changed stored P14 "
    "rank stops the run",
)

# The report and the tables
rep = OUT["report"].read_text()
cols_ok, block = True, []
for ln in rep.splitlines() + [""]:
    if ln.startswith("|"):
        block.append(ln.count("|"))
    elif block:
        cols_ok &= len(set(block)) == 1
        block = []
sections = (
    "## The top 10 per arm",
    "## First new artist",
    "## Caps per primary artist",
)
tabs = {}
for name in B.TAB_FILES:
    with open(CHAIN / B.FIG_DIR / name, newline="") as fh:
        tabs[name] = list(csv.reader(fh))
tab_ok = (
    len(tabs[B.TAB_FILES[0]]) == 1 + len(B.ARMS) + 1
    and len(tabs[B.TAB_FILES[1]]) == 1 + len(B.ARMS)
    and len(tabs[B.TAB_FILES[2]]) == 1 + len(B.ARMS) * len(B.CAPS)
    and all(len({len(r) for r in t}) == 1 for t in tabs.values())
)
row_gam = tabs[B.TAB_FILES[0]][1]
hdr = tabs[B.TAB_FILES[0]][0]
tab_ok &= (
    row_gam[0] == "gam"
    and float(row_gam[hdr.index("known_share_mean")])
    == doc["results"]["arms"]["gam"]["known_share_top10"]["mean"]
)
check(
    ";" not in rep and cols_ok and all(s in rep for s in sections) and tab_ok,
    "3: the report has every section, tables with even rows and no "
    "semicolon, and the three CSV tables read back with the JSON's "
    "numbers",
)

# 4: one worker against three
first = {k: Path(v).read_bytes() for k, v in OUT.items()}
real_block = B._block
calls = {"n": 0}


def stopping(c, lo, hi):
    calls["n"] += 1
    if calls["n"] == B.PROBE_USERS + 2:
        raise KeyboardInterrupt
    return real_block(c, lo, hi)


B._block = stopping
stopped = False
try:
    B.run(
        stage(),
        FakeBeyondInputs(),
        CHAIN,
        SCR / "beyond_resume",
        say=quiet,
        n_verify=6,
        inputs=BS.provenance(),
        workers=1,
    )
except KeyboardInterrupt:
    stopped = True
finally:
    B._block = real_block
CAP.lines.clear()
B.run(
    stage(),
    FakeBeyondInputs(),
    CHAIN,
    SCR / "beyond_resume",
    say=common.log.info,
    n_verify=6,
    inputs=BS.provenance(),
    workers=1,
)
again = {k: Path(v).read_bytes() for k, v in OUT.items()}
same = {k: first[k] == again[k] for k in first}
differ = [k for k, v in same.items() if not v]
check(
    stopped
    and not differ
    and any("pass resume, 1 of 4 blocks done" in ln for ln in CAP.lines),
    f"4: a run stopped inside the pass and resumed with 1 worker writes "
    f"outputs byte-identical to the run with 3 workers ({sorted(same)}). "
    f"Differ: {differ or 'none'}",
)
BS.check()

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)
