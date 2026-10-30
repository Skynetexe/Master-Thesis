"""Smoke test of pipeline/p14_cold.py and pipeline/p14.py on the
P13 chain of p13_chain.py. Every path points into a temporary folder, and
nothing touches the real model folder.

1. Units without the chain: the paired t-test against scipy, Holm and
   Benjamini-Yekutieli against values computed by hand, the permutation
   test against the t-test, the leakage shares against a brute force, the
   top lists against a full sort, and place().
2. The chain with the runner's markers, as test_tree_audit.py makes them:
   gam_refit, p13c_stability, a forced bounded p13d_bounds and
   p13c_final, and fake markers for the stages behind the other inputs.
3. p14_cold_gam through the runner: the fit at KKT_TOL in its monotone
   order, the read-back, a second call that skips, and a refused change.
4. p14_eval through the runner on synthetic inputs: the eval frame from the
   world's held-out picks, timestamps that P2's split could give, ALS
   factors, and P5's and P6's stored ranks from their own routes. Then
   independent recomputations: every eval listener's ranks, top lists and
   backgrounds by a dense full sort written here, the reasons (D1, S4) by a
   loop written here, the metrics and the Holm and BY decisions from the
   stored ranks, and the leakage shares by a brute force. A run stopped
   inside the pass and resumed with 1 worker gives the arrays of the first
   run with 2 workers. The read-back refuses a changed result, a changed
   stored P6 rank stops the run, and the report has every section.

Usage, from the repository root:
    python3 test_p14.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy import stats

import p13_chain as C
import p13_world as W
from pipeline import als
from pipeline import catalog as ct
from pipeline import common
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import config
from pipeline import evaluation as ev
from pipeline import gam
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
from pipeline import itemknn_select as IK
from pipeline import p13c as PC
from pipeline import p13d as D
from pipeline import p14 as P
from pipeline import p14_cold as PK
from pipeline import runner as RN
from pipeline import stage_p13a as s13
from pipeline import stage_p13b as s13b
from pipeline import stage_p13b2 as s13b2

failures = []
T0 = time.time()
PARAMS = dict(config.DEFAULT_PARAMS)
FAKED = ("p6_als", "p9b_colisten", "p13b2_loss", "p5_eval_baseline", "als_phase_d")
SIX = [
    "itemknn_select",
    "gam_knn_slices",
    "gam_refit",
    "p13c_stability",
    "p13d_bounds",
    "p13c_final",
]


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

# 1: units without the chain
rng = np.random.default_rng(14)
a_, b_ = rng.normal(0.3, 1.0, 300), rng.normal(0.0, 1.0, 300)
tt = P.t_test(a_ - b_, "two-sided")
ref = stats.ttest_rel(a_, b_)
tg = P.t_test(a_ - b_, "greater")
ref_g = stats.ttest_1samp(a_ - b_, 0.0, alternative="greater")
check(
    abs(tt["t"] - ref.statistic) < 1e-12
    and abs(tt["p"] - ref.pvalue) < 1e-12
    and abs(tg["p"] - ref_g.pvalue) < 1e-12,
    "1: the paired t-test equals scipy's (two-sided and one-sided)",
)
p4 = [0.01, 0.04, 0.03, 0.005]
h4 = P.holm(p4)
c4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
b4 = P.by_adjust(p4)
want_b = [4 * c4 * 0.01 / 2, 4 * c4 * 0.03 / 3, 4 * c4 * 0.03 / 3, 4 * c4 * 0.005 / 1]
check(
    np.allclose(h4, [0.03, 0.06, 0.06, 0.02], rtol=0, atol=1e-15)
    and np.allclose(b4, want_b, rtol=0, atol=1e-15),
    "1: Holm and Benjamini-Yekutieli equal values computed by hand",
)
d_ = rng.normal(0.08, 1.0, 400)
pt = P.t_test(d_, "two-sided")["p"]
pp = P.perm_test(d_, "two-sided", [1, 2], n_perm=20_000)
pp2 = P.perm_test(d_, "two-sided", [1, 2], n_perm=20_000)
pg = P.perm_test(d_, "greater", [1, 3], n_perm=20_000)
check(
    abs(pp - pt) < 0.02 and pp == pp2 and abs(pg - P.t_test(d_, "greater")["p"]) < 0.02,
    f"1: the permutation test agrees with the t-test (p {pp:.4f} against "
    f"{pt:.4f}) and repeats exactly with its seed",
)

# leakage: a brute force on pairs that never share a minute with an anchor
n_pairs, n_users = 4000, 30
tu = np.sort(rng.integers(1, n_users + 1, n_pairs))
tf = 120 * rng.integers(0, 50_000, n_pairs) + 30
tl = tf + 120 * rng.integers(0, 2000, n_pairs)
tw = rng.integers(1, 50, n_pairs)
fu = np.arange(1, n_users + 1)
anc = 120 * rng.integers(0, 52_000, (n_users, 2)) + 75
Cl = P.leakage_counts(tu, tf, tl, tw, fu, anc)
sh = P.leakage_shares(anc, Cl)
ok = True
for i, u in enumerate(fu):
    own = tu == u
    for c in range(2):
        t_ = anc[i, c]
        ok &= sh["global_lo_pairs"][i, c] == np.mean(tf > t_)
        ok &= sh["global_hi_pairs"][i, c] == np.mean(tl > t_)
        ok &= sh["global_lo_listens"][i, c] == tw[tf > t_].sum() / tw.sum()
        ok &= sh["own_hi_listens"][i, c] == tw[own & (tl > t_)].sum() / tw[own].sum()
        ok &= sh["own_lo_pairs"][i, c] == np.mean(tf[own] > t_)
anc_same = anc.copy()
anc_same[0, 0] = tf[5] + 10  # the minute of a pair's first play
sh2 = P.leakage_shares(anc_same, Cl)
exact = np.mean(tf > anc_same[0, 0])
check(
    ok and sh2["global_lo_pairs"][0, 0] <= exact <= sh2["global_hi_pairs"][0, 0],
    "1: the leakage shares equal a brute force when no pair shares the "
    "anchor's minute, and bound it when one does",
)

ok = True
for trial in range(30):
    n = int(rng.integers(5, 200))
    s = np.round(rng.normal(size=n), 1)
    s[rng.random(n) < 0.2] = -np.inf
    keep = int(rng.integers(1, 25))
    ids, sc = np.empty(0, dtype=np.int64), np.empty(0)
    for a in range(0, n, 17):
        b = min(a + 17, n)
        fl = sc[-1] if len(sc) == keep else -np.inf
        i2, s2 = P._range_top(s[a:b], a, keep, fl)
        ids, sc = P._merge_top(ids, sc, i2, s2, keep)
    fin = np.flatnonzero(np.isfinite(s))
    want = fin[np.lexsort((fin, -s[fin]))][:keep]
    ok &= np.array_equal(ids, want)
check(
    ok,
    "1: the top lists by ranges equal a full sort (ties by slot, "
    "-inf never listed) in 30 random cases",
)

# term_sums against catalog.reference_scores, on tables with flat parts
kn5 = tuple(np.linspace(0.0, 1.0, 9))
tt5 = [
    gam.Term("tag_affinity", "tag_affinity", kn5, zero_state=True, missing_state=True),
    gam.Term(
        "genre_affinity",
        "genre_affinity",
        kn5,
        zero_state=True,
        missing_state=True,
        tiers=(1, 2, 3),
    ),
    gam.Term(
        "audio_fit", "audio_fit", tuple(np.linspace(-3.0, 0.0, 7)), missing_state=True
    ),
    gam.Term(
        "artist_affinity", "artist_affinity", kn5, zero_state=True, missing_state=True
    ),
]
T5 = np.round(2 * rng.normal(size=(3, sum(t.size for t in tt5)))) / 2
T5[1, : tt5[0].size] = 0.0  # one term absent from one table
sc5 = ct.CatalogScorer(tt5, T5, W.SPACE, chunk=64)
ok5, split5, groups5 = True, 0, 0
for u in range(1, 41):
    st5 = cn.build_user_state(
        *cd.history_of(W.u_indptr, W.u_indices, W.u_data, u), W.SPACE
    )
    full5 = cn.full_catalog(st5, W.SPACE)
    ref5 = ct.reference_scores(full5, tt5, T5, 0.0, W.tier)
    sub5 = np.sort(rng.choice(W.NI, 60, replace=False))
    ok5 &= np.array_equal(P.term_sums(tt5, T5, full5, W.tier), ref5) and np.array_equal(
        P.term_sums(
            tt5,
            T5,
            {
                t.source: (full5[t.source][0][sub5], full5[t.source][1][sub5])
                for t in tt5
            },
            W.tier[sub5],
        ),
        ref5[:, sub5],
    )
    old5 = sc5.chunk_scores(st5, W.SPACE, sc5.prepare(st5, W.SPACE), 0.0, 0, W.NI).T
    for c in range(3):
        _, inv5, cnt5 = np.unique(ref5[c][1:], return_inverse=True, return_counts=True)
        g5 = np.flatnonzero(cnt5 > 1)
        groups5 += len(g5)
        split5 += sum(len(np.unique(old5[c][1:][inv5 == g])) > 1 for g in g5)
check(
    ok5 and split5 > 0,
    f"1: term_sums equals catalog.reference_scores bit for bit for 40 "
    f"listeners (tiers, flat parts, a zero table), on the catalog and on "
    f"samples of slots. CatalogScorer's arithmetic splits {split5} of the "
    f"reference's {groups5} exact ties",
)

t1 = gam.Term(
    "tag_affinity", "tag_affinity", (0.0, 0.5, 1.0), zero_state=True, missing_state=True
)
t2 = gam.Term("audio_fit", "audio_fit", (-2.0, 0.0), missing_state=True)
t3 = gam.Term(
    "genre_affinity", "genre_affinity", (0.0, 1.0), zero_state=True, missing_state=True
)
src = np.arange(t1.size + t3.size, dtype=float) + 1
out_ = P.place([t1, t2, t3], [t1, t3], src)
o2 = t1.size + t2.size
try:
    P.place(
        [t1, t2],
        [
            gam.Term(
                "tag_affinity",
                "tag_affinity",
                (0.0, 1.0),
                zero_state=True,
                missing_state=True,
            )
        ],
        np.ones(t1.size - 1),
    )
    refused = False
except ValueError:
    refused = True
check(
    np.array_equal(out_[: t1.size], src[: t1.size])
    and not out_[t1.size : o2].any()
    and np.array_equal(out_[o2:], src[t1.size :])
    and refused,
    "1: place() copies each term by name, leaves a missing term at 0 and "
    "refuses other knots",
)

# 2: the chain and its markers
print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_p14_"))
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
rc_f = execute(STS)
ptr = json.loads((CHAIN / PC.FINAL_FILE).read_text())
check(
    rc_a == 0
    and rc_f == 0
    and ptr["model"]["changed"]
    and all(common.marker_path(n).exists() for n in SIX),
    f"2: the chain has current markers for the six stages and a bounded "
    f"final model ({clock()})",
)

# 3: the cold-track GAM
COLD = PK.StageP14Cold(CHAIN, SCR)
COLD._stage = stage
CAP.lines.clear()
rc = execute(STS + [COLD])
cdoc = json.loads((CHAIN / PK.OUT_FILES[0]).read_text())
cz = dict(np.load(CHAIN / PK.OUT_FILES[1]))
base = stage()._base_context()
M_fin = PC.load_p13d_model(CHAIN, base)[0]
kept_c = [n for n in M_fin["kept"] if n != PK.DROP]
check(
    rc == 0
    and cdoc["model"]["kept"] == kept_c
    and cdoc["check"]["kkt"] <= s13.KKT_TOL
    and cdoc["check"]["monotone_violation"] <= s13b.MONOTONE_TOL
    and cdoc["check"]["dense"]["listeners"] == 6
    and common.marker_path("p14_cold_gam").exists()
    and any("selection pass" in ln and "p14_cold_gam" in ln for ln in CAP.lines),
    f"3: p14_cold_gam fits {len(kept_c)} terms to KKT_TOL in their "
    f"monotone order, passes the dense check and writes its marker "
    f"({clock()})",
)
lay_c, idx_c = PC.layout_of(base, kept_c, cdoc["model"]["bounded"])
Tc = np.asarray(cdoc["model"]["tables"])
Tf = np.zeros(base["L_u"])
Tf[idx_c] = Tc
s_c = (
    np.where(cz["ranks"][0] <= 100, 1 / np.log2(cz["ranks"][0] + 1.0), 0)
    .mean(axis=1)
    .mean()
)
check(
    abs(cdoc["selection"]["score"] - s_c) < 1e-15
    and np.array_equal(
        cz["final_ranks"], np.load(CHAIN / ptr["files"]["arrays"])["ranks"][0]
    )
    and set(cdoc["model"]["bounded"]) <= set(M_fin["source"]["bounded"]),
    f"3: the cold-track selection score follows from its stored ranks "
    f"({s_c:.5f}) and keeps the final model's bounds",
)
CAP.lines.clear()
rc2 = execute(STS + [COLD])
check(
    rc2 == 0 and any("skip p14_cold_gam" in ln for ln in CAP.lines),
    "3: a second runner call skips p14_cold_gam",
)
saved = (CHAIN / PK.OUT_FILES[0]).read_bytes()
bad = json.loads(saved)
bad["selection"]["score"] += 1e-6
(CHAIN / PK.OUT_FILES[0]).write_text(json.dumps(bad))
try:
    COLD.check()
    refused = False
except ValueError as exc:
    refused = "selection" in str(exc)
(CHAIN / PK.OUT_FILES[0]).write_bytes(saved)
COLD.check()
check(
    refused,
    "3: the cold-track read-back refuses a changed score and "
    "accepts the stored outputs",
)

# 4: P14 on synthetic inputs
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
anchor = P.anchor_of(FRAME)
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
    def __init__(self, stored=None):
        self.d = CHAIN
        self.stored = stored or STORED

    def key(self):
        return {"fake": "test_p14"}

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
        return self.stored[which]


P.EVAL_CHUNK, P.EVAL_BLOCK, P.PROBE_USERS = 64, 7, 3
P.N_PERM = 2000
EVS = P.StageP14(CHAIN, SCR)
EVS._stage = stage
EVS._io = lambda: FakeInputs()
EVS.workers = 2
CAP.lines.clear()
rc = execute(STS + [COLD, EVS])
doc = json.loads((CHAIN / P.OUT_FILES[0]).read_text())
z = dict(np.load(CHAIN / P.OUT_FILES[1]))
check(
    rc == 0
    and common.marker_path("p14_eval").exists()
    and doc["data"]["listeners"] == NE
    and doc["check"]["dense"]["listeners"] == 6
    and doc["check"]["popularity_equals_p5"]
    and doc["check"]["als_p6_equals_p6"]
    and doc["check"]["als_d_selection_gap"] <= 1e-12
    and any("probe listener" in ln for ln in CAP.lines)
    and any("eval pass" in ln and "2 workers" in ln for ln in CAP.lines),
    f"4: p14_eval runs through the runner with 2 workers, its checks "
    f"pass and it writes its marker ({clock()})",
)
CAP.lines.clear()
check(
    execute(STS + [COLD, EVS]) == 0 and any("skip p14_eval" in ln for ln in CAP.lines),
    "4: a second runner call skips p14_eval",
)

# The GAM arms by a dense full sort written here
G14 = P.load_models(CHAIN, base)
terms = G14["terms"]
tables = G14["tables"]
offs = P.offsets_of(terms)


def term_value(t, tab, v, h):
    """A term's value per slot by interpolation on its knots."""
    K = t.K
    kn = t.knot_array
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t.zero_state:
        val = np.where(h & (v == 0.0), tab[K + gam.ZERO], val)
    return np.where(h, val, tab[K + gam.MISS] if t.missing_state else 0.0)


def dense_terms(full, T):
    """(J, n_slots) term values of a table of the final layout."""
    return np.stack(
        [
            term_value(t, T[o : o + t.size], full[t.source][0], full[t.source][1])
            for t, o in zip(terms, offs)
        ]
    )


def dense_score(full, T):
    """The sum of the term values in term order, from 0."""
    s = np.zeros(W.NI)
    for v in dense_terms(full, T):
        s += v
    return s


def exact(sc, t):
    """(P5 rank, best and worst rank of the tie) of slot t."""
    g = int(np.count_nonzero(sc > sc[t]))
    e = int(np.count_nonzero(sc == sc[t]))
    return 1 + g + int(np.count_nonzero(sc[:t] == sc[t])), 1 + g, g + e


def exact_checks(zz, k, full, cand, tabs, picks_k):
    """Ranks, tie spans, pick scores and top lists of one listener in
    the arrays zz (listeners first) against the dense sum, exactly."""
    ok = True
    for c in range(len(tabs)):
        sc = np.where(cand, dense_score(full, tabs[c]), -np.inf)
        for p in range(2):
            t = int(picks_k[p])
            got = (
                int(zz["ranks"][k, c, p]),
                int(zz["span_lo"][k, c, p]),
                int(zz["span_hi"][k, c, p]),
            )
            ok &= got == exact(sc, t) and zz["pick_score"][k, c, p] == sc[t]
        order = np.lexsort((np.arange(W.NI), -sc))[: P.TOP_KEEP]
        ok &= np.array_equal(zz["top_id"][k, c], order) and np.array_equal(
            zz["top_score"][k, c], sc[order]
        )
    return bool(ok)


TOL = 1e-9
ok_r = ok_b = True
fulls, cands = {}, {}
zr = dict(z, ranks=np.stack([z[f"rank_{a}"] for a in P.GAM_ARMS], axis=1))
for k, u in enumerate(EU):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    state = cn.build_user_state(items, weights, W.SPACE)
    full = cn.full_catalog(state, W.SPACE)
    cand = np.ones(W.NI, dtype=bool)
    cand[0] = False
    cand[items] = False
    fulls[k], cands[k] = full, cand
    ok_r &= exact_checks(zr, k, full, cand, tables, FRAME["picks"][k])
    vt = dense_terms(full, tables[0])
    for j, (t, o) in enumerate(zip(terms, offs)):
        bg = z["bsum"][k, o : o + t.size] @ tables[0][o : o + t.size] / z["n_cand"][k]
        ok_b &= abs(float(vt[j][cand].mean()) - float(bg)) <= TOL
    ok_b &= int(z["n_cand"][k]) == int(cand.sum())
check(
    ok_r
    and ok_b
    and doc["check"]["slots_checked"] > 0
    and set(doc["ties"]) == set(P.GAM_ARMS),
    f"4: all {NE} eval listeners: the three GAM arms' ranks, tie spans, "
    f"pick scores and top lists equal a dense sum in term order exactly, "
    f"and the per-term backgrounds within {TOL:.0e} "
    f"({doc['check']['slots_checked']:,} slots checked in the pass)",
)

# Ties: the tables without the popularity term, as on the real data,
# for every listener of the world, through the pass's own function
tabs0 = tables.copy()
for j_, t_ in enumerate(terms):
    if t_.source == "popularity":
        tabs0[:, offs[j_] : offs[j_] + t_.size] = 0.0
U_ALL = np.arange(1, len(W.u_indptr) - 1)
PK_ALL = np.array([[W.held[u][0], W.held[u][2]] for u in U_ALL], dtype=np.int64)
sc0 = ct.CatalogScorer(terms, tabs0, W.SPACE, chunk=P.EVAL_CHUNK)
bc0, ib0 = P.basis_counts(sc0, W.SPACE)
names_ = [t.name for t in terms]
ctx0 = dict(
    scorer=sc0,
    space=W.SPACE,
    users=U_ALL,
    picks=PK_ALL,
    u_indptr=W.u_indptr,
    u_indices=W.u_indices,
    u_data=W.u_data,
    chunk=P.EVAL_CHUNK,
    base_counts=bc0,
    item_basis=ib0,
    groups=[
        (gn, [names_.index(n) for n in g])
        for gn, g in PC.reason_groups(G14["M"]["kept"])
    ],
    sources=[t.source for t in terms],
)
outs0 = [P._eval_one(ctx0, k) for k in range(len(U_ALL))]
z0 = {key: np.stack([o[key] for o in outs0]) for key in outs0[0]}
ok0, moved0 = True, 0
for k, u in enumerate(U_ALL):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    state = cn.build_user_state(items, weights, W.SPACE)
    cand = np.ones(W.NI, dtype=bool)
    cand[0] = False
    cand[items] = False
    ok0 &= exact_checks(z0, k, cn.full_catalog(state, W.SPACE), cand, tabs0, PK_ALL[k])
    old0 = sc0.chunk_scores(state, W.SPACE, sc0.prepare(state, W.SPACE), 0.0, 0, W.NI).T
    for c in range(len(tabs0)):
        so = np.where(cand, old0[c], -np.inf)
        moved0 += sum(
            exact(so, int(PK_ALL[k, p]))[0] != int(z0["ranks"][k, c, p])
            for p in range(2)
        )
tied0 = int(np.count_nonzero(z0["span_hi"] > z0["span_lo"]))
check(
    ok0 and tied0 > 0,
    f"4: without the popularity term, as on the real data, {tied0} of "
    f"{z0['ranks'].size} pick ranks lie in a tie. For all {len(U_ALL)} "
    f"listeners the pass's ranks, tie spans and top lists equal the dense "
    f"sum exactly ({clock()})",
)
# How many pick ranks CatalogScorer's arithmetic (sparse products) would
# move against the dense sum depends on the rounding of the installed numpy
# and scipy: some versions move a few ranks in this world, others none. So
# the count is information, not a check.
print(
    f"INFO  4: CatalogScorer's arithmetic moves {moved0} pick ranks against "
    f"the dense sum with this numpy and scipy",
    flush=True,
)

# The reasons by a loop written here
groups = PC.reason_groups(G14["M"]["kept"])
jx = {t.name: j for j, t in enumerate(terms)}


def reason_values(T):
    """(G, n, 10) values and candidates against the candidate sets."""
    val = np.zeros((len(groups), NE, P.REASON_K))
    cnd = np.zeros((len(groups), NE, P.REASON_K), dtype=bool)
    for k in range(NE):
        vt = dense_terms(fulls[k], T)
        top = z["top_id"][k, 0, : P.REASON_K]
        for g, (_, names) in enumerate(groups):
            for nme in names:
                j = jx[nme]
                t = terms[j]
                v, h = fulls[k][t.source]
                val[g, k] += vt[j][top] - vt[j][cands[k]].mean()
                on = h[top] & ~(t.zero_state & (v[top] == 0.0))
                cnd[g, k] |= on
    return val, cnd


def top3(val, cnd):
    G_ = val.shape[0]
    r = np.zeros(val.shape, dtype=np.int64)
    for k in range(NE):
        for i in range(P.REASON_K):
            idx = [g for g in range(G_) if cnd[g, k, i]]
            idx.sort(key=lambda g: (-val[g, k, i], g))
            for g in idx[:3]:
                r[g, k, i] = 1
    return r.astype(bool)


M14 = G14["M"]
v0, cnd0 = reason_values(M14["tables"])
pos = np.zeros(v0.shape, dtype=np.int64)
tp = np.zeros(v0.shape, dtype=np.int64)
boot = G14["boot"]["listener"]
for th in boot:
    vb, _ = reason_values(M14["lay"].tables(th))
    pos += cnd0 & (vb > 0)
    tp += cnd0 & top3(vb, cnd0)
B = len(boot)
d1 = (
    cnd0
    & (v0 > 0)
    & (pos >= math.ceil(0.95 * B - 1e-9))
    & (tp >= math.ceil(0.8 * B - 1e-9))
)
allpos = cnd0.copy()
tm = np.zeros(v0.shape, dtype=np.int64)
for T in G14["member_tables"]:
    vm, _ = reason_values(T)
    allpos &= vm > 0
    tm += cnd0 & top3(vm, cnd0)
s4 = allpos & (tm >= math.ceil(0.8 * len(G14["member_tables"]) - 1e-9))
shown = d1 & s4
near = np.abs(v0) < 1e-9
check(
    np.array_equal(cnd0, z["candidate"])
    and np.max(np.abs(v0 - z["reason_value"])) < 1e-9
    and np.array_equal(shown | near, z["shown"] | near)
    and int(z["shown"].sum()) > 0,
    f"4: the candidates, reason values and shown reasons (D1 over {B} "
    f"replicates, S4 over {len(G14['member_tables'])} members) equal a "
    f"loop written here ({int(z['shown'].sum())} of {z['shown'].size} "
    f"shown)",
)
odds_ok = True
for r_, (g, _) in zip(doc["reasons"]["rows"], enumerate(groups)):
    sg = z["shown"][g]
    if sg.any():
        odds_ok &= (
            abs(r_["odds_mean"] - float(np.exp(z["reason_value"][g][sg]).mean()))
            < 1e-12
        )
s10 = z["top_score"][:, 0, :10]
phi_sum = z["reason_value"].sum(axis=0)
margin = s10 - (z["score_sum"][:, 0] / z["n_cand"])[:, None]
check(
    odds_ok and np.max(np.abs(phi_sum - margin)) < 1e-9,
    "4: the reasons' odds factors follow from their values, and the "
    "reason values sum to the score minus the mean candidate score",
)

# The leave-the-top-10 share by a loop written here
cnt = lv = 0
for k in range(NE):
    for i in range(P.REASON_K):
        sh_ = [g for g in range(len(groups)) if z["shown"][g, k, i]]
        if not sh_:
            continue
        g = max(sh_, key=lambda g_: (z["reason_value"][g_, k, i], -g_))
        s_new = z["top_score"][k, 0, i] - z["reason_value"][g, k, i]
        s11, id11 = z["top_score"][k, 0, 10], z["top_id"][k, 0, 10]
        cnt += 1
        lv += int(s_new < s11 or (s_new == s11 and z["top_id"][k, 0, i] > id11))
lt = doc["reasons"]["leave_top10"]
check(
    lt["tracks"] == cnt and (cnt == 0 or abs(lt["share"] - lv / cnt) < 1e-12),
    f"4: the share of tracks that leave the top 10 without their top "
    f"reason follows a loop written here ({lv} of {cnt})",
)

# Metrics, tests and the drop from the stored ranks
ok = True
for arm in P.ARMS:
    r = z[f"rank_{arm}"].astype(float)
    ndcg = np.where(r <= 100, 1 / np.log2(r + 1), 0).mean(axis=1).mean()
    mrr = (1 / r).mean(axis=1).mean()
    r10 = (r <= 10).mean(axis=1).mean()
    ok &= (
        abs(doc["arms"][arm]["ndcg_at_100"] - ndcg) < 1e-15
        and abs(doc["arms"][arm]["mrr"] - mrr) < 1e-15
        and abs(doc["arms"][arm]["recall_at_10"] - r10) < 1e-15
    )
pop_ok = all(
    int(z["rank_popularity"][k, p])
    == ev.brute_rank(C.POP, FRAME["picks"][k, p], TRAIN[int(u)])
    for k, u in enumerate(EU)
    for p in range(2)
)
knn_ok = True
lists = C.LISTS
for k, u in enumerate(EU):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(u))
    w_ = np.zeros(W.NI)
    w_[items] = weights
    for kind, arm in (("same_artist", "knn_same"), ("both", "knn_both")):
        s = np.zeros(W.NI)
        for kk in (
            ["same_artist", "other_artists"] if kind == "both" else ["same_artist"]
        ):
            ip, nb, cs = lists[kk]
            for i in range(W.NI):
                s[i] += float(np.dot(cs[ip[i] : ip[i + 1]], w_[nb[ip[i] : ip[i + 1]]]))
        s[0] = -np.inf
        s[items] = -np.inf
        for p in range(2):
            pk = FRAME["picks"][k, p]
            lo = 1 + int(np.count_nonzero(s > s[pk] + 1e-12))
            hi = int(np.count_nonzero(s >= s[pk] - 1e-12))
            knn_ok &= lo <= int(z[f"rank_{arm}"][k, p]) <= hi
check(
    ok and pop_ok and knn_ok,
    "4: the metrics follow from the stored ranks, popularity equals P5's "
    "brute force and ItemKNN a loop written here for every listener",
)
lm = {
    a: np.where(
        z[f"rank_{a}"] <= 100, 1 / np.log2(z[f"rank_{a}"].astype(float) + 1), 0
    ).mean(axis=1)
    for a in P.ARMS
}
g_ = lm["gam"]
pv = []
for r_ in doc["tests"]["confirmatory"]:
    d_ = (
        1.02 * g_ - lm[r_["arm"]]
        if r_["alternative"] == "greater"
        else g_ - lm[r_["arm"]]
    )
    alt = "greater" if r_["alternative"] == "greater" else "two-sided"
    pv.append(stats.ttest_1samp(d_, 0.0, alternative=alt).pvalue)
order = np.argsort(pv)
holm_ = np.empty(4)
run_ = 0.0
for i, kk in enumerate(order):
    run_ = max(run_, min(1.0, (4 - i) * pv[kk]))
    holm_[kk] = run_
check(
    all(
        abs(r_["p"] - p_) < 1e-12
        and abs(r_["p_adjusted"] - h_) < 1e-12
        and r_["reject"] == (h_ <= 0.05)
        for r_, p_, h_ in zip(doc["tests"]["confirmatory"], pv, holm_)
    )
    and len(doc["tests"]["exploratory"]) == 31
    and doc["tests"]["confirmatory"][0]["kind"] == "non-inferiority",
    "4: the confirmatory p-values and Holm decisions equal scipy's t-test "
    "and a Holm loop written here, and the exploratory family holds 31 "
    "tests",
)
sel_g = np.load(CHAIN / ptr["files"]["arrays"])["ranks"][0].astype(float)
s_sel = np.where(sel_g <= 100, 1 / np.log2(sel_g + 1), 0).mean(axis=1)
check(
    abs(doc["drop"]["gam"]["drop"] - (g_.mean() - s_sel.mean())) < 1e-15
    and set(doc["drop"]) == set(P.ARMS),
    "4: the drop from selection to eval follows from the two samples' "
    "ranks, for every arm",
)

# A slice and the evidence, by direct computation
pc_ = C.pop_centiles()[FRAME["picks"]]
band = pc_ > 90
gr_ = z["rank_gam"].astype(float)
dcg = np.where(gr_ <= 100, 1 / np.log2(gr_ + 1), 0)
row = next(r_ for r_ in doc["slices"]["popularity band"] if r_["level"] == "91-100")
ok_slice = (
    row["picks"] == int(band.sum())
    and abs(row["arms"]["gam"]["ndcg_at_100"]["mean"] - float(dcg[band].mean())) < 1e-15
)
ok_ev = True
name_of = {g: n for g, (n, _) in enumerate(groups)}
for k in range(5):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, int(EU[k]))
    state = cn.build_user_state(items, weights, W.SPACE)
    for i in range(P.REASON_K):
        ev_ = cn.evidence(state, W.SPACE, int(z["top_id"][k, 0, i]))
        for g, (gname, names) in enumerate(groups):
            src = [
                terms[jx[n]].source
                for n in names
                if terms[jx[n]].source not in cn.CREDIT_CONCEPTS
            ]
            if not src:
                continue
            e = ev_[src[0]]
            rows = list(e["terms"]) if e["has_data"] else []
            if src[0] == "audio_fit":
                rows = sorted(rows, key=lambda r: (-r["term"], r["id"]))
                ids = [cn.AUDIO_DIMS.index(r["id"]) for r in rows]
            else:
                ids = [int(r["id"]) for r in rows]
            got = z["ev_id"][k, i, g][: min(len(ids), P.EVIDENCE_K)]
            ok_ev &= (
                int(z["ev_count"][k, i, g]) == len(ids)
                and list(got) == ids[: P.EVIDENCE_K]
            )
        art = ev_["artist_affinity"]
        if art["has_data"]:
            ok_ev &= int(z["ev_credit"][k, i]) == int(art["credit"]) and int(
                z["ev_credit_tracks"][k, i]
            ) == len(art["terms"])
check(
    ok_slice and ok_ev,
    "4: the popularity band 91-100 and the evidence of 5 listeners' top "
    "lists follow from a direct computation",
)

# Leakage by a brute force
ok = True
for k, u in enumerate(EU):
    own = TR[0] == u
    for c in range(2):
        t_ = anchor[k, c]
        ok &= z["leak_global_lo_pairs"][k, c] == np.mean(TR[1] > t_)
        ok &= z["leak_global_hi_pairs"][k, c] == np.mean(TR[2] > t_)
        ok &= z["leak_global_hi_listens"][k, c] == TR[3][TR[2] > t_].sum() / TR[3].sum()
        ok &= z["leak_own_hi_pairs"][k, c] == np.mean(TR[2][own] > t_)
ok &= bool(np.all(z["leak_own_lo_pairs"] == 0)) and bool(
    np.all(z["leak_own_hi_pairs"][:, 0] == 0)
)
check(
    ok and np.any(z["leak_own_hi_pairs"][:, 1] > 0),
    "4: the leakage shares equal a brute force, with the zeros that P2's "
    "split makes",
)

# Resume with 1 worker, against the run with 2 workers
first = {k: v.copy() for k, v in z.items()}
real_one = P._eval_one
calls = {"n": 0}


def stopping(c, k):
    calls["n"] += 1
    if calls["n"] == 12:
        raise KeyboardInterrupt
    return real_one(c, k)


P._eval_one = stopping
stopped = False
try:
    P.run(
        stage(),
        FakeInputs(),
        CHAIN,
        SCR / "p14_resume",
        say=quiet,
        n_verify=6,
        inputs=EVS.provenance(),
        workers=1,
    )
except KeyboardInterrupt:
    stopped = True
finally:
    P._eval_one = real_one
CAP.lines.clear()
P.run(
    stage(),
    FakeInputs(),
    CHAIN,
    SCR / "p14_resume",
    say=common.log.info,
    n_verify=6,
    inputs=EVS.provenance(),
    workers=None,
)
z2 = dict(np.load(CHAIN / P.OUT_FILES[1]))
same = all(
    np.array_equal(first[k], z2[k], equal_nan=first[k].dtype.kind == "f") for k in first
)
check(
    stopped
    and same
    and any("pass resume" in ln for ln in CAP.lines)
    and any("p14: selection memory" in ln for ln in CAP.lines),
    "4: a run stopped inside the pass and resumed with P13's worker "
    "setting (1 here) gives the arrays of the run with 2 workers",
)

# The read-back, a changed P6 rank and the report
saved = (CHAIN / P.OUT_FILES[0]).read_bytes()
bad = json.loads(saved)
bad["arms"]["gam"]["ndcg_at_100"] += 1e-6
(CHAIN / P.OUT_FILES[0]).write_text(json.dumps(bad))
try:
    EVS.check()
    refused = False
except ValueError as exc:
    refused = "'arms'" in str(exc)
(CHAIN / P.OUT_FILES[0]).write_bytes(saved)
EVS.check()
bad_p6 = dict(STORED)
r6 = STORED["p6"][1].copy()
r6[3, 1] += 1
bad_p6["p6"] = (STORED["p6"][0], r6)
try:
    P.run(
        stage(),
        FakeInputs(bad_p6),
        CHAIN,
        SCR / "p14_bad",
        say=quiet,
        n_verify=6,
        inputs=EVS.provenance(),
        workers=1,
    )
    stopped_p6 = False
except ValueError as exc:
    stopped_p6 = "P6's stored eval ranks" in str(exc)
check(
    refused and stopped_p6,
    "4: the read-back refuses a changed result, and a changed stored P6 "
    "rank stops the run",
)
rep = (CHAIN / "reports" / P.REPORT_NAME).read_text()
cols_ok, block = True, []
for ln in rep.splitlines() + [""]:
    if ln.startswith("|"):
        block.append(ln.count("|"))
    elif block:
        cols_ok &= len(set(block)) == 1
        block = []
sections = (
    "## Accuracy",
    "## Confirmatory tests",
    "## Exploratory tests",
    "## Drop from selection to eval",
    "## Slices",
    "## Reasons",
    "## Leakage",
)
check(
    ";" not in rep and cols_ok and all(s in rep for s in sections),
    "4: the report has every section, tables with even rows and no " "semicolon",
)
cold_rep = (CHAIN / "reports" / PK.REPORT_NAME).read_text()
check(
    ";" not in cold_rep and "## Selection" in cold_rep and "## Shapes" in cold_rep,
    "3: the cold-track report has its " "sections and no semicolon",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)
