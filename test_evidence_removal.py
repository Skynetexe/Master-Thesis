"""Smoke test of pipeline/evidence_removal.py (the runner stage
evidence_removal) on the P13 chain of p13_chain.py. Every path points into
a temporary folder, and nothing touches the real model folder.

1. A hand-made world and listener whose top track rests on one artist,
   with the answers known by hand: with the shown co-listened tracks
   removed it stays first, with all of its artist's tracks removed it
   leaves the top 10, and with only them kept it stays. The same for the
   other passes, the scan against the module's dense route, and units:
   the profile without units and without audio against a profile built
   here, concept_rows() against one-slot ranges, and the bound against
   every slot without artist evidence.
2. The chain with its markers, p14_cold_gam and p14_eval through the
   runner on synthetic inputs, as test_beyond.py makes them.
3. evidence_removal through the runner with 3 workers: every new rank of
   every pass of every eval listener against a route written here (the
   passes built with sets from the world's lists and P14's stored
   evidence, concepts.full_catalog, a dense sum in term order and a full
   sort), then the same with every candidate reason shown, so that every
   kind of pass runs. Every measure against loops written here, a second
   call that skips, a refused change of the results, a changed stored
   evidence unit and a changed P14 rank that stop the run, the report and
   the tables.
4. One worker against three: a run stopped inside the pass and resumed
   with one worker writes outputs byte-identical to the run with three.
5. The rule of the listeners: with a sample of 20 and a limit of 0 hours
   the stage keeps the first 20 of the random order, and with a large
   limit every listener runs.

Usage, from the repository root:
    python3 test_evidence_removal.py
With P13B2_BASE_CACHE=<folder> and P13_CHAIN_CACHE=<folder> the chain's
outputs are reused while their code stays the same.
With KEEP_TMP=1 the temporary folder stays, for a look at the outputs.
"""

import csv
import dataclasses
import json
import logging
import math
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import p13_chain as C
import p13_world as W
from pipeline import als
from pipeline import catalog as ct
from pipeline import common
from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import config
from pipeline import evaluation as ev
from pipeline import evidence_removal as ER
from pipeline import gam
from pipeline import gam_knn_diag as G
from pipeline import gam_refit as GR
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


def term_value(t, tab, v, h):
    K = t.K
    kn = t.knot_array
    x = np.clip(np.where(h, v, kn[0]), kn[0], kn[-1])
    val = np.interp(x, kn, tab[:K])
    if t.zero_state:
        val = np.where(h & (v == 0.0), tab[K + gam.ZERO], val)
    return np.where(h, val, tab[K + gam.MISS] if t.missing_state else 0.0)


def dense_score(full, terms_, T_, n):
    """The sum of the term values in term order, from 0."""
    s = np.zeros(n)
    for t, o in zip(terms_, P.offsets_of(terms_)):
        s += term_value(t, T_[o : o + t.size], full[t.source][0], full[t.source][1])
    return s


def full_sort_ranks(s, excl, slots):
    """Ranks by a full sort written here: score descending, then id."""
    out = set(int(x) for x in excl) | {0}
    cand = [i for i in range(len(s)) if i not in out]
    order = sorted(cand, key=lambda i: (-float(s[i]), i))
    pos = {i: p + 1 for p, i in enumerate(order)}
    return [pos[int(t)] for t in slots]


def drop_units(state, space, units):
    """A profile without units, written here: the shares set to 0 and the
    norm by the definition of concepts.build_user_state."""
    share, norm = dict(state.term_share), dict(state.term_norm)
    for src, ids in units.items():
        name = cn.TERM_CONCEPTS[src]
        p = np.array(share[name])
        for i in ids:
            p[int(i)] = 0.0
        share[name] = p
        norm[name] = float(np.sqrt(np.sum((space.idf[name] * p) ** 2)))
    return dataclasses.replace(state, term_share=share, term_norm=norm)


def no_audio(state):
    D_ = len(state.audio_dim_ok)
    return dataclasses.replace(
        state,
        audio_mu=np.zeros(D_),
        audio_sigma=np.ones(D_),
        audio_dim_ok=np.zeros(D_, dtype=bool),
    )


# 1: a hand-made world
NH = 40
ART = [None] + [0] * 9 + [1] * 5 + [2] * 5 + [3] * 5 + [4] * 15  # A B C D E
cr_rows = [[] if a is None else [a] for a in ART]
credit_h = sp.csr_matrix(
    (
        np.ones(NH - 1),
        np.array(ART[1:]),
        np.concatenate(([0], np.cumsum([len(r) for r in cr_rows]))),
    ),
    shape=(NH, 5),
)
nb_rows, cs_rows = [[]], [[]]
for i in range(1, NH):
    same = [j for j in range(1, NH) if j != i and ART[j] == ART[i]]
    nb_rows.append(same)
    cs_rows.append([0.5] * len(same))
ip_nb = np.concatenate(([0], np.cumsum([len(r) for r in nb_rows])))
same_h = sp.csr_matrix(
    (
        np.array([x for r in cs_rows for x in r]),
        np.array([x for r in nb_rows for x in r], dtype=np.int64),
        ip_nb,
    ),
    shape=(NH, NH),
)
empty = sp.csr_matrix((NH, 3))
SPACE_H = cn.ItemSide(
    term_mats={
        "tags_direct": empty.copy(),
        "tags_artist": empty.copy(),
        "genres": empty.copy(),
    },
    credit_mats={"artists": credit_h, "works": sp.csr_matrix((NH, 1))},
    colisten={"same_artist": same_h, "other_artists": sp.csr_matrix((NH, NH))},
    audio=np.full((NH, len(cn.AUDIO_DIMS)), np.nan),
    popularity=np.zeros(NH),
    tier=np.zeros(NH, dtype=np.int64),
)
TERMS_H = [
    gam.Term(
        "artist_affinity",
        "artist_affinity",
        (0.0, 0.25, 0.5, 1.0),
        zero_state=True,
        missing_state=True,
    ),
    gam.Term(
        "colisten_same_artist",
        "colisten_same_artist",
        (0.0, 0.1, 0.5),
        zero_state=True,
        missing_state=True,
    ),
]
T_H = np.array(
    [[0.0, 1.0, 2.0, 3.0, -2.0, -2.0, 0.0, 0.0, 0.0, 1.0, 2.0, -1.0, -1.0, 0.0, 0.0]]
)
HIST_H = list(range(1, 8)) + [10, 15, 20]
u_ip_h = np.array([0, 0, len(HIST_H)], dtype=np.int64)
u_ix_h = np.array(HIST_H, dtype=np.int32)
u_dt_h = np.array([2000] * 7 + [1000] * 3, dtype=np.uint16)
GROUPS_H = [("artist", [0, 1])]
st_h = cn.build_user_state(*cd.history_of(u_ip_h, u_ix_h, u_dt_h, 1), SPACE_H)
full_h = cn.full_catalog(st_h, SPACE_H)
s_h = dense_score(full_h, TERMS_H, T_H[0], NH)
order_h = sorted(
    [i for i in range(1, NH) if i not in HIST_H], key=lambda i: (-s_h[i], i)
)
top_h = np.array(order_h[:10])
PICKS_H = np.array([21, 30])
ctx_e = {"space": SPACE_H, "groups": GROUPS_H, "sources": [t.source for t in TERMS_H]}
evs = [P._evidence(ctx_e, st_h, int(t)) for t in top_h]
Z_H = {
    "user_id": np.array([1]),
    "picks": PICKS_H[None],
    "top_id": top_h[None, None],
    "top_score": s_h[top_h][None, None],
    "top_val": np.stack([full_h[t.source][0][top_h] for t in TERMS_H], axis=1)[None],
    "top_has": np.stack([full_h[t.source][1][top_h] for t in TERMS_H], axis=1)[None],
    "shown": np.ones((1, 1, 10), dtype=bool),
    "ev_count": np.stack([e[0] for e in evs])[None],
    "ev_id": np.stack([e[1] for e in evs])[None],
    "ev_val": np.stack([e[2] for e in evs])[None],
    "ev_credit": np.array([[e[3] for e in evs]]),
    "ev_credit_tracks": np.array([[e[4] for e in evs]]),
    "rank_gam": np.array([full_sort_ranks(s_h, HIST_H, PICKS_H)]),
}
SCORER_H = ct.CatalogScorer(TERMS_H, T_H, SPACE_H, chunk=8)
CTX_H = ER.context(
    SCORER_H,
    SPACE_H,
    TERMS_H,
    T_H,
    GROUPS_H,
    Z_H,
    (u_ip_h, u_ix_h, u_dt_h),
    chunk=8,
    stride=3,
)
RES_H = ER.listener(CTX_H, 0)
DEN_H = ER.listener(CTX_H, 0, dense=True)
check(
    top_h.tolist() == [8, 9, 11, 12, 13, 14, 16, 17, 18, 19]
    and Z_H["rank_gam"][0].tolist() == [11, 20]
    and Z_H["ev_id"][0, 0, 0].tolist() == [1, 2, 3, 4, 5]
    and int(Z_H["ev_count"][0, 0, 0]) == 7
    and CTX_H["bound"] == -3.0,
    "1: the hand-made listener's top 10 (A A, then B, C and D by id), "
    "pick ranks 11 and 20, artist A's shown evidence (5 of 7 co-listened "
    "tracks) and the bound -3 are as built",
)
check(
    RES_H["rank_shown"].tolist() == [1, 2, 11]
    and RES_H["rank_all"].tolist() == [13, 14, 11, 12, 13, 14, 11, 12, 13, 14]
    and RES_H["rank_kept"].tolist() == [1, 2, 1, 2, 3, 4, 1, 2, 3, 4]
    and RES_H["rank_list"].tolist() == [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    "1: the answers known by hand: the top track stays first without its "
    "5 shown co-listened tracks (2 of artist A's remain), leaves for rank "
    "13 without all of A's tracks and stays first with only them, a "
    "track of B leaves for rank 11 without its one co-listened track, "
    "and with every artist's evidence removed 6 of the 10 stay",
)
check(
    all(ER._same({k: RES_H[k]}, {k: DEN_H[k]}) for k in RES_H if k not in ER.SKIP_DENSE)
    and int(RES_H["plans"]) == 11
    and int(RES_H["full_scans"]) == 6
    and int(DEN_H["full_scans"]) == 11,
    f"1: the scans equal the module's dense route on all 11 passes, and "
    f"the bound sends 5 of them to the evidence slots "
    f"({int(RES_H['full_scans'])} full scans)",
)
want_h = {}
for name, keep in (
    ("shown0", [6, 7, 10, 15, 20]),
    ("shown2", list(range(1, 8)) + [15, 20]),
    ("allA", [10, 15, 20]),
    ("keptA", list(range(1, 8))),
    ("list", [20]),
):
    w_ = [2.0 if h <= 7 else 1.0 for h in keep]
    s_ = dense_score(
        cn.full_catalog(
            cn.build_user_state(np.array(keep), np.array(w_), SPACE_H), SPACE_H
        ),
        TERMS_H,
        T_H[0],
        NH,
    )
    want_h[name] = full_sort_ranks(s_, HIST_H, top_h)
check(
    want_h["shown0"][0] == 1
    and want_h["shown2"][2] == 11
    and want_h["allA"][:2] == [13, 14]
    and want_h["keptA"][:2] == [1, 2]
    and want_h["list"] == RES_H["rank_list"].tolist(),
    "1: a dense sum written here on the histories built by hand gives the "
    "same ranks",
)
bad_h = dict(Z_H, ev_id=Z_H["ev_id"].copy())
bad_h["ev_id"][0, 0, 0, 1] = 6
CTX_B = ER.context(
    SCORER_H,
    SPACE_H,
    TERMS_H,
    T_H,
    GROUPS_H,
    bad_h,
    (u_ip_h, u_ix_h, u_dt_h),
    chunk=8,
    stride=3,
)
check(
    raises(lambda: ER.listener(CTX_B, 0), "stored evidence lists"),
    "1: a stored evidence unit that differs from concepts.evidence stops "
    "the listener",
)

# Units on the synthetic world
rng = np.random.default_rng(16)
ok_u, ok_rows, ok_aud = True, True, True
for u in (3, 17, 41):
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, u)
    st0 = cn.build_user_state(items, weights, W.SPACE)
    units = {"tag_affinity": np.array([1, 5, 7]), "genre_affinity": np.array([0, 2])}
    a_ = ER.without_units(st0, W.SPACE, units)
    b_ = drop_units(st0, W.SPACE, units)
    for name in ("tags_direct", "genres", "tags_artist"):
        loop = math.sqrt(
            sum(
                (W.SPACE.idf[name][t] * b_.term_share[name][t]) ** 2
                for t in range(len(b_.term_share[name]))
            )
        )
        ok_u &= (
            np.array_equal(a_.term_share[name], b_.term_share[name])
            and a_.term_norm[name] == b_.term_norm[name]
            and abs(loop - b_.term_norm[name]) <= 1e-12 * max(1.0, loop)
        )
    ok_u &= a_.total == st0.total and np.array_equal(
        st0.term_share["tags_artist"], a_.term_share["tags_artist"]
    )
    fa = cn.full_catalog(ER.without_audio(st0), W.SPACE)
    ok_aud &= not fa["audio_fit"][1].any()
    rows = np.sort(rng.choice(np.arange(W.NI), size=90, replace=False))
    scorer_w = ct.CatalogScorer(
        [gam.Term("x", "audio_fit", (-5.0, 0.0))], np.zeros((1, 6)), W.SPACE, chunk=64
    )
    pre = scorer_w.prepare(st0, W.SPACE)
    for src in (
        "tag_affinity",
        "artist_tag_affinity",
        "genre_affinity",
        "artist_affinity",
        "colisten_same_artist",
        "audio_fit",
        "popularity",
    ):
        pre_s = dict(pre)
        saved = scorer_w.user_sources
        scorer_w.user_sources = [src]
        try:
            pre_s.update(scorer_w.prepare(st0, W.SPACE))
        finally:
            scorer_w.user_sources = saved
        v1, h1 = ER.concept_rows(scorer_w, st0, W.SPACE, pre_s, src, rows)
        v2 = np.array(
            [
                scorer_w.concept_range(st0, W.SPACE, pre_s, src, int(r), int(r) + 1)[0][
                    0
                ]
                for r in rows
            ]
        )
        h2 = np.array(
            [
                scorer_w.concept_range(st0, W.SPACE, pre_s, src, int(r), int(r) + 1)[1][
                    0
                ]
                for r in rows
            ]
        )
        vf, hf = scorer_w.concept_range(st0, W.SPACE, pre_s, src, 0, W.NI)
        ok_rows &= (
            np.array_equal(v1, v2, equal_nan=True)
            and np.array_equal(h1, h2)
            and np.array_equal(v1, vf[rows], equal_nan=True)
            and np.array_equal(h1, hf[rows])
        )
check(
    ok_u and ok_aud,
    "1: the profile without units equals a profile built here (shares, "
    "norm, and a norm summed in a loop within 1e-12), keeps the total "
    "weight and the other sources, and no audio data leaves audio fit "
    "without data at every slot",
)
check(
    ok_rows,
    "1: concept_rows() equals one-slot ranges and the "
    "catalogue's range on random slots for all seven concepts",
)

# 2: the chain and P14
print(W.summary())
TMP = Path(tempfile.mkdtemp(prefix="test_evidence_removal_"))
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
    by_ = dict(RN.STAGE_BY_NAME)
    by_.update({s.name: s for s in sts})
    return RN.execute(sts, PARAMS, [], by_name=by_, **kw)


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
        return {"fake": "test_evidence_removal"}

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


P.EVAL_CHUNK, P.EVAL_BLOCK, P.PROBE_USERS = 64, 7, 3
P.N_PERM = 500
EVS = P.StageP14(CHAIN, SCR)
EVS._stage = stage
EVS._io = lambda: FakeInputs()
EVS.workers = 1
rc = execute(STS + [COLD, EVS])
check(
    rc_a == 0 and rc == 0 and common.marker_path("p14_eval").exists(),
    f"2: the chain, p14_cold_gam and p14_eval run through the runner " f"({clock()})",
)
base = stage()._base_context()
G14 = P.load_models(CHAIN, base)
terms = G14["terms"]
T_GAM = np.asarray(G14["tables"][0], dtype=np.float64)
Z14 = dict(np.load(CHAIN / P.OUT_FILES[1]))
groups = [
    (name, [[t.name for t in terms].index(n) for n in g])
    for name, g in PC.reason_groups(G14["M"]["kept"])
]
names = [n for n, _ in groups]
GA = names.index("artist")
G_CONTENT = [
    g
    for g, (n, js) in enumerate(groups)
    if n != "artist" and terms[js[0]].source in cn.TERM_CONCEPTS
]
G_AUDIO = next(
    g for g, (n, js) in enumerate(groups) if terms[js[0]].source == "audio_fit"
)
ip_a, cols_a = W.SRC.credits["artists"][0], W.SRC.credits["artists"][1]
ARTS = [set(cols_a[ip_a[i] : ip_a[i + 1]].tolist()) for i in range(W.NI)]
ip_s, nb_s, _ = C.LISTS["same_artist"]
NBRS = [set(nb_s[ip_s[i] : ip_s[i + 1]].tolist()) for i in range(W.NI)]

# the bound against every slot without artist evidence
U_GAM = ER.bound_of(terms, T_GAM[None])
ok_b, n_out = True, 0
for k in range(0, NE, 6):
    u = int(EU[k])
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, u)
    hist_arts = set().union(*[ARTS[int(h)] for h in items])
    hs = set(int(h) for h in items)
    full = cn.full_catalog(cn.build_user_state(items, weights, W.SPACE), W.SPACE)
    s_ = dense_score(full, terms, T_GAM, W.NI)
    for i in range(1, W.NI):
        if not (ARTS[i] & hist_arts) and not (NBRS[i] & hs):
            n_out += 1
            ok_b &= s_[i] <= U_GAM
check(
    ok_b and n_out > 0,
    f"1: every one of {n_out} slots without artist evidence (10 listeners "
    f"of the chain) scores at most the bound {U_GAM:+.4f}",
)


# 3: the stage


def expected(k, shown, Z=None, T_=None):
    """Every rank code of listener k of the P14 arrays Z by a route written
    here, with the shown reasons shown (G, 10) and the table T_."""
    Z = Z14 if Z is None else Z
    T_ = T_GAM if T_ is None else T_
    u = int(Z["user_id"][k])
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, u)
    hist = {int(h): float(w) for h, w in zip(items, weights)}
    top = [int(t) for t in Z["top_id"][k, 0, :10]]

    def ranks(keep, slots, units=None, audio=False):
        if not keep:
            return [ER.NO_HISTORY] * len(slots)
        h_ = sorted(keep)
        state = cn.build_user_state(
            np.array(h_), np.array([hist[h] for h in h_]), W.SPACE
        )
        if units:
            state = drop_units(state, W.SPACE, units)
        if audio:
            state = no_audio(state)
        s_ = dense_score(cn.full_catalog(state, W.SPACE), terms, T_, W.NI)
        return full_sort_ranks(s_, items, slots)

    def cited(i, g):
        n_ = min(int(Z["ev_count"][k, i, g]), P.EVIDENCE_K)
        return [int(x) for x in Z["ev_id"][k, i, g, :n_]]

    def art_ev(t):
        return {h for h in hist if (ARTS[h] & ARTS[t]) or h in NBRS[t]}

    out = {
        "rank_shown": [0] * 3,
        "rank_all": [0] * 10,
        "rank_kept": [0] * 10,
        "rank_content": [[0] * 10 for _ in G_CONTENT],
        "rank_audio": [0] * 10,
        "rank_list": [0] * 10,
    }
    art = shown[GA]
    for i in range(3):
        if art[i] and Z["ev_count"][k, i, GA] > 0:
            out["rank_shown"][i] = ranks(set(hist) - set(cited(i, GA)), [top[i]])[0]
    sets = {}
    for i in range(10):
        if art[i]:
            sets.setdefault(frozenset(ARTS[top[i]]), []).append(i)
    E_all = set()
    for pos in sets.values():
        E = set().union(*[art_ev(top[i]) for i in pos])
        E_all |= E
        r_all = ranks(set(hist) - E, [top[i] for i in pos])
        r_kept = ranks(E, [top[i] for i in pos])
        for q, i in enumerate(pos):
            out["rank_all"][i], out["rank_kept"][i] = r_all[q], r_kept[q]
    units_all = {}
    for j, g in enumerate(G_CONTENT):
        pos = [i for i in range(10) if shown[g][i]]
        if not pos:
            continue
        src = terms[groups[g][1][0]].source
        U = sorted(set().union(*[set(cited(i, g)) for i in pos]))
        units_all[src] = U
        r_ = ranks(set(hist), [top[i] for i in pos], units={src: U})
        for q, i in enumerate(pos):
            out["rank_content"][j][i] = r_[q]
    aud = bool(shown[G_AUDIO].any())
    if aud:
        pos = [i for i in range(10) if shown[G_AUDIO][i]]
        r_ = ranks(set(hist), [top[i] for i in pos], audio=True)
        for q, i in enumerate(pos):
            out["rank_audio"][i] = r_[q]
    if E_all or units_all or aud:
        out["rank_list"] = ranks(set(hist) - E_all, top, units=units_all, audio=aud)
    return out


ER.CHUNK, ER.BLOCK, ER.PROBE_USERS, ER.CHECK_STRIDE = 64, 7, 3, 5
ES = ER.StageEvidenceRemoval(CHAIN, SCR)
ES._stage = stage
ES.workers = 3
CAP.lines.clear()
rc = execute(STS + [COLD, EVS, ES])
OUT = {k: Path(v) for k, v in ES.outputs().items()}
doc = json.loads(OUT["results"].read_text()) if OUT["results"].exists() else None
z = dict(np.load(OUT["arrays"])) if OUT["arrays"].exists() else {}
real_stage = RN.STAGE_BY_NAME.get("evidence_removal")
check(
    isinstance(real_stage, ER.StageEvidenceRemoval)
    and [s.name for s in RN.SCRIPT_STAGES][-2:]
    == ["beyond_accuracy", "evidence_removal"]
    and real_stage.upstream == ["p14_eval", "p14_cold_gam"]
    and real_stage.guarded,
    "3: the runner holds evidence_removal after beyond_accuracy, guarded, "
    "with p14_eval and p14_cold_gam upstream",
)
check(
    rc == 0
    and doc is not None
    and common.marker_path("evidence_removal").exists()
    and doc["data"]["rule"] == "all"
    and doc["data"]["listeners"] == NE
    and any("probe listener" in ln for ln in CAP.lines)
    and any("3 workers" in ln for ln in CAP.lines)
    and any("lie in the first" in ln for ln in CAP.lines),
    f"3: evidence_removal runs through the runner with 3 workers on all "
    f"{NE} eval listeners, its checks pass and it writes its marker "
    f"({clock()})",
)
CAP.lines.clear()
check(
    execute(STS + [COLD, EVS, ES]) == 0
    and any("skip evidence_removal" in ln for ln in CAP.lines),
    "3: a second runner call skips evidence_removal",
)
KEYS = (
    "rank_shown",
    "rank_all",
    "rank_kept",
    "rank_content",
    "rank_audio",
    "rank_list",
)
bad_k = []
for k in range(NE):
    want = expected(k, Z14["shown"][:, k, :])
    for key in KEYS:
        if np.asarray(want[key]).tolist() != z[key][k].tolist():
            bad_k.append((k, key))
kinds_seen = {key: int((z[key] >= 1).sum()) for key in KEYS}
check(
    not bad_k
    and np.all(z["rank_base"] == np.arange(1, 11))
    and np.array_equal(z["rank_pick"], Z14["rank_gam"]),
    f"3: for all {NE} eval listeners every new rank of every pass equals "
    f"the route written here, and the base ranks equal P14's (tracks "
    f"tested: {kinds_seen}). Differ: {bad_k[:5] or 'none'}",
)

# every candidate reason shown, so that every kind of pass runs
cand = np.asarray(Z14["candidate"], dtype=bool).copy()
for g in G_CONTENT:
    cand[g] &= Z14["ev_count"][:, :, g] > 0
Z_ALL = dict(Z14, shown=cand)
SC_ALL = ct.CatalogScorer(terms, T_GAM[None], W.SPACE, chunk=64)
CTX_ALL = ER.context(
    SC_ALL,
    W.SPACE,
    terms,
    T_GAM[None],
    groups,
    Z_ALL,
    (W.u_indptr, W.u_indices, W.u_data),
    chunk=64,
    stride=5,
)
bad_a, n_kinds = [], {key: 0 for key in KEYS}
bounded = full_n = 0
for k in range(0, NE, 3):
    got = ER.listener(CTX_ALL, k)
    den = ER.listener(CTX_ALL, k, dense=True)
    want = expected(k, cand[:, k, :])
    for key in KEYS:
        n_kinds[key] += int((np.asarray(got[key]) >= 1).sum())
        if np.asarray(want[key]).tolist() != np.asarray(got[key]).tolist():
            bad_a.append((k, key))
    if not all(
        ER._same({key: got[key]}, {key: den[key]})
        for key in got
        if key not in ER.SKIP_DENSE
    ):
        bad_a.append((k, "dense"))
    full_n += int(got["full_scans"])
    bounded += int(got["plans"]) - int(got["full_scans"])
check(
    not bad_a and all(v > 0 for v in n_kinds.values()) and full_n > 0,
    f"3: with every candidate reason shown (20 listeners), every pass "
    f"equals the route written here and the module's dense route (tracks "
    f"tested: {n_kinds}, {bounded} bounded and {full_n} full passes). "
    f"Differ: {bad_a[:5] or 'none'}",
)

# a stronger artist shape, so that the tracks with artist evidence score
# above the bound and the bounded scans run on the chain's world
T_ART = T_GAM.copy()
j_aff = [t.name for t in terms].index("artist_affinity")
o_aff = P.offsets_of(terms)[j_aff]
T_ART[o_aff : o_aff + terms[j_aff].K] += 6.0
KS = np.arange(0, NE, 3)
ctx_ev = {"space": W.SPACE, "groups": groups, "sources": [t.source for t in terms]}
src_of = [terms[js[0]].source for _, js in groups]
like = {
    key: []
    for key in (
        "user_id",
        "picks",
        "top_id",
        "top_score",
        "top_val",
        "top_has",
        "shown",
        "ev_count",
        "ev_id",
        "ev_val",
        "ev_credit",
        "ev_credit_tracks",
        "rank_gam",
    )
}
for k in KS:
    u = int(EU[k])
    items, weights = cd.history_of(W.u_indptr, W.u_indices, W.u_data, u)
    st_ = cn.build_user_state(items, weights, W.SPACE)
    full = cn.full_catalog(st_, W.SPACE)
    s_ = dense_score(full, terms, T_ART, W.NI)
    hs = set(int(h) for h in items)
    order = sorted(
        [i for i in range(1, W.NI) if i not in hs], key=lambda i: (-float(s_[i]), i)
    )
    top = np.array(order[:10])
    e_ = [P._evidence(ctx_ev, st_, int(t)) for t in top]
    sh = np.zeros((len(groups), 10), dtype=bool)
    for g, (n_, js) in enumerate(groups):
        if n_ == "artist":
            for t in ("artist_affinity", "colisten_same_artist"):
                v, h = full[t]
                sh[g] |= h[top] & (np.nan_to_num(v[top]) > 0)
        elif g in G_CONTENT:
            sh[g] = np.array([e[0][g] > 0 for e in e_])
        elif g == G_AUDIO:
            sh[g] = full["audio_fit"][1][top]
    like["user_id"].append(u)
    like["picks"].append(Z14["picks"][k])
    like["top_id"].append(top[None])
    like["top_score"].append(s_[top][None])
    like["top_val"].append(np.stack([full[t.source][0][top] for t in terms], axis=1))
    like["top_has"].append(np.stack([full[t.source][1][top] for t in terms], axis=1))
    like["shown"].append(sh)
    for key, j in (
        ("ev_count", 0),
        ("ev_id", 1),
        ("ev_val", 2),
        ("ev_credit", 3),
        ("ev_credit_tracks", 4),
    ):
        like[key].append(np.stack([np.asarray(e[j]) for e in e_]))
    like["rank_gam"].append(full_sort_ranks(s_, items, Z14["picks"][k]))
Z_LIKE = {key: np.stack([np.asarray(x) for x in v]) for key, v in like.items()}
Z_LIKE["shown"] = np.transpose(Z_LIKE["shown"], (1, 0, 2))
SC_ART = ct.CatalogScorer(terms, T_ART[None], W.SPACE, chunk=64)
CTX_ART = ER.context(
    SC_ART,
    W.SPACE,
    terms,
    T_ART[None],
    groups,
    Z_LIKE,
    (W.u_indptr, W.u_indices, W.u_data),
    chunk=64,
    stride=5,
)
bad_b, n_b = [], {key: 0 for key in KEYS}
bounded = full_n = 0
for j in range(len(KS)):
    got = ER.listener(CTX_ART, j)
    den = ER.listener(CTX_ART, j, dense=True)
    want = expected(j, Z_LIKE["shown"][:, j, :], Z_LIKE, T_ART)
    for key in KEYS:
        n_b[key] += int((np.asarray(got[key]) >= 1).sum())
        if np.asarray(want[key]).tolist() != np.asarray(got[key]).tolist():
            bad_b.append((j, key))
    if not all(
        ER._same({key: got[key]}, {key: den[key]})
        for key in got
        if key not in ER.SKIP_DENSE
    ):
        bad_b.append((j, "dense"))
    full_n += int(got["full_scans"])
    bounded += int(got["plans"]) - int(got["full_scans"])
check(
    not bad_b
    and all(v > 0 for v in n_b.values())
    and bounded > 0
    and full_n > 0
    and CTX_ART["bound"] == ER.bound_of(terms, T_GAM[None]),
    f"3: with the artist shape raised by 6 (P14's inputs built here for "
    f"20 listeners), {bounded} bounded and {full_n} full passes equal the "
    f"route written here and the module's dense route (tracks tested: "
    f"{n_b}). Differ: {bad_b[:5] or 'none'}",
)


# The saved blocks load one at a time
parts = sorted((SCR / ER.WORK_NAME / "pass").glob("part_*.npz"))
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
n_open = len(os.listdir("/proc/self/fd"))
resource.setrlimit(resource.RLIMIT_NOFILE, (n_open + 4, hard))
try:
    arr = ER.load_parts(parts)
    low_ok = True
except OSError:
    low_ok = False
finally:
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
order_ = np.random.default_rng(ER.ORDER_SEED).permutation(NE)
srt_ = np.argsort(order_, kind="stable")
check(
    low_ok
    and len(parts) == 9
    and len(os.listdir("/proc/self/fd")) == n_open
    and all(
        np.array_equal(arr[key][srt_], z[key], equal_nan=True)
        for key in z
        if key in arr
    ),
    f"3: the {len(parts)} saved blocks load one at a time with only 4 free "
    f"file handles, no file stays open, and they give the stored arrays",
)

# The measures by loops written here


def qs(vals):
    v = sorted(vals)
    if not v:
        return {f"p{q}": None for q in ER.QUANTILES}
    return {f"p{q}": v[max(0, math.ceil(q / 100 * len(v)) - 1)] for q in ER.QUANTILES}


def loop_block(r, measure):
    tested = [
        (k, j) for k in range(r.shape[0]) for j in range(r.shape[1]) if r[k, j] >= 1
    ]
    hits = [
        float(r[k, j] > 10 if measure == "leave" else r[k, j] <= 10) for k, j in tested
    ]
    return (
        len(tested),
        len({k for k, _ in tested}),
        sum(hits) / len(hits) if hits else None,
        qs([int(r[k, j]) for k, j in tested]),
    )


ok_m = {}
res = doc["results"]
for name_, key, measure in (
    ("artist_shown", "rank_shown", "leave"),
    ("artist_all", "rank_all", "leave"),
    ("artist_kept", "rank_kept", "stay"),
    ("audio", "rank_audio", "leave"),
    ("list", "rank_list", "leave"),
):
    n_t, n_l, sh, q = loop_block(z[key], measure)
    b = res[name_]
    ok_m[name_] = (
        b["tested"] == n_t
        and b["listeners"] == n_l
        and b["rank"] == q
        and (
            sh is None
            and b["share"]["mean"] is None
            or abs(b["share"]["mean"] - sh) < 1e-12
        )
    )
for j, g in enumerate(G_CONTENT):
    n_t, n_l, sh, q = loop_block(z["rank_content"][:, j, :], "leave")
    b = res["content"][names[g]]
    ok_m[names[g]] = (
        b["tested"] == n_t
        and b["rank"] == q
        and (
            sh is None
            and b["share"]["mean"] is None
            or abs(b["share"]["mean"] - sh) < 1e-12
        )
    )
ran = np.all(z["rank_list"] >= 1, axis=1)
kept_l = [int((z["rank_list"][k] <= 10).sum()) for k in range(NE) if ran[k]]
kl = res["list"]["kept_in_top"]
ok_m["kept"] = (kl is None and len(kept_l) < 2) or (
    kl is not None and abs(kl["mean"] - float(np.mean(kept_l))) < 1e-12
)
ct_ = Z14["ev_credit_tracks"]
r1 = z["rank_all"]
tested_ = r1 >= 1
q1, q2 = np.quantile(ct_[tested_].astype(float), [1 / 3, 2 / 3])
lev_n = [
    int(((ct_ <= q1) & tested_).sum()),
    int(((ct_ > q1) & (ct_ <= q2) & tested_).sum()),
    int(((ct_ > q2) & tested_).sum()),
]
ok_m["tertiles"] = [
    t["tested"] for t in res["artist_all"]["tertiles"]["levels"]
] == lev_n and sum(lev_n) == int(tested_.sum())
check(
    all(ok_m.values()),
    f"3: the tracks tested, listeners, shares, rank percentiles, the "
    f"list's kept tracks and the tertile counts equal loops written here. "
    f"Failing: {[k for k, v in ok_m.items() if not v] or 'none'}",
)

# The read-back, a changed stored unit and a changed P14 rank
saved = OUT["results"].read_bytes()
bad = json.loads(saved)
bad["results"]["artist_all"]["share"]["mean"] = 0.123
OUT["results"].write_text(json.dumps(bad))
refused = raises(ES.check, "results")
OUT["results"].write_bytes(saved)
ES.check()
real_load = ER.load_p14


def bad_load(which):
    def fn(d):
        doc_, zz = real_load(d)
        zz = dict(zz)
        if which == "unit":
            e_ = zz["ev_id"].copy()
            k_, i_ = 4, 0
            e_[k_, i_, GA, 0] = (
                int(e_[k_, i_, GA, 0]) + 1 if e_[k_, i_, GA, 0] >= 0 else 3
            )
            zz["ev_id"] = e_
        else:
            r_ = zz["rank_gam"].copy()
            r_[7, 1] += 1
            zz["rank_gam"] = r_
        return doc_, zz

    return fn


stops = {}
for which, text in (("unit", "stored evidence lists"), ("rank", "the test picks rank")):
    ER.load_p14 = bad_load(which)
    try:
        stops[which] = raises(
            lambda: ER.run(
                stage(),
                CHAIN,
                SCR / f"er_bad_{which}",
                say=quiet,
                n_verify=1,
                inputs=ES.provenance(),
                workers=1,
            ),
            text,
        )
    finally:
        ER.load_p14 = real_load
check(
    refused and all(stops.values()),
    f"3: the read-back refuses a changed result, and a changed stored "
    f"evidence unit and a changed P14 pick rank stop the run ({stops})",
)

rep = OUT["report"].read_text()
cols_ok, blk = True, []
for ln in rep.splitlines() + [""]:
    if ln.startswith("|"):
        blk.append(ln.count("|"))
    elif blk:
        cols_ok &= len(set(blk)) == 1
        blk = []
tabs = {}
for name_ in ER.TAB_FILES:
    with open(CHAIN / ER.FIG_DIR / name_, newline="") as fh:
        tabs[name_] = list(csv.reader(fh))
main = tabs[ER.TAB_FILES[0]]
hdr = main[0]
row2 = next(r for r in main[1:] if r[0] == "2")
tab_ok = (
    len(main) == 1 + 5 + 1 + len(G_CONTENT)
    and len(tabs[ER.TAB_FILES[1]]) == 1 + 6
    and all(len({len(r) for r in t}) == 1 for t in tabs.values())
    and int(row2[hdr.index("tested")]) == res["artist_all"]["tested"]
)
banned = ("2026",)
check(
    ";" not in rep
    and cols_ok
    and tab_ok
    and all(
        s in rep
        for s in (
            "## Counterfactuals",
            "## By the number of history tracks",
            "## The whole list",
            "## Concept values after the removal",
        )
    )
    and not any(b in rep for b in banned)
    and not any(b in Path(ER.__file__).read_text() for b in banned),
    "3: the report has every section, tables with even rows, no "
    "semicolon, no dates or development history, and the two CSV tables "
    "read back with the JSON's numbers",
)

# 4: one worker against three
first = {k: Path(v).read_bytes() for k, v in OUT.items()}
real_listener = ER.listener
calls = {"n": 0}


def stopping(c, k, dense=False):
    calls["n"] += 1
    if calls["n"] == ER.PROBE_USERS + 9:
        raise KeyboardInterrupt
    return real_listener(c, k, dense)


ER.listener = stopping
stopped = False
try:
    ER.run(
        stage(),
        CHAIN,
        SCR / "er_resume",
        say=quiet,
        n_verify=4,
        inputs=ES.provenance(),
        workers=1,
    )
except KeyboardInterrupt:
    stopped = True
finally:
    ER.listener = real_listener
CAP.lines.clear()
ER.run(
    stage(),
    CHAIN,
    SCR / "er_resume",
    say=common.log.info,
    n_verify=4,
    inputs=ES.provenance(),
    workers=1,
)
again = {k: Path(v).read_bytes() for k, v in OUT.items()}
differ = [k for k in first if first[k] != again[k]]
check(
    stopped and not differ and any("resume, 1 of 9 blocks" in ln for ln in CAP.lines),
    f"4: a run stopped inside the pass and resumed with 1 worker writes "
    f"outputs byte-identical to the run with 3 workers ({sorted(first)}). "
    f"Differ: {differ or 'none'}",
)
ES.check()

# 5: the rule of the listeners
real = (ER.SAMPLE_SIZE, ER.HOURS_LIMIT)
try:
    ER.SAMPLE_SIZE, ER.HOURS_LIMIT = 20, 0.0
    CAP.lines.clear()
    d5 = ER.run(
        stage(),
        CHAIN,
        SCR / "er_sample",
        say=common.log.info,
        n_verify=2,
        inputs=ES.provenance(),
        workers=2,
    )
    z5 = dict(np.load(OUT["arrays"]))
    want5 = np.sort(np.random.default_rng(ER.ORDER_SEED).permutation(NE)[:20])
    ok5 = (
        d5["data"]["rule"] == "sample"
        and d5["data"]["listeners"] == 20
        and np.array_equal(z5["p14_index"], want5)
        and any("would take about" in ln for ln in CAP.lines)
    )
    ES.check()
    same5 = all(np.array_equal(z5[key], z[key][want5]) for key in KEYS)
    ER.HOURS_LIMIT = 1e9
    d6 = ER.run(
        stage(),
        CHAIN,
        SCR / "er_all",
        say=quiet,
        n_verify=1,
        inputs=ES.provenance(),
        workers=2,
    )
    z6 = dict(np.load(OUT["arrays"]))
    ok6 = (
        d6["data"]["rule"] == "all"
        and d6["data"]["listeners"] == NE
        and all(np.array_equal(z6[key], z[key]) for key in KEYS)
    )
    ES.check()
finally:
    ER.SAMPLE_SIZE, ER.HOURS_LIMIT = real
check(
    ok5 and same5 and ok6,
    "5: with a sample of 20 and a limit of 0 hours the stage keeps the "
    "first 20 listeners of the random order (their ranks equal the full "
    "run's), and with a large limit every listener runs",
)

print(f"\n{len(failures)} failures, {clock()}")
if os.environ.get("KEEP_TMP"):
    print(f"kept {TMP}")
else:
    shutil.rmtree(TMP, ignore_errors=True)
sys.exit(1 if failures else 0)
