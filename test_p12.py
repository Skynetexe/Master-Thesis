"""Checks for pipeline/concept_data.py and pipeline/stage_p12.py on
synthetic data. Every path points into a temporary folder, and the
DuckDB loaders are replaced by synthetic stand-ins, so nothing touches
the real model folder.

1. The builders: compact(), tier_codes(), audio_array(),
   popularity_array() and history_of(). load_sources() turns the files
   and the query results into the same Sources as a direct build.
2. Reference, the second implementation, equals concepts.pairs()
   through user_rows() for every row of every user, the phantom slot
   included. A term listed twice counts once on both sides. The
   co-listening rows stay equal after ItemSide sorts the shared arrays.
3. The stage: run() writes, for every row, what user_rows() gives for
   that user alone. verify() passes and records the gaps it measured.
4. The worker count and the block count do not change a single byte.
5. Resume: a stopped run continues after its last saved block, a
   finished run does no work, a damaged block is computed again, and a
   change of the upstream outputs or the parameters starts again.
6. verify() fails for each planted fault.
7. The report helpers equal brute force: the pair states and both AUC
   forms, the rank AUC, Spearman's rho and the row layout of the
   negatives.
8. report() writes every section, the explanation coverage, the named
   examples with a contrast negative, and the gaps from verify().

Usage, from the repository root:
    python3 test_p12.py
"""

import copy
import json
import logging
import math
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import stage_p12
from pipeline.common import VerificationError

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def same_bytes(a, b):
    """Equal dtype, shape and bytes. NaN is never equal to NaN, so the
    arrays are compared byte by byte."""
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def raises(fn, text):
    """True when fn() raises the pipeline's VerificationError with text
    in its message. Any other exception is a bug, not a caught fault."""
    try:
        fn()
    except VerificationError as exc:
        return text in str(exc)
    except Exception as exc:  # reported, then False
        print(f"      unexpected {type(exc).__name__}: {exc}")
        return False
    return False


rng = np.random.default_rng(12)
N = 400  # item slots, slot 0 is the phantom
D = len(cn.AUDIO_DIMS)

# 1: builders
ind, ids = np.array([0, 2, 2, 5]), np.array([900, 30, 30, 7, 900])
c_ind, c_cols, n_cols, id_of = cd.compact(ind, ids)
check(
    n_cols == 3
    and id_of.tolist() == [7, 30, 900]
    and id_of[c_cols].tolist() == ids.tolist()
    and c_ind.tolist() == ind.tolist(),
    "compact() maps ids to sorted columns and back",
)
codes = cd.tier_codes([3, 1], ["url", "isrc"], 5)
unmatched = cn.MATCH_TIERS.index("unmatched")
check(
    codes.tolist() == [unmatched, 0, unmatched, 2, unmatched],
    "tier_codes() codes the tiers and leaves the other slots unmatched",
)
try:
    cd.tier_codes([1], ["isrc_typo"], 3)
    check(False, "tier_codes() accepts an unknown tier name")
except ValueError:
    check(True, "tier_codes() rejects an unknown tier name")
cols = {
    name: np.ma.masked_array(np.arange(2, dtype=float) + d, mask=[d == 1, False])
    for d, name in enumerate(cn.AUDIO_DIMS)
}
aud = cd.audio_array([2, 4], cols, 5)
check(
    np.isnan(aud[[0, 1, 3]]).all()
    and aud[2, 0] == 0.0
    and np.isnan(aud[2, 1])
    and aud[4, 1] == 2.0,
    "audio_array() turns a masked (NULL) value into NaN",
)
pa = cd.popularity_array([1, 3], [0, 9], 4)
check(
    pa.tolist() == [0.0, 0.0, 0.0, math.log(10.0)],
    "popularity_array() holds log1p(listens) and 0 elsewhere",
)
hi, hw = cd.history_of(
    np.array([0, 2, 5]),
    np.array([4, 9, 1, 2, 3]),
    np.array([693, 1000, 5, 6, 7], dtype=np.uint16),
    0,
)
check(
    hi.tolist() == [4, 9] and hw.tolist() == [0.693, 1.0] and hi.dtype == np.int64,
    "history_of() reads one user's row and divides the data by 1000",
)

# Synthetic Sources


def rows_csr(n_cols, p_empty, max_per_row):
    rows = [[]]
    for _ in range(1, N):
        if rng.random() < p_empty:
            rows.append([])
            continue
        k = int(rng.integers(1, max_per_row + 1))
        rows.append(sorted(rng.choice(n_cols, size=k, replace=False).tolist()))
    return rows


def pack(rows, dtype=np.int32):
    indptr = np.concatenate(([0], np.cumsum([len(r) for r in rows])))
    return indptr.astype(np.int64), np.array([x for r in rows for x in r], dtype=dtype)


term_rows = {
    "tags_direct": rows_csr(30, 0.45, 3),
    "tags_artist": rows_csr(30, 0.2, 6),
    "genres": rows_csr(12, 0.35, 2),
}
DUP_ITEM = next(i for i in range(1, N) if len(term_rows["tags_artist"][i]))
term_rows["tags_artist"][DUP_ITEM] = (
    term_rows["tags_artist"][DUP_ITEM] + term_rows["tags_artist"][DUP_ITEM][:1]
)
terms = {}
for name, rows in term_rows.items():
    ip, ix = pack(rows)
    terms[name] = (ip, ix, 30 if name != "genres" else 12)
term_names = {name: [f"{name}-{t}" for t in range(terms[name][2])] for name in terms}

credits, credit_ids = {}, {}
for name, n_ent, p_empty in (("artists", 60, 0.03), ("works", 40, 0.7)):
    rows = rows_csr(n_ent, p_empty, 2)
    ip, raw = pack([[100_000 + 7 * a for a in r] for r in rows], np.int64)
    c = cd.compact(ip, raw)
    credits[name], credit_ids[name] = c[:3], c[3]

colisten = {}
for kind, n_nb in (("same_artist", 6), ("other_artists", 3)):
    rows_n, rows_c = [[]], [[]]
    for i in range(1, N):
        if rng.random() < 0.3:
            rows_n.append([])
            rows_c.append([])
            continue
        k = int(rng.integers(1, n_nb + 1))
        nb = rng.choice(np.setdiff1d(np.arange(1, N), [i]), size=k, replace=False)
        cs = rng.uniform(0.01, 1.0, size=k)
        order = np.argsort(-cs)  # P9b stores rows by cosine
        rows_n.append(nb[order].tolist())
        rows_c.append(cs[order].tolist())
    ip, nb = pack(rows_n)
    colisten[kind] = (ip, nb, np.array([x for r in rows_c for x in r]))

audio = rng.normal(size=(N, D))
audio[0] = np.nan
NO_AUDIO = rng.choice(np.arange(1, N), size=60, replace=False)
audio[NO_AUDIO] = np.nan
audio[
    rng.choice(np.arange(1, N), size=40, replace=False), cn.AUDIO_DIMS.index("z_tempo")
] = np.nan
listens = rng.integers(0, 3000, size=N)
listens[rng.choice(np.arange(1, N), size=20, replace=False)] = 0
popularity = np.log1p(listens.astype(np.float64))
tier = rng.integers(0, len(cn.MATCH_TIERS), size=N)

SRC = cd.Sources(
    N, terms, credits, credit_ids, colisten, audio, popularity, tier, term_names, {}
)
SRC_PRISTINE = copy.deepcopy(SRC)

# Users. Slot 0 of the user CSR stays empty. Two users have histories
# without direct tags or without audio.
NU = 70
untagged = [i for i in range(1, N) if not term_rows["tags_direct"][i]]
hist = [np.zeros(0, dtype=np.int32)]
for u in range(1, NU):
    if u == 5:
        h = rng.choice(untagged, size=12, replace=False)
    elif u == 6:
        h = rng.choice(NO_AUDIO, size=8, replace=False)
    else:
        h = rng.choice(
            np.arange(1, N), size=int(rng.choice([1, 5, 30, 120])), replace=False
        )
    hist.append(np.sort(h).astype(np.int32))
hist[7] = np.sort(np.union1d(hist[7], [DUP_ITEM])).astype(np.int32)
u_indptr = np.concatenate(([0], np.cumsum([len(h) for h in hist])))
u_indices = np.concatenate(hist).astype(np.int32)
u_data = np.round(1000 * np.log1p(rng.integers(1, 400, size=len(u_indices)))).astype(
    np.uint16
)

# 1b: load_sources() glue
# The files go to a temporary model folder, and a stand-in answers each
# DuckDB query, so the check covers the file keys, the column names that
# the code reads, compact() of the credits and the tier codes.
glue_dir = Path(tempfile.mkdtemp(prefix="test_p12_glue_"))
real_model_dir, real_fetch = cd.MODEL_DIR, cd._fetch
real_load_credits = cd.ar.load_credits
real_features_table = cd.features_table
cd.MODEL_DIR = glue_dir
cd.features_table = lambda name: f"{name}.parquet"  # the stand-in answers the query
for name, fname in cd.TERM_FILES.items():
    ip, ix, nc = SRC_PRISTINE.terms[name]
    np.savez_compressed(
        glue_dir / fname, indices=ix, indptr=ip, shape=np.array([N, nc], dtype=np.int64)
    )
for kind, fname in cd.COLISTEN_FILES.items():
    ip, nb, cs = SRC_PRISTINE.colisten[kind]
    np.savez(
        glue_dir / fname,
        indptr=ip,
        neighbor=nb,
        cosine=cs,
        co=np.ones(len(nb), dtype=np.int32),
    )
a_ip, a_cols, _ = SRC_PRISTINE.credits["artists"]
a_raw = SRC_PRISTINE.credit_ids["artists"][a_cols]
w_ip, w_cols, _ = SRC_PRISTINE.credits["works"]
w_items = np.repeat(np.arange(N), np.diff(w_ip))
w_raw = SRC_PRISTINE.credit_ids["works"][w_cols]
has_audio = ~np.isnan(SRC_PRISTINE.audio).all(axis=1)
audio_ids = np.flatnonzero(has_audio)
listen_ids = np.flatnonzero(listens[1:] > 0) + 1  # no phantom row
queries = []


def glue_fetch(sql):
    queries.append(sql)
    if "vocab_tags" in sql:
        names = term_names["tags_direct"]
        return {"n": np.array(names, dtype=object), "i": np.arange(len(names))}
    if "vocab_genres" in sql:
        names = term_names["genres"]
        return {"n": np.array(names, dtype=object), "i": np.arange(len(names))}
    if "item_works" in sql:
        return {
            "item_id": np.concatenate((w_items, w_items[:3])),
            "work_id": np.concatenate((w_raw, w_raw[:3])),
        }
    if "item_audio_std" in sql:
        out = {"item_id": audio_ids}
        for d, dim in enumerate(cn.AUDIO_DIMS):
            x = SRC_PRISTINE.audio[audio_ids, d]
            out[dim] = np.ma.masked_array(np.nan_to_num(x, nan=-7.0), mask=np.isnan(x))
        return out
    if "stats_item_train" in sql:
        return {"item_id": listen_ids, "listens": listens[listen_ids]}
    if "item_covariates" in sql:
        return {
            "item_id": np.arange(1, N),
            "tier": np.array([cn.MATCH_TIERS[t] for t in tier[1:]], dtype=object),
        }
    raise AssertionError(f"unexpected query {sql}")


cd._fetch = glue_fetch
cd.ar.load_credits = lambda n: (a_ip, a_raw, {"credit_rows": len(a_raw)})
try:
    loaded = cd.load_sources(N, lambda msg: None)
    glue_ok = (
        all(
            np.array_equal(loaded.terms[k][0], SRC_PRISTINE.terms[k][0])
            and np.array_equal(loaded.terms[k][1], SRC_PRISTINE.terms[k][1])
            and loaded.terms[k][2] == SRC_PRISTINE.terms[k][2]
            for k in SRC_PRISTINE.terms
        )
        and loaded.term_names["tags_artist"] == term_names["tags_direct"]
        and all(
            np.array_equal(loaded.credits[k][1], SRC_PRISTINE.credits[k][1])
            and np.array_equal(loaded.credit_ids[k], SRC_PRISTINE.credit_ids[k])
            for k in ("artists", "works")
        )
        and all(
            np.array_equal(loaded.colisten[k][j], SRC_PRISTINE.colisten[k][j])
            for k in cd.COLISTEN_FILES
            for j in range(3)
        )
        and np.array_equal(loaded.audio, SRC_PRISTINE.audio, equal_nan=True)
        and np.array_equal(loaded.popularity[1:], popularity[1:])
        and loaded.popularity[0] == 0.0
        and np.array_equal(loaded.tier[1:], tier[1:])
        and loaded.tier[0] == cn.MATCH_TIERS.index("unmatched")
    )
    check(
        glue_ok and len(queries) == 7 and all(" AS " in q for q in queries),
        "load_sources() reads every file and query into the same Sources "
        "as the direct build, with NULL audio as NaN and a work listed "
        "twice counted once",
    )
    cd._fetch = lambda sql: (
        {"n": np.array(["a", "b"], dtype=object), "i": np.array([0, 2])}
        if "vocab" in sql
        else glue_fetch(sql)
    )
    try:
        cd.load_sources(N, lambda msg: None)
        check(False, "load_sources() accepts a vocabulary with a gap")
    except ValueError as exc:
        check(
            "does not number its terms" in str(exc),
            "load_sources() rejects a vocabulary that does not number "
            "its terms 0..n-1",
        )
    cd._fetch = glue_fetch
    ip, nb, cs = SRC_PRISTINE.colisten["other_artists"]
    np.savez(
        glue_dir / cd.COLISTEN_FILES["other_artists"],
        indptr=ip[:-1],
        neighbor=nb,
        cosine=cs,
    )
    try:
        cd.load_sources(N, lambda msg: None)
        check(False, "load_sources() accepts a co-listening file that " "misses a slot")
    except ValueError as exc:
        check(
            "does not cover the item slots" in str(exc),
            "load_sources() rejects a co-listening file that misses a " "slot",
        )
    try:
        cd.load_sources(N + 1, lambda msg: None)
        check(False, "load_sources() accepts term files of another " "catalog size")
    except ValueError as exc:
        check(
            "the catalog has" in str(exc),
            "load_sources() rejects term files of another catalog size",
        )
finally:
    cd.MODEL_DIR, cd._fetch = real_model_dir, real_fetch
    cd.ar.load_credits = real_load_credits
    cd.features_table = real_features_table

# 2: Reference vs concepts
space_ref = cd.item_side(copy.deepcopy(SRC_PRISTINE), transposed=False)
ref = cd.Reference(SRC_PRISTINE)
worst, compared, dup_seen = 0.0, 0, False
same_all = True
for u in range(1, NU):
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, u)
    cand = np.arange(N)
    v, h, t = cd.user_rows(space_ref, items_h, w_h, cand)
    ru = ref.user(items_h, w_h)
    for i in range(N):
        want = ref.values(ru, i)
        for j, c in enumerate(cn.CONTINUOUS):
            wv, wh = want[c]
            ok = bool(h[i, j]) == wh and (
                not wh or abs(v[i, j] - wv) <= 1e-12 * max(1.0, abs(wv))
            )
            if wh:
                worst = max(worst, abs(v[i, j] - wv) / max(1.0, abs(wv)))
            same_all &= ok
            compared += 1
        same_all &= int(t[i]) == want["match_tier"]
    dup_seen |= DUP_ITEM in set(items_h.tolist())
check(
    same_all and dup_seen,
    f"Reference equals concepts.pairs() for {NU - 1} users x {N} slots x "
    f"{len(cn.CONTINUOUS)} concepts ({compared:,} values, largest "
    f"relative gap {worst:.1e}), the duplicated term included",
)
v0, h0, _ = cd.user_rows(space_ref, *cd.history_of(u_indptr, u_indices, u_data, 3), [0])
check(
    not h0.any() and np.isnan(v0).all(), "the phantom slot is missing for every concept"
)


def unsorted_rows(indptr, nbr):
    return sum(
        bool((np.diff(nbr[indptr[i] : indptr[i + 1]]) < 0).any()) for i in range(1, N)
    )


shared = copy.deepcopy(SRC_PRISTINE)
before = unsorted_rows(*shared.colisten["same_artist"][:2])
_ = cd.item_side(shared, transposed=False)
after = unsorted_rows(*shared.colisten["same_artist"][:2])
ref_after = cd.Reference(shared)
same_after = True
for u in (1, 7, 30):
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, u)
    a, b = ref.user(items_h, w_h), ref_after.user(items_h, w_h)
    for i in range(1, N):
        x, y = ref.values(a, i), ref_after.values(b, i)
        for c in cn.COLISTEN_CONCEPTS:
            same_after &= x[c][1] == y[c][1] and (
                not x[c][1] or abs(x[c][0] - y[c][0]) <= 1e-12 * max(1.0, abs(x[c][0]))
            )
check(
    before > 0 and after == 0 and same_after,
    f"ItemSide sorts the shared co-listening arrays in place ({before} "
    f"rows out of id order before, {after} after), and the Reference "
    f"gives the same values before and after",
)
bin_space = cd.item_side(copy.deepcopy(SRC_PRISTINE), transposed=False)
m = bin_space.term_mats["tags_artist"]
row = m[DUP_ITEM]
check(
    float(row.data.max()) == 1.0
    and row.nnz == len(set(term_rows["tags_artist"][DUP_ITEM])),
    "item_side() counts a term listed twice once",
)

# The pairs file
K = 4
p_users = np.array([u for u in range(1, NU) if u not in (11, 12)])
rows_item, rows_kind, rows_pick, rows_draw = [], [], [], []
p_indptr = [0]
p_pop = listens[1:] ** 0.75 + 1e-9
p_pop = p_pop / p_pop.sum()
for u in p_users:
    allowed = np.setdiff1d(np.arange(1, N), hist[u])
    n_pos = int(rng.integers(1, 3))
    pos = np.sort(rng.choice(allowed, size=n_pos, replace=False))
    uni = rng.choice(allowed, size=K)
    pp = rng.choice(np.arange(1, N), size=K * 3, p=p_pop)
    pp = pp[np.isin(pp, allowed)][:K]
    while len(pp) < K:
        pp = np.append(pp, rng.choice(allowed))
    rows_item += pos.tolist() + uni.tolist() + pp.tolist()
    rows_kind += [0] * n_pos + [1] * K + [2] * K
    rows_pick += [1] * n_pos + [0] * (2 * K)
    rows_draw += [0] * n_pos + list(range(K)) * 2
    p_indptr.append(len(rows_item))
split = (rng.random(len(p_users)) < 0.2).astype(np.int8)
split[0] = 1
PAIRS = dict(
    params=np.array(json.dumps({"version": "fusion-pairs-v1", "neg_per_kind": K})),
    user_id=p_users.astype(np.int32),
    split=split,
    indptr=np.array(p_indptr, dtype=np.int64),
    item=np.array(rows_item, dtype=np.int32),
    kind=np.array(rows_kind, dtype=np.int8),
    pick=np.array(rows_pick, dtype=np.int8),
    draw=np.array(rows_draw, dtype=np.int16),
    n_allowed=np.full(len(p_users), N - 10, dtype=np.int64),
    pop_allowed=np.ones(len(p_users)),
    consumed=np.ones((len(p_users), 2), dtype=np.int64),
)
R = len(rows_item)

tmp = Path(tempfile.mkdtemp(prefix="test_p12_"))
stage_p12.MODEL_DIR = tmp / "model"
stage_p12.SCRATCH_DIR = tmp / "scratch"
(tmp / "model" / "reports").mkdir(parents=True)
np.savez(tmp / "model" / "fusion_pairs.npz", **PAIRS)
stage_p12.WORKERS, stage_p12.TASKS_PER_WORKER = 1, 6
stage_p12.VERIFY_USERS = 10**6
stage_p12.REPORT_EVIDENCE_USERS = 10**6
stage_p12.REPORT_CORR_ROWS = 10**6
UP = {"p25_train": {"stats_item_train": "a" * 64}}
lines = []


class Grab(logging.Handler):
    def emit(self, record):
        lines.append(record.getMessage())


logging.getLogger("pipeline").addHandler(Grab())
logging.getLogger("pipeline").setLevel(logging.INFO)


def new_stage():
    s = stage_p12.StageP12()
    s._id_slots = lambda: (N, 1)
    s._sources = lambda n: SRC
    s._csr = lambda: (u_indptr, u_indices, u_data)
    s._upstream_outputs = lambda: copy.deepcopy(UP)
    s._track_names = lambda ids: {int(i): f"Artist {i} - Song {i}" for i in ids}
    s._entity_names = lambda kind, ids: {int(i): f"{kind} name {i}" for i in ids}
    return s


stage = new_stage()
out = stage.outputs()
stage.run({}, [])
z = np.load(out["features"])
first = {k: z[k].copy() for k in z.files}

# 3: run, verify
want_v = np.empty((R, len(cn.CONTINUOUS)), dtype=np.float32)
want_h = np.empty((R, len(cn.CONTINUOUS)), dtype=bool)
want_t = np.empty(R, dtype=np.int8)
for j, u in enumerate(p_users):
    a, b = p_indptr[j], p_indptr[j + 1]
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, int(u))
    v, h, t = cd.user_rows(space_ref, items_h, w_h, PAIRS["item"][a:b])
    want_v[a:b], want_h[a:b], want_t[a:b] = v, h, t
check(
    np.array_equal(first["values"], want_v, equal_nan=True)
    and np.array_equal(first["has"], want_h)
    and np.array_equal(first["match_tier"], want_t)
    and first["values"].dtype == np.float32
    and json.loads(str(first["params"])) == stage_p12._params(),
    f"run() writes user_rows() of each user alone for all {R} rows",
)
check(
    not raises(lambda: stage.verify({}), "")
    and stage._verified
    and stage._verified["reference_rows"] == 4 * len(p_users)
    and stage._verified["evidence_lists"] > 0
    and stage._verified["reference_gap"] < 1e-6
    and stage._verified["evidence_gap"] < 1e-12,
    f"verify() passes and records its gaps ({stage._verified})",
)
has_all = first["has"]
check(
    bool((~has_all[:, :-1]).any(axis=0).all())
    and bool(
        (has_all & (np.nan_to_num(first["values"]) == 0))[:, :7].any(axis=0).sum() >= 5
    ),
    "every concept but popularity has missing rows, and most have "
    "rows with data and a value of 0",
)
u5 = int(np.flatnonzero(p_users == 5)[0])
s5, e5 = p_indptr[u5], p_indptr[u5 + 1]
check(
    not has_all[s5:e5, 0].any()
    and bool(space_ref.has_terms["tags_direct"][PAIRS["item"][s5:e5]].any()),
    "a history without direct tags leaves tag_affinity missing, even "
    "for tagged candidates",
)

# 4: workers and blocks
stage_p12.WORKERS, stage_p12.TASKS_PER_WORKER = 3, 4
new_stage().run({}, [])
z2 = np.load(out["features"])
check(
    all(same_bytes(first[k], z2[k]) for k in first) and set(first) == set(z2.files),
    "three workers and 12 blocks write the same arrays as one worker",
)
stage_p12.WORKERS, stage_p12.TASKS_PER_WORKER = 1, 6

# 5: resume
real_block = stage_p12._block
calls = []


def counting(task):
    calls.append(task)
    return real_block(task)


def stopping_after(n):
    def fn(task):
        if len(calls) >= n:
            raise KeyboardInterrupt
        calls.append(task)
        return real_block(task)

    return fn


def same_as_first():
    zz = np.load(out["features"])
    return set(first) == set(zz.files) and all(
        same_bytes(first[k], zz[k]) for k in first
    )


stage_p12._block = counting
new_stage().run({}, [])
check(
    len(calls) == 6 and same_as_first(),
    "a change of the block layout recomputes all 6 blocks",
)
calls.clear()
lines.clear()
new_stage().run({}, [])
check(
    calls == [] and any("every block is done" in x for x in lines),
    "a finished run with the same key does no work",
)

UP["p25_train"]["stats_item_train"] = "b" * 64
calls.clear()
stage_p12._block = stopping_after(2)
try:
    new_stage().run({}, [])
    stopped = False
except KeyboardInterrupt:
    stopped = True
work = tmp / "scratch" / "p12"
state = json.loads((work / "state.json").read_text())
check(
    stopped
    and sorted(state["parts"]) == ["0", "1"]
    and len(list(work.glob("part_*.npz"))) == 2,
    "a run stopped after two blocks keeps exactly those two",
)
calls.clear()
lines.clear()
stage_p12._block = counting
new_stage().run({}, [])
check(
    len(calls) == 4
    and same_as_first()
    and any("blocks resume, 2 of 6 done" in x for x in lines)
    and not list(work.glob("part_*.npz")),
    "the next run computes the 4 missing blocks, writes the same "
    "arrays and removes the parts",
)

UP["p25_train"]["stats_item_train"] = "c" * 64
calls.clear()
stage_p12._block = stopping_after(3)
try:
    new_stage().run({}, [])
except KeyboardInterrupt:
    pass
np.savez(
    work / "part_00001.npz",
    values=np.zeros((1, 9), np.float32),
    has=np.zeros((1, 9), bool),
    tier=np.zeros(1, np.int8),
)
calls.clear()
stage_p12._block = counting
new_stage().run({}, [])
check(
    len(calls) == 4 and same_as_first(),
    "a changed upstream output starts again, and a damaged saved block "
    "is computed again (3 saved, 1 damaged, 4 computed)",
)

real_sha = stage_p12.file_sha256


def other_code_sha(path):
    if Path(path).name == "concepts.py":
        return "0" * 64
    return real_sha(path)


calls.clear()
stage_p12.file_sha256 = other_code_sha
new_stage().run({}, [])
stage_p12.file_sha256 = real_sha
check(
    len(calls) == 6 and same_as_first(),
    "a change in the code of concepts.py recomputes every block",
)

calls.clear()
stage_p12.VERSION = "pair-features-test"
st = new_stage()
st.run({}, [])
check(
    len(calls) == 6 and not raises(lambda: st.verify({}), ""),
    "a parameter change recomputes every block",
)
stage_p12.VERSION = "pair-features-v1"
calls.clear()
new_stage().run({}, [])
check(len(calls) == 6 and same_as_first(), "the old parameters give the old file again")
stage_p12._block = real_block

inputs = new_stage().current_inputs([])
check(
    set(inputs)
    == (
        {f"marker:{s}" for s in stage_p12.StageP12.upstream}
        | {
            "concepts_module",
            "concept_data_module",
            "artists_module",
            "item_artists",
            "item_works",
        }
    )
    and all(
        len(inputs[k]) == 64
        for k in ("concepts_module", "concept_data_module", "artists_module")
    ),
    "current_inputs() names every upstream marker, the three modules "
    "and the two feature tables",
)

# 6: planted faults
good = {k: v.copy() for k, v in first.items()}
col = {c: j for j, c in enumerate(cn.CONTINUOUS)}
row_user = np.repeat(np.arange(len(p_users)), np.diff(p_indptr))


def with_damage(fn, text):
    dd = {k: v.copy() for k, v in good.items()}
    fn(dd)
    np.savez(out["features"], **dd)
    result = raises(lambda: new_stage().verify({}), text)
    np.savez(out["features"], **good)
    return result


def first_row(mask):
    return int(np.flatnonzero(mask)[0])


def set_cell(dd, r, c, value, has):
    dd["values"][r, col[c]] = value
    dd["has"][r, col[c]] = has


def flip_term(dd):
    j = col["tag_affinity"]
    counts = np.bincount(row_user[good["has"][:, j]], minlength=len(p_users))
    user = int(np.flatnonzero(counts >= 2)[0])
    r = first_row((row_user == user) & good["has"][:, j])
    set_cell(dd, r, "tag_affinity", np.nan, False)


def scale_user(c, factor, add=0.0):
    def fn(dd):
        j = col[c]
        user = int(row_user[first_row(good["has"][:, j])])
        m = (row_user == user) & good["has"][:, j]
        dd["values"][m, j] = dd["values"][m, j] * factor + add

    return fn


def swap_rows(dd):
    j = int(np.flatnonzero(np.diff(p_indptr) > 0)[3])
    a = int(p_indptr[j])
    b = (
        a
        + 1
        + int(
            np.flatnonzero(
                good["values"][a + 1 : p_indptr[j + 1], col["popularity"]]
                != good["values"][a, col["popularity"]]
            )[0]
        )
    )
    for key in ("values", "has", "match_tier"):
        dd[key][[a, b]] = dd[key][[b, a]]


def set_params(dd):
    p = json.loads(str(dd["params"]))
    p["sigma_floor"] = 0.5
    dd["params"] = np.array(json.dumps(p))


def drop_last(dd):
    for key in ("values", "has", "match_tier"):
        dd[key] = dd[key][:-1]


no_audio_row = first_row(~bin_space.audio_mask[PAIRS["item"]].any(axis=1))
no_other_row = first_row(~good["has"][:, col["colisten_other_artists"]])
artist_row = first_row(good["has"][:, col["artist_affinity"]])
pop_j = col["popularity"]
faults = [
    (
        lambda dd: dd["has"].__setitem__((artist_row, col["artist_affinity"]), False),
        "NaN does not mark exactly",
        "a missing flag without NaN",
    ),
    (
        lambda dd: set_cell(dd, artist_row, "artist_affinity", np.nan, False),
        "disagrees with the credits",
        "an artist credit marked missing",
    ),
    (
        lambda dd: set_cell(dd, no_other_row, "colisten_other_artists", 0.1, True),
        "co-listening lists",
        "a co-listening value for a track without a " "list",
    ),
    (
        flip_term,
        "item's terms or with the user's other rows",
        "a term state that disagrees with the user's other rows",
    ),
    (
        lambda dd: set_cell(dd, no_audio_row, "audio_fit", -1.0, True),
        "item without audio",
        "audio_fit for a track without audio",
    ),
    (
        lambda dd: dd["values"].__setitem__(
            (5, pop_j),
            np.nextafter(dd["values"][5, pop_j], np.float32(np.inf), dtype=np.float32),
        ),
        "popularity differs",
        "popularity changed by one float32 step",
    ),
    (
        lambda dd: dd["match_tier"].__setitem__(
            4, (int(dd["match_tier"][4]) + 1) % len(cn.MATCH_TIERS)
        ),
        "match_tier differs",
        "a changed match tier",
    ),
    (
        lambda dd: set_cell(dd, artist_row, "artist_affinity", 1.5, True),
        "falls outside",
        "an artist share above 1",
    ),
    (
        scale_user("audio_fit", 1.001),
        "the reference gives",
        "one user's audio_fit scaled by 1.001",
    ),
    (
        scale_user("tag_affinity", 1 - 1e-4),
        "the reference gives",
        "one user's tag_affinity scaled by 0.9999",
    ),
    (
        scale_user("colisten_same_artist", 1.0, 1e-4),
        "the reference gives",
        "one user's colisten_same_artist raised by 1e-4",
    ),
    (swap_rows, "", "two rows of a user swapped"),
    (set_params, "was built with", "other stored parameters"),
    (drop_last, "shape or type", "a missing last row"),
    (
        lambda dd: dd.__setitem__("values", dd["values"].astype(np.float64)),
        "shape or type",
        "values stored as float64",
    ),
]
for fn, text, label in faults:
    check(with_damage(fn, text), f"verify() catches {label}")

pairs_path = tmp / "model" / "fusion_pairs.npz"
bad = {k: v.copy() for k, v in PAIRS.items()}
bad["item"][first_row(bad["kind"] == 1)] = int(
    np.setdiff1d(np.arange(1, N), bad["item"][:1])[0]
)
np.savez(pairs_path, **bad)
check(
    raises(lambda: new_stage().verify({}), "other fusion pairs"),
    "verify() catches features built from other fusion pairs",
)
np.savez(pairs_path, **PAIRS)

real_evidence = cn.evidence


def skewed(state, space, item):
    e = real_evidence(state, space, item)
    for c in ("artist_affinity", "colisten_same_artist"):
        if e[c]["has_data"] and e[c]["terms"]:
            e[c]["terms"][0]["term"] += 1e-6
    return e


cn.evidence = skewed
check(
    raises(lambda: new_stage().verify({}), "evidence sums to"),
    "verify() catches an evidence list that does not sum to its value",
)
cn.evidence = real_evidence
check(
    not raises(lambda: new_stage().verify({}), ""),
    "verify() passes again after every fault is removed",
)

# 7: report helpers
ok_ps, ok_ml = True, True
for trial in range(30):
    n_rows = 60
    v = np.round(rng.normal(size=n_rows), 1)
    h = rng.random(n_rows) < 0.7
    v[~h] = np.nan
    P, KK = int(rng.integers(1, 8)), int(rng.integers(1, 5))
    pr = rng.integers(0, n_rows, size=P)
    nr = rng.integers(0, n_rows, size=(P, KK))
    s = stage_p12.pair_states(v, h, pr, nr)
    cnt = {"both": 0, "pos_only": 0, "neg_only": 0, "neither": 0}
    win_b = win_m = 0.0
    for a in range(P):
        for b in range(KK):
            x, y = pr[a], nr[a, b]
            if h[x] and h[y]:
                cnt["both"] += 1
                sc = 1.0 if v[x] > v[y] else 0.5 if v[x] == v[y] else 0.0
                win_b += sc
                win_m += sc
            elif h[x]:
                cnt["pos_only"] += 1
                win_m += 1.0
            elif h[y]:
                cnt["neg_only"] += 1
            else:
                cnt["neither"] += 1
                win_m += 0.5
    ok_ps &= all(s[k] == cnt[k] for k in cnt) and s["pairs"] == P * KK
    want_b = win_b / cnt["both"] if cnt["both"] else math.nan
    ok_ml &= math.isclose(s["auc_missing_last"], win_m / (P * KK)) and (
        (math.isnan(want_b) and math.isnan(s["auc_both"]))
        or math.isclose(s["auc_both"], want_b)
    )
check(
    ok_ps and ok_ml,
    "pair_states() equals a pair-by-pair count in 30 "
    "trials with ties and missing rows, for both AUCs",
)
ok_ra = True
for trial in range(30):
    x = np.round(rng.normal(size=50), 1)
    lab = rng.random(50) < 0.4
    got = stage_p12.rank_auc(rankdata(x), lab)
    pos, neg = x[lab], x[~lab]
    want = np.mean([1.0 if p > q else 0.5 if p == q else 0.0 for p in pos for q in neg])
    ok_ra &= math.isclose(got, want)
check(
    ok_ra and math.isnan(stage_p12.rank_auc(rankdata([1, 2]), [True, True])),
    "rank_auc() equals a pair-by-pair count with ties, and is NaN " "for one class",
)
ok_sp = True
for trial in range(20):
    x = np.round(rng.normal(size=40), 1)
    y = np.round(x + rng.normal(size=40), 1)
    ok_sp &= math.isclose(stage_p12.spearman(x, y), spearmanr(x, y)[0], abs_tol=1e-12)
check(ok_sp, "spearman() equals scipy's spearmanr with ties")
pos_rows, negs, k_neg = stage_p12.StageP12()._neg_rows(PAIRS)
ok_nr = k_neg == K
for n_pos, r in enumerate(pos_rows):
    j = int(np.searchsorted(PAIRS["indptr"], r, side="right") - 1)
    s0, e0 = PAIRS["indptr"][j], PAIRS["indptr"][j + 1]
    ok_nr &= (np.flatnonzero(PAIRS["kind"][s0:e0] == 1) + s0).tolist() == negs[1][
        n_pos
    ].tolist()
    ok_nr &= (np.flatnonzero(PAIRS["kind"][s0:e0] == 2) + s0).tolist() == negs[2][
        n_pos
    ].tolist()
check(
    ok_nr and len(pos_rows) == int((PAIRS["kind"] == 0).sum()),
    "_neg_rows() pairs each positive with its own user's negatives",
)
blocks = stage_p12._blocks([5, 1, 1, 1, 9, 2], 4)
check(
    blocks[0][0] == 0
    and blocks[-1][1] == 6
    and all(b[1] == c[0] for b, c in zip(blocks, blocks[1:]))
    and all(b[1] > b[0] for b in blocks)
    and stage_p12._blocks([3, 3], 10) == [(0, 1), (1, 2)]
    and stage_p12._blocks([], 4) == [],
    "_blocks() covers every user once, in order, without empty blocks",
)

# 8: report
st = new_stage()
st.verify({})
st.report({})
text = out["report"].read_text()
need = [
    "## 1. Three states per concept",
    "### Explanation coverage",
    "## 2. Each concept alone, within users",
    "### Against uniform negatives",
    "### Against p_pop negatives",
    "## 3. Overlap between concepts",
    "### Popularity behind the missing states",
    "## 4. Evidence size",
    "## 5. Named examples",
    "uniform negative, drawn at random)",
    "Artist ",
    "evidence lists sum to their concept values",
    f"| positives, train | {int(((PAIRS['kind'] == 0) & (np.repeat(split, np.diff(p_indptr)) == 0)).sum()):,} |",
]
missing = [x for x in need if x not in text]
check(not missing, f"report() writes every section ({missing or 'all'})")
vals, hs = first["values"], first["has"]
row_split = np.repeat(split, np.diff(p_indptr))
group_rows = [
    np.flatnonzero(
        (PAIRS["kind"] == kd) & ((row_split == sc) if sc is not None else True)
    )
    for _, kd, sc in stage_p12.GROUPS
]


def pct(n, d):
    return f"{100.0 * n / d:.1f}%" if d else "n/a"


def fmt(x, digits=3):
    return "n/a" if x is None or math.isnan(x) else f"{x:.{digits}f}"


j = col["artist_affinity"]
cells = []
for rows_g in group_rows:
    with_data = [r for r in rows_g if hs[r, j]]
    zero = [r for r in with_data if vals[r, j] == 0]
    med = (
        float(np.median([float(vals[r, j]) for r in with_data]))
        if with_data
        else math.nan
    )
    cells.append(
        f"{pct(len(with_data), len(rows_g))} / "
        f"{pct(len(zero), len(rows_g))} / {fmt(med, 4)}"
    )
want_states = "| artist_affinity | " + " | ".join(cells) + " |"
cov_rows = []
for (label, _, _), rows_g in zip(stage_p12.GROUPS, group_rows):
    k1 = k2 = only_audio = neither = hist_m = 0
    for r in rows_g:
        k = sum(1 for c in stage_p12.PERSONAL if hs[r, col[c]] and vals[r, col[c]] > 0)
        hist_m += any(
            hs[r, col[c]] and vals[r, col[c]] > 0
            for c in (
                "artist_affinity",
                "work_affinity",
                "colisten_same_artist",
                "colisten_other_artists",
            )
        )
        au = bool(hs[r, col["audio_fit"]])
        k1 += k >= 1
        k2 += k >= 2
        only_audio += k == 0 and au
        neither += k == 0 and not au
    nr = len(rows_g)
    cov_rows.append(
        f"| {label} | {nr:,} | {pct(k1, nr)} | "
        f"{pct(hist_m, nr)} | {pct(k2, nr)} | "
        f"{pct(only_audio, nr)} | {pct(neither, nr)} |"
    )
cnt = {"both": 0, "pos": 0, "neg": 0, "none": 0}
wins = wins_ml = 0.0
for r in np.flatnonzero(PAIRS["kind"] == 0):
    u = int(np.searchsorted(p_indptr, r, side="right") - 1)
    for q in range(p_indptr[u], p_indptr[u + 1]):
        if PAIRS["kind"][q] != 1:
            continue
        if hs[r, j] and hs[q, j]:
            cnt["both"] += 1
            sc = (
                1.0
                if vals[r, j] > vals[q, j]
                else 0.5 if vals[r, j] == vals[q, j] else 0.0
            )
            wins += sc
            wins_ml += sc
        elif hs[r, j]:
            cnt["pos"] += 1
            wins_ml += 1.0
        elif hs[q, j]:
            cnt["neg"] += 1
        else:
            cnt["none"] += 1
            wins_ml += 0.5
tot = sum(cnt.values())
want_alone = (
    f"| artist_affinity | {fmt(wins / cnt['both'])} | "
    f"{pct(cnt['both'], tot)} | {pct(cnt['pos'], tot)} | "
    f"{pct(cnt['neg'], tot)} | {pct(cnt['none'], tot)} | "
    f"{fmt(wins_ml / tot)} |"
)
jt, jp = col["tag_affinity"], col["popularity"]
aucs = []
for rows_g in group_rows:
    yes = [float(vals[r, jp]) for r in rows_g if hs[r, jt]]
    no = [float(vals[r, jp]) for r in rows_g if not hs[r, jt]]
    aucs.append(
        fmt(
            float(
                np.mean(
                    [1.0 if a > b else 0.5 if a == b else 0.0 for a in yes for b in no]
                )
            )
            if yes and no
            else math.nan
        )
    )
want_pop = "| tag_affinity | " + " | ".join(aucs) + " |"
ja = col["audio_fit"]
q_cells = []
for (label, _, _), rows_g in zip(stage_p12.GROUPS, group_rows):
    if label == "positives, selection":
        continue
    v = sorted(float(vals[r, ja]) for r in rows_g if hs[r, ja])
    q_cells.append(
        " / ".join(fmt(float(np.percentile(v, q)), 4) for q in (1, 50, 99))
        if v
        else "n/a"
    )
want_range = "| audio_fit | " + " | ".join(q_cells) + " |"
missing = [
    x
    for x in [want_states, want_alone, want_pop, want_range] + cov_rows
    if x not in text
]
check(
    not missing,
    f"the report's numbers equal direct counts: the three states, the "
    f"value range, the explanation coverage, the pair states and both "
    f"AUCs, and the popularity AUC ({missing or 'all'})",
)
ev_rows = [
    ln
    for ln in text.splitlines()
    if ln.startswith("| ")
    and ln.count("|") == 9
    and ln.split("|")[2].strip() in ("positive", "negative")
]
top1_ok = True
for ln in ev_rows:
    cells = [x.strip() for x in ln.split("|")[1:-1]]
    if cells[6] != "n/a" and cells[7] != "n/a":
        top1_ok &= 0.0 < float(cells[6]) <= float(cells[7]) <= 1.0
check(
    len(ev_rows) == 2 * (len(cn.CONTINUOUS) - 1) and top1_ok,
    "the evidence table has a row per concept and group, and each "
    "top-1 share lies in (0, top-3 share]",
)
import re

hist_cols = [
    col[c]
    for c in (
        "artist_affinity",
        "work_affinity",
        "colisten_same_artist",
        "colisten_other_artists",
    )
]
uni_rows = np.flatnonzero(PAIRS["kind"] == 1)
hist_val = np.nanmax(
    np.where(hs[uni_rows][:, hist_cols], vals[uni_rows][:, hist_cols], np.nan), axis=1
)
hist_val = np.nan_to_num(hist_val, nan=0.0)
top_rows = uni_rows[hist_val == hist_val.max()]  # ties go to the key
top_pairs = {
    (
        str(PAIRS["item"][r]),
        str(p_users[np.searchsorted(p_indptr, r, side="right") - 1]),
    )
    for r in top_rows
}
headings = re.findall(r"\n### .*\(item (\d+), listener (\d+), ([^)]+)\)", text)
roles = [h[2] for h in headings]
check(
    roles
    == ["positive"] * stage_p12.REPORT_EXAMPLES
    + ["uniform negative, drawn at random", "uniform negative with a history match"]
    and hist_val.max() > 0
    and [h[:2] for h in headings if h[2] == "uniform negative with a history match"][0]
    in top_pairs
    and text.count("| popularity | value |") == len(headings)
    and "| no data |" in text,
    f"report() names {stage_p12.REPORT_EXAMPLES} positives, a random "
    f"uniform negative and the uniform negative with the strongest "
    f"history match ({len(top_pairs)} rows share the largest value)",
)
phrase = {
    **{
        c: "no term shared with the listener's history"
        for c in ("tag_affinity", "artist_tag_affinity", "genre_affinity")
    },
    **{
        c: ": not in the listener's history"
        for c in ("artist_affinity", "work_affinity")
    },
    **{
        c: "no neighbour in the listener's history"
        for c in ("colisten_same_artist", "colisten_other_artists")
    },
}
no_match = [ln for ln in text.splitlines() if "| no match |" in ln]
kinds_seen = {ln.split("|")[1].strip() for ln in no_match}
check(
    re.search(r", \d+ history tracks?: ", text) is not None
    and re.search(r"\| \d+ shared terms?: ", text) is not None
    and len(no_match) > 0
    and {"artist_affinity", "colisten_same_artist"} <= kinds_seen
    and all(phrase[ln.split("|")[1].strip()] in ln for ln in no_match),
    f"the examples count the units behind each concept and say in words "
    f"when the data exist but do not match ({len(no_match)} no-match "
    f"lines, concepts {sorted(kinds_seen)})",
)

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL P12 CHECKS PASSED")
