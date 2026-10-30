"""The synthetic world of test_p13a.py as a module, for the tests of the
stages after P13a: the same code, the same seed and the same draw order,
so the same catalog, listeners, pairs and features.

A small catalog with every concept source, listeners with histories, and
held-out picks drawn from a planted additive preference (artist match,
tag match and popularity). The pairs come from P11's own functions
(fusion_pairs.user_rows) and the features from P12's own function
(concept_data.user_rows), so a stage reads data in the real layout.
"""

import copy

import numpy as np

from pipeline import concept_data as cd
from pipeline import concepts as cn
from pipeline import fusion_pairs as fp

rng = np.random.default_rng(13013)
NI = 700
DA = len(cn.AUDIO_DIMS)

# The world


def rows_csr(n_cols, p_empty, max_per_row):
    rows = [[]]
    for _ in range(1, NI):
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
    "tags_direct": rows_csr(20, 0.5, 3),
    "tags_artist": rows_csr(20, 0.2, 5),
    "genres": rows_csr(8, 0.4, 2),
}
terms_src = {k: (*pack(v), 20 if k != "genres" else 8) for k, v in term_rows.items()}
credits, credit_ids = {}, {}
for name, n_ent, p_empty in (("artists", 70, 0.03), ("works", 40, 0.7)):
    ip, raw = pack(rows_csr(n_ent, p_empty, 2), np.int64)
    c = cd.compact(ip, raw)
    credits[name], credit_ids[name] = c[:3], c[3]
colisten = {}
for kind_, nnb in (("same_artist", 5), ("other_artists", 3)):
    rn, rc = [[]], [[]]
    for i in range(1, NI):
        if rng.random() < 0.3:
            rn.append([])
            rc.append([])
            continue
        k = int(rng.integers(1, nnb + 1))
        nb = rng.choice(np.setdiff1d(np.arange(1, NI), [i]), size=k, replace=False)
        rn.append(nb.tolist())
        rc.append(rng.uniform(0.01, 1.0, size=k).tolist())
    ip, nb = pack(rn)
    colisten[kind_] = (ip, nb, np.array([x for r in rc for x in r]))
audio = rng.normal(size=(NI, DA))
audio[0] = np.nan
audio[rng.choice(np.arange(1, NI), 120, replace=False)] = np.nan
listens = rng.integers(1, 3000, NI)
popularity = np.log1p(listens.astype(np.float64))
popularity[0] = 0.0
tier = rng.integers(0, 6, NI)
tier[np.isnan(audio).all(axis=1)] = cn.MATCH_TIERS.index("unmatched")
SRC = cd.Sources(
    NI, terms_src, credits, credit_ids, colisten, audio, popularity, tier, {}, {}
)
SPACE = cd.item_side(copy.deepcopy(SRC), transposed=True)

NU = 520
hist = [np.zeros(0, dtype=np.int32)]
for u in range(1, NU):
    h = rng.choice(np.arange(1, NI), size=int(rng.integers(4, 60)), replace=False)
    hist.append(np.sort(h).astype(np.int32))
u_indptr = np.concatenate(([0], np.cumsum([len(h) for h in hist])))
u_indices = np.concatenate(hist).astype(np.int32)
u_data = np.round(1000 * np.log1p(rng.integers(1, 300, size=len(u_indices)))).astype(
    np.uint16
)

# Held-out picks from a planted additive preference.
held = {}
for u in range(1, NU):
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, u)
    st = cn.build_user_state(items_h, w_h, SPACE)
    full = cn.full_catalog(st, SPACE)
    art = np.nan_to_num(full["artist_affinity"][0])
    tag = np.nan_to_num(full["artist_tag_affinity"][0])
    score = 3.0 * (art > 0) + 2.0 * tag + 0.6 * popularity
    score[0] = -np.inf
    score[items_h] = -np.inf
    pr = np.exp(score - score.max())
    pr /= pr.sum()
    pk = rng.choice(NI, size=4, replace=False, p=pr)
    if u % 17 == 0:
        pk[1] = pk[2]  # val_last is the test_first pick (rule 3)
    if u % 23 == 0:
        pk[3] = pk[1]  # val_last and val_first coincide (rule 4)
    held[u] = pk
users_all = np.arange(1, NU)
SEL = np.sort(rng.choice(users_all, size=90, replace=False))
cols = [np.repeat(users_all, 4), np.concatenate([held[u] for u in users_all])]
flags = [np.tile(np.eye(4, dtype=bool)[j], len(users_all)) for j in range(4)]
p_users, picks4 = fp.pick_table(cols[0], cols[1], *flags)
p_indptr, p_items, p_flags, _ = fp.positives(picks4)
core = np.arange(1, NI)
p_pop = listens[core].astype(np.float64) ** 0.75
p_pop /= p_pop.sum()
cum = np.cumsum(p_pop)
p_pop_slot = np.zeros(NI)
p_pop_slot[core] = p_pop
samplers = {
    fp.KIND_UNIFORM: fp.uniform_sampler(1, NI),
    fp.KIND_POP: fp.pop_sampler(core, cum),
}
parts = {k: [] for k in ("item", "kind", "pick", "draw")}
keep_u, lens, n_allowed, pop_allowed = [], [], [], []
for j, u in enumerate(p_users):
    a, b = p_indptr[j], p_indptr[j + 1]
    if b == a:
        continue
    row = u_indices[u_indptr[u] : u_indptr[u + 1]]
    out = fp.user_rows(
        7,
        int(u),
        p_items[a:b],
        p_flags[a:b],
        row,
        picks4[j],
        16,
        samplers,
        NI - 1,
        p_pop_slot,
        float(p_pop.sum()),
    )
    for key, arr in zip(("item", "kind", "pick", "draw"), out[:4]):
        parts[key].append(arr)
    keep_u.append(int(u))
    lens.append(len(out[0]))
    n_allowed.append(out[4])
    pop_allowed.append(out[5])
keep_u = np.array(keep_u, dtype=np.int64)
P = {
    "user_id": keep_u,
    "split": np.where(np.isin(keep_u, SEL), fp.SPLIT_SELECTION, fp.SPLIT_TRAIN).astype(
        np.int8
    ),
    "indptr": np.concatenate(([0], np.cumsum(lens))).astype(np.int64),
    "item": np.concatenate(parts["item"]).astype(np.int32),
    "kind": np.concatenate(parts["kind"]),
    "n_allowed": np.array(n_allowed, dtype=np.int64),
    "pop_allowed": np.array(pop_allowed, dtype=np.float64),
}
P_PARAMS = {"version": "fusion-pairs-v1", "neg_per_kind": 16}
vals, hass, tiers = [], [], []
space_pairs = cd.item_side(copy.deepcopy(SRC), transposed=False)
for j, u in enumerate(keep_u):
    a, b = P["indptr"][j], P["indptr"][j + 1]
    items_h, w_h = cd.history_of(u_indptr, u_indices, u_data, int(u))
    v_, h_, t_ = cd.user_rows(
        space_pairs, items_h, w_h, P["item"][a:b].astype(np.int64)
    )
    vals.append(v_.astype(np.float32))
    hass.append(h_)
    tiers.append(t_.astype(np.int8))
F = {
    "values": np.concatenate(vals),
    "has": np.concatenate(hass),
    "match_tier": np.concatenate(tiers),
}
F_PARAMS = {"concepts": list(cn.CONTINUOUS)}
SEL_PICKS = np.array([[held[u][1], held[u][3]] for u in SEL], dtype=np.int64)


def summary():
    return (
        f"world: {NI} slots, {len(keep_u)} listeners with pairs "
        f"({int((P['split'] == 0).sum())} train), {len(P['item'])} rows, "
        f"{len(SEL)} selection listeners"
    )
