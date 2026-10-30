"""Checks for pipeline/colisten.py on small synthetic data. No real data.

1. share_artist(), artist_in_history() and artist_tracks() equal brute
   force with Python sets, tracks without credits included.
2. sample_users() picks each user's k most-played and k least-popular
   items, ties to the smaller id, exactly as a per-user sort does, with
   and without an eligibility mask.
3. top_per_row() keeps each row's top n by score, ties to the smaller
   column, including ties across the cut.
4. candidate_pairs() equals a dense brute force, with and without the
   same-artist exclusion. Two workers give the same arrays as one.
5. union_candidates() merges two candidate sets with their supports.
6. exact_co_listeners() equals the dense co-occurrence matrix, with one
   and two workers, with many tracks per block and with sliced partners.
7. directed_topn() and rank_rows() equal a brute-force sort by exact
   cosine, ties to the smaller id. rank_rows() merges repeated entries
   and refuses a repeated pair with two counts.
8. same_artist_lists() equals the exact same-artist top-n of every
   track, with one worker and big blocks, and with two workers and row
   blocks of one row.
9. exact_topn() equals the dense whole-catalog top-n, overall and split
   by artist.
10. Without sampling (k above every history length, no candidate cap),
    the other-artist chain equals the exact other-artist top-n of every
    track. So the chain adds no error of its own beyond the sampling.
11. knn_problems() passes both graphs and names a planted fault.
12. pick_hits() equals a brute-force set intersection on a directed graph.
13. Progress lines appear for the long steps.

Usage, from the repository root:
    python3 test_colisten.py
"""

import numpy as np

from pipeline import colisten as cl

failures = []


def check(cond, msg):
    print(("PASS  " if cond else "FAIL  ") + msg)
    if not cond:
        failures.append(msg)


def csr_from_dense(M, D=None):
    r, c = np.nonzero(M)
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(r, minlength=M.shape[0])))
    ).astype(np.int64)
    data = None if D is None else D[r, c].astype(np.uint16)
    return indptr, c.astype(np.int32), data


rng = np.random.default_rng(29)
NU, NI = 260, 90  # slot 0 is phantom on both sides
P = rng.random((NU, NI)) < rng.uniform(0.03, 0.5, size=(NU, 1))
P[0, :] = False
P[:, 0] = False
P[7, :] = False  # a user without data
P[:, 85:] = False  # rare tracks: 1 to 3 listeners
for x in range(85, NI):
    P[rng.choice(np.arange(1, NU), size=int(rng.integers(1, 4)), replace=False), x] = (
        True
    )
P[7, :] = False
P[:, 86] = P[:, 87] = False  # two tracks at the support floor of
P[[11, 12, 13], 86] = True  # the same-artist test (3), played by
P[[11, 12, 13], 87] = True  # the same 3 listeners
L = np.where(P, rng.integers(1, 60, size=(NU, NI)), 0)  # many ties
D = np.where(P, np.round(1000 * np.log1p(L)), 0).astype(np.int64)
u_indptr, u_indices, u_data = csr_from_dense(P, D)
i_indptr, i_indices, _ = csr_from_dense(P.T)
pop = np.diff(i_indptr).astype(np.int64)
C_full = P.T.astype(np.int64) @ P.astype(np.int64)
lines = []

# Artists: 0 to 2 credits per track from 9 artists, some tracks without.
item_art = [set() for _ in range(NI)]
for x in range(1, NI):
    item_art[x] = set(
        rng.choice(9, size=int(rng.choice([0, 1, 1, 1, 2])), replace=False).tolist()
    )
item_art[86], item_art[87] = {0}, {0, 5}  # they share artist 0
ai = np.array([x for x in range(NI) for _ in item_art[x]], dtype=np.int64)
aa = np.array([a for x in range(NI) for a in sorted(item_art[x])], dtype=np.int64)
art_indptr = np.concatenate(([0], np.cumsum(np.bincount(ai, minlength=NI)))).astype(
    np.int64
)
art = (art_indptr, aa)
SHARE = np.array(
    [[bool(item_art[x] & item_art[y]) for y in range(NI)] for x in range(NI)]
)

# 1: artist helpers.
pa, pb = rng.integers(0, NI, size=2000), rng.integers(0, NI, size=2000)
check(
    np.array_equal(cl.share_artist(pa, pb, *art, chunk=333), SHARE[pa, pb])
    and SHARE[pa, pb].any()
    and (~SHARE[pa, pb]).any(),
    "share_artist() equals brute force, over chunks",
)
users = np.arange(1, NU, 5)
picks = rng.integers(0, NI, size=(len(users), 3))
picks[::4, 1] = 0
hit = cl.artist_in_history(users, picks, u_indptr, u_indices, *art)
want = np.array(
    [
        [
            p > 0 and any(item_art[p] & item_art[h] for h in np.flatnonzero(P[u]))
            for p in picks[k]
        ]
        for k, u in enumerate(users)
    ]
)
check(
    np.array_equal(hit, want) and want.any() and (~want).any(),
    "artist_in_history() equals brute force",
)
elig = pop >= 3
elig[0] = False
a_indptr, a_items = cl.artist_tracks(art_indptr, aa, elig)
ok = len(a_indptr) == 10
for a in range(9):
    got = a_items[a_indptr[a] : a_indptr[a + 1]].tolist()
    ok &= got == [x for x in range(NI) if a in item_art[x] and elig[x]]
check(ok, "artist_tracks() lists each artist's eligible tracks in id order")

# 2: samples.
K = 6


def brute_samples(mask):
    S_most = np.zeros_like(P)
    S_low = np.zeros_like(P)
    for u in range(NU):
        it = np.flatnonzero(P[u] & mask)
        S_most[u, it[np.lexsort((it, -D[u, it]))[:K]]] = True
        S_low[u, it[np.lexsort((it, pop[it]))[:K]]] = True
    return S_most, S_low


def as_dense(indptr, idx):
    M = np.zeros_like(P)
    for u in range(NU):
        M[u, idx[indptr[u] : indptr[u + 1]]] = True
    return M


for label, mask in (("all items", None), ("eligible items only", elig)):
    s_indptr, s_most, s_low = cl.sample_users(
        u_indptr, u_indices, u_data, pop, K, log=lines.append, batch=37, eligible=mask
    )
    want_m, want_l = brute_samples(np.ones(NI, bool) if mask is None else mask)
    rows_sorted = all(
        np.all(np.diff(s_most[s_indptr[u] : s_indptr[u + 1]]) > 0)
        and np.all(np.diff(s_low[s_indptr[u] : s_indptr[u + 1]]) > 0)
        for u in range(NU)
    )
    check(
        np.array_equal(as_dense(s_indptr, s_most), want_m)
        and np.array_equal(as_dense(s_indptr, s_low), want_l)
        and rows_sorted,
        f"sample_users() equals a per-user sort ({label}), rows sorted",
    )
check(
    not (as_dense(s_indptr, s_low)[:, ~elig]).any(),
    "an ineligible track takes no sample slot",
)
s_indptr, s_most, s_low = cl.sample_users(
    u_indptr, u_indices, u_data, pop, K, eligible=elig
)
got_most, got_low = as_dense(s_indptr, s_most), as_dense(s_indptr, s_low)

# 3: top_per_row with ties across the cut.
indptr_t = np.array([0, 5, 7, 7], dtype=np.int64)
cols_t = np.array([9, 3, 5, 1, 8, 2, 4])
score_t = np.array([0.5, 0.9, 0.5, 0.5, 0.1, 0.3, 0.3])
keep = cl.top_per_row(indptr_t, cols_t, score_t, 3)
check(
    set(cols_t[:5][keep[:5]].tolist()) == {3, 1, 5} and keep[5:7].all(),
    "top_per_row() breaks ties at the cut by the smaller column",
)


def brute_candidates(S, ms, per_item, exclude_same):
    C = S.T.astype(np.int64) @ S.astype(np.int64)
    h = S.sum(axis=0).astype(np.float64)
    pairs = {}
    for a in range(NI):
        b = np.array(
            [
                x
                for x in range(NI)
                if x != a and C[a, x] >= ms and not (exclude_same and SHARE[a, x])
            ]
        )
        if len(b) == 0:
            continue
        sc = C[a, b] / np.sqrt(h[a] * h[b])
        for x in b[np.lexsort((b, -sc))[:per_item]]:
            pairs[(min(a, x), max(a, x))] = int(C[a, x])
    keys = sorted(pairs)
    return (
        np.array([k[0] for k in keys]),
        np.array([k[1] for k in keys]),
        np.array([pairs[k] for k in keys]),
    )


# 4: candidates, with and without the exclusion, one and two workers.
MS, PER = 2, 4
for name, samp, dense in (
    ("most-played", s_most, got_most),
    ("least-popular", s_low, got_low),
):
    for excl in (False, True):
        want = brute_candidates(dense, MS, PER, excl)
        kw = {"art": art} if excl else {}
        got1 = cl.candidate_pairs(
            s_indptr, samp, NI, MS, PER, workers=1, block_cost=150, **kw
        )
        got2 = cl.candidate_pairs(
            s_indptr, samp, NI, MS, PER, workers=2, block_cost=150, **kw
        )
        same_free = (not excl) or not SHARE[got1[0], got1[1]].any()
        check(
            len(want[0]) > 10
            and same_free
            and all(np.array_equal(w, g) for w, g in zip(want, got1))
            and all(np.array_equal(x, y) for x, y in zip(got1, got2)),
            f"candidate_pairs() ({name}, "
            f"{'other artists only' if excl else 'all pairs'}) equals "
            f"brute force, and 2 workers equal 1 ({len(want[0])} pairs)",
        )

# 5: union.
cm = cl.candidate_pairs(s_indptr, s_most, NI, MS, PER, workers=1, art=art)
cn = cl.candidate_pairs(s_indptr, s_low, NI, MS, PER, workers=1, art=art)
ua, ub, usm, usl = cl.union_candidates(cm, cn, NI)
want = {}
for a, b, s in zip(*cm):
    want.setdefault((int(a), int(b)), [0, 0])[0] = int(s)
for a, b, s in zip(*cn):
    want.setdefault((int(a), int(b)), [0, 0])[1] = int(s)
keys = sorted(want)
check(
    list(zip(ua.tolist(), ub.tolist())) == keys
    and usm.tolist() == [want[k][0] for k in keys]
    and usl.tolist() == [want[k][1] for k in keys],
    "union_candidates() merges both sets with their supports",
)

# 6: exact counts.
co1 = cl.exact_co_listeners(
    ua, ub, i_indptr, i_indices, NU, workers=1, block_cost=200, log=lines.append
)
co2 = cl.exact_co_listeners(ua, ub, i_indptr, i_indices, NU, workers=2, block_cost=200)
check(
    np.array_equal(co1, C_full[ua, ub]) and np.array_equal(co1, co2),
    "exact_co_listeners() equals the dense co-occurrence matrix, with 1 "
    "and 2 workers",
)
co4 = cl.exact_co_listeners(ua, ub, i_indptr, i_indices, NU, workers=1)
check(
    np.array_equal(co4, co1),
    "exact_co_listeners() gives the same counts when one block holds many "
    "tracks (the bitmap is cleared between them)",
)
saved_limit = cl.GATHER_LIMIT
cl.GATHER_LIMIT = 7  # force partner slices on hub tracks
co3 = cl.exact_co_listeners(ua, ub, i_indptr, i_indices, NU, workers=2, block_cost=200)
cl.GATHER_LIMIT = saved_limit
check(
    np.array_equal(co3, co1),
    "exact_co_listeners() gives the same counts when partners are sliced",
)
check(
    np.all(usm <= co1) and np.all(usl <= co1),
    "in-sample supports never exceed the exact counts",
)


# 7: directed_topn and rank_rows.
def brute_rank(x, cands, top_n, ms):
    cands = np.array([y for y in cands if C_full[x, y] >= ms], dtype=np.int64)
    if len(cands) == 0:
        return cands, cands, np.zeros(0)
    cos = cl.cosine(C_full[x, cands], pop[x], pop[cands])
    o = np.lexsort((cands, -cos))[:top_n]
    return cands[o], C_full[x, cands][o], cos[o]


def graph_equals(g, lists):
    ok = True
    for x in range(NI):
        s, e = g["indptr"][x], g["indptr"][x + 1]
        nb, c, cs = lists(x)
        ok &= (
            np.array_equal(g["neighbor"][s:e], nb)
            and np.array_equal(g["co"][s:e], c)
            and np.array_equal(g["cosine"][s:e], cs)
        )
    return ok


TOPN = 3
knn = cl.directed_topn(
    ua, ub, co1, usm, usl, pop, NI, TOPN, 3, log=lines.append, n_buckets=5
)
partners = {x: set() for x in range(NI)}
for a, b in zip(ua.tolist(), ub.tolist()):
    partners[a].add(b)
    partners[b].add(a)
check(
    graph_equals(knn, lambda x: brute_rank(x, sorted(partners[x]), TOPN, 3)),
    "directed_topn() equals a brute-force sort by exact cosine, ties "
    "to the smaller id, support floor 3",
)
er, en = rng.integers(1, NI, size=900), rng.integers(1, NI, size=900)
m = er != en
er, en = er[m], en[m]
ec = C_full[er, en]
dup = rng.integers(0, len(er), size=100)
rr = cl.rank_rows(
    np.concatenate((er, er[dup])).astype(np.int32),
    np.concatenate((en, en[dup])).astype(np.int32),
    np.concatenate((ec, ec[dup])).astype(np.int32),
    pop,
    NI,
    4,
    2,
    n_buckets=3,
)
sets = {x: set(en[er == x].tolist()) for x in range(NI)}
check(
    graph_equals(rr, lambda x: brute_rank(x, sorted(sets[x]), 4, 2)),
    "rank_rows() equals brute force and merges repeated entries",
)
try:
    cl.rank_rows(np.array([5, 5]), np.array([6, 6]), np.array([3, 4]), pop, NI, 4, 2)
    check(False, "rank_rows() refuses a repeated pair with two counts")
except RuntimeError:
    check(True, "rank_rows() refuses a repeated pair with two counts")


# 8: same-artist lists.
def exact_kind(x, kind, top_n, ms):
    cands = [
        y
        for y in range(NI)
        if y != x and (SHARE[x, y] if kind == "same" else not SHARE[x, y])
    ]
    return brute_rank(x, cands, top_n, ms)


for label, kw in (
    ("1 worker, big blocks", {"workers": 1}),
    ("2 workers, one-row blocks", {"workers": 2, "block_cost": 50, "pair_budget": 1}),
):
    same = cl.same_artist_lists(
        *art, i_indptr, i_indices, pop, 3, TOPN, log=lines.append, **kw
    )
    s86 = same["neighbor"][same["indptr"][86] : same["indptr"][87]]
    check(
        graph_equals(same, lambda x: exact_kind(x, "same", TOPN, 3))
        and len(same["neighbor"]) > 20
        and 87 in s86.tolist(),
        f"same_artist_lists() equals the exact same-artist top {TOPN} of "
        f"every track ({label})",
    )

# 9: exact_topn overall and by kind.
items = np.array([1, 2, 5, 11, 40, 86])
ex = cl.exact_topn(
    items, u_indptr, u_indices, i_indptr, i_indices, TOPN, 3, chunk=50, log=lines.append
)
ek = cl.exact_topn(
    items, u_indptr, u_indices, i_indptr, i_indices, TOPN, 3, chunk=50, art=art
)
ok = True
for x in items:
    x = int(x)
    for got, want in (
        (ex[x], brute_rank(x, [y for y in range(NI) if y != x], TOPN, 3)),
        (ek[x]["same"], exact_kind(x, "same", TOPN, 3)),
        (ek[x]["other"], exact_kind(x, "other", TOPN, 3)),
    ):
        ok &= all(np.array_equal(g, w) for g, w in zip(got, want))
check(ok, "exact_topn() equals the dense whole-catalog top-n, overall and " "by kind")

# 10: without sampling, the other-artist chain is exact.
big_k = int(np.diff(u_indptr).max()) + 1
f_indptr, f_most, f_low = cl.sample_users(u_indptr, u_indices, u_data, pop, big_k)
fm = cl.candidate_pairs(f_indptr, f_most, NI, 1, 10**6, workers=1, art=art)
fa, fb, fsm, fsl = cl.union_candidates(fm, fm, NI)
fco = cl.exact_co_listeners(fa, fb, i_indptr, i_indices, NU, workers=1)
fknn = cl.directed_topn(fa, fb, fco, fsm, fsl, pop, NI, TOPN, 3)
check(
    np.array_equal(fsm, fco)
    and graph_equals(fknn, lambda x: exact_kind(x, "other", TOPN, 3)),
    "without sampling, supports equal exact counts and the other-artist "
    "chain equals the exact other-artist top-n of every track",
)

# 11: knn_problems.
same = cl.same_artist_lists(*art, i_indptr, i_indices, pop, 3, TOPN, workers=1)
bad = {k: v.copy() for k, v in same.items()}
r0 = int(np.flatnonzero(np.diff(bad["indptr"]) >= 2)[0])
s0 = int(bad["indptr"][r0])
bad["co"][s0] += 1
high = {k: v.copy() for k, v in fknn.items()}
high["support_most"][0] = high["co"][0] + 1
low_s = {k: v.copy() for k, v in fknn.items()}
low_s["support_most"][0] = low_s["support_low"][0] = 2
check(
    not cl.knn_problems(same, pop, TOPN, 3)
    and not cl.knn_problems(fknn, pop, TOPN, 3)
    and any("cosine" in p for p in cl.knn_problems(bad, pop, TOPN, 3))
    and any("exceeds" in p for p in cl.knn_problems(high, pop, TOPN, 3))
    and any("no sample" in p for p in cl.knn_problems(low_s, pop, TOPN, 3)),
    "knn_problems() passes both graphs and names a changed count and "
    "both support faults",
)

# 12: pick hits on the directed graph.
hits = cl.pick_hits(knn["indptr"], knn["neighbor"], u_indptr, u_indices, users, picks)
want = np.full(picks.shape, -1)
for k, u in enumerate(users):
    hist = set(np.flatnonzero(P[u]).tolist())
    for j in range(3):
        p = picks[k, j]
        if p > 0:
            nb = knn["neighbor"][knn["indptr"][p] : knn["indptr"][p + 1]]
            want[k, j] = len(set(nb.tolist()) & hist)
check(
    np.array_equal(hits, want),
    "pick_hits() equals a brute-force set " "intersection on the directed graph",
)

# 13: progress.
wanted = [
    "most-played and least-popular",
    "exact co-listener counts",
    "top other-artist neighbours per track",
    "exact same-artist counts",
    "top same-artist neighbours per track",
    "validation tracks",
    "done in",
]
missing = [w for w in wanted if not any(w in line for line in lines)]
check(not missing, f"progress lines appear (missing: {missing or 'none'})")

print()
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    raise SystemExit(1)
print("ALL COLISTEN CHECKS PASSED")
