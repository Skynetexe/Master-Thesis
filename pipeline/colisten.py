"""Exact item-item co-listening for the two co-listening concepts.

Stage P9b uses this module.

- v1 replaced P9's sampled counts with exact counts. Its report and a
  recall diagnosis then showed two problems. Mid and
  rare tracks got almost no neighbours (a median list of 2 in pop band
  51-90, 0 in band 11-50). And for 91 percent of tracks the closest
  co-listened track is by the same artist, in the exact lists too, so a
  single co-listening concept mostly repeats artist_affinity.
- v2 (this version) gives each track two lists, one per concept. Two
  tracks share an artist when their credits intersect under the rule in
  pipeline/artists.py.

Same-artist lists (colisten_same_artist) are exact. For each artist, the
listeners of its tracks give the exact co-listener count of every pair
of its tracks, from one sparse product per artist. The work is the sum,
over listeners and artists, of the squared number of that artist's
tracks in the history, far below the whole-catalog cost. Each track
keeps its top_n same-artist neighbours by exact cosine.

Other-artist lists (colisten_other_artists) come from two samples of k
tracks per listener, as in v1: the k most-played and the k
least-popular. Two changes from v1. Tracks with fewer than min_support
listeners take no sample slot, because they can never have a neighbour.
And same-artist pairs are dropped before each track keeps its per_item
best candidates, so the candidate budget goes to other artists. Every
candidate pair is recounted exactly on all train listeners. Each track
keeps its top_n other-artist candidates by exact cosine.

Both lists use the same cosine: the exact co-listener count over the
square root of the two tracks' train listener counts (Sarwar et al.
2001). Ties go to the smaller item_id. The graphs are directed: a
track's neighbours need not list it back.

Validation: exact_lists() computes a track's exact whole-catalog list
with no sampling, and exact_topn() cuts it by kind.

All functions take plain arrays and have no file or DuckDB dependency,
so test_colisten.py checks each one against brute force.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import scipy.sparse as sp

from .als import Progress

_CTX = {}


def _quiet(msg):
    pass


def _workers(workers):
    return workers or max(1, min(16, os.cpu_count() or 1))


def _run_tasks(fn, tasks, costs, workers, prog):
    """fn over tasks, in task order. Forked workers share _CTX."""
    results = []
    if workers <= 1 or len(tasks) <= 1:
        for task, cost in zip(tasks, costs):
            results.append(fn(task))
            prog.advance(cost)
        return results
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=mp.get_context("fork")
    ) as pool:
        for out, cost in zip(pool.map(fn, tasks), costs):
            results.append(out)
            prog.advance(cost)
    return results


def _ranges(costs, target):
    """Contiguous index ranges whose summed cost stays near target. A
    single element above target gets a range of its own."""
    edges = [0]
    acc = 0
    for k, c in enumerate(costs):
        if acc and acc + c > target:
            edges.append(k)
            acc = 0
        acc += int(c)
    edges.append(len(costs))
    return [
        (edges[k], edges[k + 1])
        for k in range(len(edges) - 1)
        if edges[k + 1] > edges[k]
    ]


def _gather_index(indptr, rows):
    """Positions of the concatenated CSR rows `rows`, and the start of each
    row inside that concatenation."""
    starts = np.asarray(indptr[rows], dtype=np.int64)
    lens = np.asarray(indptr[rows + 1], dtype=np.int64) - starts
    offs = np.concatenate(([0], np.cumsum(lens)))
    idx = np.repeat(starts - offs[:-1], lens) + np.arange(offs[-1])
    return idx, offs


def _cat(parts, dtype):
    return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)


# Artists


def share_artist(a, b, art_indptr, art_ids, chunk=4_000_000):
    """For each pair (a[k], b[k]): True when the two tracks share at least
    one artist credit. art_indptr and art_ids are a credit CSR over item
    slots (pipeline/artists.py). A track without credits shares none."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    art_ids = np.asarray(art_ids, dtype=np.int64)
    out = np.zeros(len(a), dtype=bool)
    base = int(art_ids.max()) + 1 if len(art_ids) else 1
    for s in range(0, len(a), chunk):
        ia, oa = _gather_index(art_indptr, a[s : s + chunk])
        ib, ob = _gather_index(art_indptr, b[s : s + chunk])
        n = len(oa) - 1
        ka = np.repeat(np.arange(n, dtype=np.int64), np.diff(oa)) * base + art_ids[ia]
        pb = np.repeat(np.arange(n, dtype=np.int64), np.diff(ob))
        hit = np.isin(pb * base + art_ids[ib], ka)
        out[s + pb[hit]] = True
    return out


def artist_in_history(users, picks, u_indptr, u_indices, art_indptr, art_ids):
    """hit[k, j]: True when pick j of user k shares an artist with a track
    in the user's train history. False where the pick is missing (item
    0)."""
    hit = np.zeros(picks.shape, dtype=bool)
    art_ids = np.asarray(art_ids, dtype=np.int64)
    for k, u in enumerate(users):
        s, e = int(u_indptr[u]), int(u_indptr[u + 1])
        hist = np.asarray(u_indices[s:e]).astype(np.int64)
        idx, _ = _gather_index(art_indptr, hist)
        hist_art = np.unique(art_ids[idx])
        for j in range(picks.shape[1]):
            p = int(picks[k, j])
            if p > 0:
                hit[k, j] = bool(
                    np.isin(art_ids[art_indptr[p] : art_indptr[p + 1]], hist_art).any()
                )
    return hit


def artist_tracks(art_indptr, art_ids, eligible):
    """Artist id -> its eligible tracks, as CSR arrays over artist ids,
    each row sorted by item_id."""
    n_items = len(art_indptr) - 1
    item = np.repeat(np.arange(n_items, dtype=np.int64), np.diff(art_indptr))
    art = np.asarray(art_ids, dtype=np.int64)
    keep = np.asarray(eligible, dtype=bool)[item]
    item, art = item[keep], art[keep]
    order = np.lexsort((item, art))
    item, art = item[order], art[order]
    n_art = int(np.max(art_ids)) + 1 if len(art_ids) else 1
    a_indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(art, minlength=n_art)))
    ).astype(np.int64)
    return a_indptr, item


# Samples


def sample_users(
    u_indptr, u_indices, u_data, pop, k, log=None, batch=4096, eligible=None
):
    """Two samples per user, as CSR rows over users with sorted item ids:
    the k most-played items (u_data descending, then item_id) and the k
    least-popular items (pop ascending, then item_id). With eligible (a
    boolean mask over items), only eligible items take part. A user with
    k such items or fewer keeps all of them in both. Returns (indptr,
    most, low)."""
    log = log or _quiet
    n_users = len(u_indptr) - 1
    base = np.int64(len(pop) + 1)
    lens = np.zeros(n_users, dtype=np.int64)
    most, low = [], []
    prog = Progress(
        "colisten: each listener's most-played and " "least-popular tracks",
        n_users,
        log,
        unit=" users",
    )
    for lo in range(0, n_users, batch):
        hi = min(lo + batch, n_users)
        s, e = int(u_indptr[lo]), int(u_indptr[hi])
        items = np.asarray(u_indices[s:e]).astype(np.int64)
        data = np.asarray(u_data[s:e]).astype(np.int64)
        offs = np.asarray(u_indptr[lo : hi + 1], dtype=np.int64) - s
        if eligible is not None:
            keep = np.asarray(eligible, dtype=bool)[items]
            offs = np.concatenate(([0], np.cumsum(keep, dtype=np.int64)))[offs]
            items, data = items[keep], data[keep]
        key_low = pop[items] * base + items  # unique within a row
        key_most = (65535 - data) * base + items
        for r in range(hi - lo):
            a, b = int(offs[r]), int(offs[r + 1])
            row = items[a:b]
            if b - a <= k:
                sel_m = sel_l = np.sort(row)
            else:
                sel_m = np.sort(row[np.argpartition(key_most[a:b], k - 1)[:k]])
                sel_l = np.sort(row[np.argpartition(key_low[a:b], k - 1)[:k]])
            lens[lo + r] = len(sel_m)
            most.append(sel_m)
            low.append(sel_l)
        prog.advance(hi - lo)
    prog.finish()
    indptr = np.concatenate(([0], np.cumsum(lens))).astype(np.int64)
    return indptr, _cat(most, np.int32), _cat(low, np.int32)


# Candidates


def top_per_row(indptr, cols, score, n):
    """Boolean mask of the entries in each row's top n, ranked by score
    descending, then by column ascending. Exact at ties."""
    keep = np.zeros(len(cols), dtype=bool)
    lens = np.diff(indptr)
    short = np.repeat(lens <= n, lens)
    keep[short] = True
    for r in np.flatnonzero(lens > n):
        s, e = int(indptr[r]), int(indptr[r + 1])
        seg = score[s:e]
        kth = np.partition(seg, len(seg) - n)[len(seg) - n]  # n-th largest
        above = np.flatnonzero(seg > kth)
        ties = np.flatnonzero(seg == kth)
        ties = ties[np.argsort(cols[s:e][ties], kind="stable")]
        keep[s + above] = True
        keep[s + ties[: n - len(above)]] = True
    return keep


def _pairs_block(task):
    lo, hi = task
    Tt, T = _CTX["Tt"], _CTX["T"]
    holders = _CTX["holders"]
    C = (Tt[lo:hi] @ T).tocsr()
    C.sum_duplicates()
    rows = np.repeat(np.arange(lo, hi, dtype=np.int64), np.diff(C.indptr))
    cols = C.indices.astype(np.int64)
    sup = C.data.astype(np.int64)
    ok = (cols != rows) & (sup >= _CTX["min_support"])
    rows, cols, sup = rows[ok], cols[ok], sup[ok]
    if _CTX["art"] is not None:  # other-artist candidates only
        other = ~share_artist(rows, cols, *_CTX["art"])
        rows, cols, sup = rows[other], cols[other], sup[other]
    order = np.lexsort((cols, rows))
    rows, cols, sup = rows[order], cols[order], sup[order]
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(rows - lo, minlength=hi - lo)))
    ).astype(np.int64)
    score = sup / np.sqrt(holders[rows].astype(np.float64) * holders[cols])
    keep = top_per_row(indptr, cols, score, _CTX["per_item"])
    rows, cols, sup = rows[keep], cols[keep], sup[keep]
    return (
        np.minimum(rows, cols).astype(np.int32),
        np.maximum(rows, cols).astype(np.int32),
        sup.astype(np.int32),
    )


def candidate_pairs(
    s_indptr,
    s_indices,
    n_items,
    min_support,
    per_item,
    workers=None,
    log=None,
    label="colisten: candidates",
    block_cost=5_000_000,
    art=None,
):
    """Undirected candidate pairs from one sample: (a, b, support) with
    a < b, sorted by (a, b). support counts the users whose sample holds
    both items. Each item keeps its per_item partners with the highest
    in-sample cosine (support over the square root of both items' holder
    counts), ties to the smaller id, among partners with support >=
    min_support. With art (a credit CSR), partners that share an artist
    with the item take no part. A pair is a candidate when either end
    keeps it."""
    log = log or _quiet
    n_users = len(s_indptr) - 1
    T = sp.csr_matrix(
        (
            np.ones(len(s_indices), dtype=np.int32),
            np.asarray(s_indices, dtype=np.int32),
            np.asarray(s_indptr, dtype=np.int64),
        ),
        shape=(n_users, n_items),
    )
    Tt = T.T.tocsr()
    holders = np.diff(Tt.indptr).astype(np.int64)
    row_len = np.diff(T.indptr).astype(np.int64)
    # Work per item: the sample entries of all its holders.
    cost = np.bincount(
        T.indices, weights=np.repeat(row_len, row_len), minlength=n_items
    ).astype(np.int64)
    tasks = _ranges(cost, block_cost)
    costs = [int(cost[a:b].sum()) for a, b in tasks]
    _CTX.update(
        Tt=Tt,
        T=T,
        holders=holders,
        min_support=int(min_support),
        per_item=int(per_item),
        art=art,
    )
    prog = Progress(label, max(sum(costs), 1), log, unit=" entries")
    try:
        parts = _run_tasks(_pairs_block, tasks, costs, _workers(workers), prog)
    finally:
        _CTX.clear()
    prog.finish()
    a = _cat([p[0] for p in parts], np.int64)
    b = _cat([p[1] for p in parts], np.int64)
    s = _cat([p[2] for p in parts], np.int64)
    key = a * n_items + b
    order = np.argsort(key, kind="stable")
    key, s = key[order], s[order]
    first = np.concatenate(([True], key[1:] != key[:-1]))
    dup_ok = np.all(s[1:][~first[1:]] == s[:-1][~first[1:]])
    if not dup_ok:
        raise RuntimeError("a pair got two different supports")
    key, s = key[first], s[first]
    return (
        (key // n_items).astype(np.int32),
        (key % n_items).astype(np.int32),
        s.astype(np.int32),
    )


def union_candidates(most, low, n_items):
    """The union of two candidate sets (a, b, support), sorted by (a, b).
    support_most and support_low are 0 where a set lacks the pair."""
    km = most[0].astype(np.int64) * n_items + most[1]
    kl = low[0].astype(np.int64) * n_items + low[1]
    keys = np.union1d(km, kl)
    sm = np.zeros(len(keys), dtype=np.int32)
    sl = np.zeros(len(keys), dtype=np.int32)
    sm[np.searchsorted(keys, km)] = most[2]
    sl[np.searchsorted(keys, kl)] = low[2]
    return (
        (keys // n_items).astype(np.int32),
        (keys % n_items).astype(np.int32),
        sm,
        sl,
    )


# Exact counts
GATHER_LIMIT = 8_000_000  # listener ids gathered at once per worker


def _count_block(task):
    g0, g1 = task
    big, small, gstart = _CTX["big"], _CTX["small"], _CTX["gstart"]
    i_indptr, i_indices = _CTX["i_indptr"], _CTX["i_indices"]
    pop = np.diff(i_indptr)
    mask = np.zeros(_CTX["n_users"], dtype=bool)
    s0, e0 = int(gstart[g0]), int(gstart[g1])
    out = np.empty(e0 - s0, dtype=np.int64)
    for g in range(g0, g1):
        s, e = int(gstart[g]), int(gstart[g + 1])
        x = int(big[s])
        ux = np.asarray(i_indices[int(i_indptr[x]) : int(i_indptr[x + 1])])
        mask[ux] = True
        parts = small[s:e].astype(np.int64)
        # Slices of partners, so a hub track never gathers too much.
        for p0, p1 in _ranges(pop[parts], GATHER_LIMIT):
            idx, offs = _gather_index(i_indptr, parts[p0:p1])
            hits = mask[np.asarray(i_indices[idx])]
            cs = np.concatenate(([0], np.cumsum(hits, dtype=np.int64)))
            out[s - s0 + p0 : s - s0 + p1] = cs[offs[1:]] - cs[offs[:-1]]
        mask[ux] = False
    return out


def exact_co_listeners(
    a, b, i_indptr, i_indices, n_users, workers=None, log=None, block_cost=50_000_000
):
    """For each pair (a[k], b[k]), the number of users in both rows of the
    item-major CSR: the exact number of train listeners who played both.
    Each pair is counted from its end with more listeners: that end's
    users go into a bitmap once, and the other end's users are looked up
    in it, so a pair costs the smaller listener count."""
    log = log or _quiet
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    pop = np.diff(np.asarray(i_indptr, dtype=np.int64))
    a_big = (pop[a] > pop[b]) | ((pop[a] == pop[b]) & (a > b))
    big = np.where(a_big, a, b)
    small = np.where(a_big, b, a)
    order = np.lexsort((small, big))
    big_s, small_s = big[order], small[order]
    gstart = np.flatnonzero(np.concatenate(([True], big_s[1:] != big_s[:-1])))
    gstart = np.concatenate((gstart, [len(big_s)])).astype(np.int64)
    # Work per group: the bitmap fill plus the partners' listeners.
    part_cost = np.concatenate(([0], np.cumsum(pop[small_s])))
    gcost = pop[big_s[gstart[:-1]]] + part_cost[gstart[1:]] - part_cost[gstart[:-1]]
    tasks = _ranges(gcost, block_cost)
    costs = [int(gcost[g0:g1].sum()) for g0, g1 in tasks]
    _CTX.update(
        big=big_s,
        small=small_s,
        gstart=gstart,
        i_indptr=np.asarray(i_indptr, dtype=np.int64),
        i_indices=i_indices,
        n_users=int(n_users),
    )
    prog = Progress(
        "colisten: exact co-listener counts", max(sum(costs), 1), log, unit=" lookups"
    )
    try:
        parts = _run_tasks(_count_block, tasks, costs, _workers(workers), prog)
    finally:
        _CTX.clear()
    prog.finish()
    counts_sorted = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    co = np.empty(len(a), dtype=np.int64)
    co[order] = counts_sorted
    return co


# Neighbours


def cosine(co, pop_a, pop_b):
    """The one formula for the co-listening cosine, used everywhere."""
    return np.asarray(co, dtype=np.float64) / np.sqrt(
        np.asarray(pop_a, dtype=np.float64) * np.asarray(pop_b, dtype=np.float64)
    )


def _rank_take(row, nbr, cos, top_n):
    """Positions of each row's top_n entries: cosine descending, then
    neighbour id ascending. The positions come out in (row, rank) order."""
    order = np.lexsort((nbr, -cos, row))
    r = row[order]
    first = np.concatenate(([True], r[1:] != r[:-1]))
    start = np.maximum.accumulate(np.where(first, np.arange(len(r)), 0))
    return order[(np.arange(len(r)) - start) < top_n]


def _csr_from_rows(row, n_items):
    return np.concatenate(([0], np.cumsum(np.bincount(row, minlength=n_items)))).astype(
        np.int64
    )


def directed_topn(
    a,
    b,
    co,
    support_most,
    support_low,
    pop,
    n_items,
    top_n,
    min_support,
    log=None,
    n_buckets=16,
):
    """Each item's top_n neighbours among the undirected pairs (a, b),
    ranked by exact cosine descending, ties to the smaller neighbour id.
    Pairs with co < min_support take no part. Returns a dict of CSR arrays
    over item slots: indptr, neighbor, cosine, co, support_most,
    support_low."""
    log = log or _quiet
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    co = np.asarray(co, dtype=np.int64)
    ok = co >= min_support
    a, b, co = a[ok], b[ok], co[ok]
    sm = np.asarray(support_most)[ok]
    sl = np.asarray(support_low)[ok]
    cos = cosine(co, pop[a], pop[b])
    edges = np.linspace(0, n_items, n_buckets + 1).astype(np.int64)
    keep = {
        k: []
        for k in ("row", "neighbor", "cosine", "co", "support_most", "support_low")
    }
    prog = Progress(
        "colisten: top other-artist neighbours per track",
        n_buckets,
        log,
        unit=" buckets",
    )
    for r0, r1 in zip(edges[:-1], edges[1:]):
        m1 = (a >= r0) & (a < r1)
        m2 = (b >= r0) & (b < r1)
        row = np.concatenate((a[m1], b[m2]))
        if len(row) == 0:
            prog.advance(1)
            continue
        take = _rank_take(
            row,
            np.concatenate((b[m1], a[m2])),
            np.concatenate((cos[m1], cos[m2])),
            top_n,
        )
        for name, v1, v2 in (
            ("row", a, b),
            ("neighbor", b, a),
            ("cosine", cos, cos),
            ("co", co, co),
            ("support_most", sm, sm),
            ("support_low", sl, sl),
        ):
            keep[name].append(np.concatenate((v1[m1], v2[m2]))[take])
        prog.advance(1)
    prog.finish()
    row = _cat(keep["row"], np.int64)
    return {
        "indptr": _csr_from_rows(row, n_items),
        "neighbor": _cat(keep["neighbor"], np.int32),
        "cosine": _cat(keep["cosine"], np.float64),
        "co": _cat(keep["co"], np.int32),
        "support_most": _cat(keep["support_most"], np.int32),
        "support_low": _cat(keep["support_low"], np.int32),
    }


def rank_rows(
    row,
    nbr,
    co,
    pop,
    n_items,
    top_n,
    min_support,
    log=None,
    n_buckets=16,
    label="colisten: top neighbours per track",
):
    """Each row's top_n among directed entries (row, nbr, co), ranked as
    directed_topn() ranks. A repeated (row, nbr) entry counts once, and
    its copies must carry the same co. Returns a dict of CSR arrays over
    item slots: indptr, neighbor, cosine, co."""
    log = log or _quiet
    # Kept in their own (often int32) type, and widened per bucket, so a
    # long entry list costs 12 bytes an entry.
    row, nbr, co = np.asarray(row), np.asarray(nbr), np.asarray(co)
    ok = co >= min_support
    if not ok.all():
        row, nbr, co = row[ok], nbr[ok], co[ok]
    edges = np.linspace(0, n_items, n_buckets + 1).astype(np.int64)
    keep = {k: [] for k in ("row", "neighbor", "co")}
    prog = Progress(label, n_buckets, log, unit=" buckets")
    for r0, r1 in zip(edges[:-1], edges[1:]):
        m = (row >= r0) & (row < r1)
        r = row[m].astype(np.int64)
        n = nbr[m].astype(np.int64)
        c = co[m].astype(np.int64)
        if len(r):
            key, first = np.unique(r * n_items + n, return_index=True)
            inv = np.searchsorted(key, r * n_items + n)
            if not np.array_equal(c, c[first][inv]):
                raise RuntimeError("a repeated pair carries two counts")
            r, n, c = r[first], n[first], c[first]
            take = _rank_take(r, n, cosine(c, pop[r], pop[n]), top_n)
            keep["row"].append(r[take])
            keep["neighbor"].append(n[take])
            keep["co"].append(c[take])
        prog.advance(1)
    prog.finish()
    r = _cat(keep["row"], np.int64)
    n = _cat(keep["neighbor"], np.int64)
    c = _cat(keep["co"], np.int64)
    return {
        "indptr": _csr_from_rows(r, n_items),
        "neighbor": n.astype(np.int32),
        "cosine": cosine(c, pop[r], pop[n]),
        "co": c.astype(np.int32),
    }


def _same_block(task):
    a0, a1 = task
    c = _CTX
    a_indptr, a_items, pop = c["a_indptr"], c["a_items"], c["pop"]
    i_indptr, i_indices = c["i_indptr"], c["i_indices"]
    ms, top_n, budget = c["min_support"], c["top_n"], c["pair_budget"]
    out_r, out_n, out_c = [], [], []
    for art in c["artists"][a0:a1]:
        tracks = a_items[a_indptr[art] : a_indptr[art + 1]]
        n = len(tracks)
        idx, offs = _gather_index(i_indptr, tracks)
        uu, inv = np.unique(np.asarray(i_indices[idx]), return_inverse=True)
        M = sp.csr_matrix(
            (
                np.ones(len(idx), dtype=np.int32),
                np.asarray(inv, dtype=np.int64).ravel(),
                offs,
            ),
            shape=(n, len(uu)),
        )
        MT = M.T.tocsr()
        step = max(1, budget // n)
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            C = sp.csr_matrix(M[r0:r1] @ MT)
            lr = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(C.indptr))
            lc = C.indices.astype(np.int64)
            co = C.data.astype(np.int64)
            ok = (lc != lr) & (co >= ms)
            lr, lc, co = lr[ok], lc[ok], co[ok]
            # Local column order is item_id order, so ties break correctly.
            keep = top_per_row(
                _csr_from_rows(lr - r0, r1 - r0),
                lc,
                cosine(co, pop[tracks[lr]], pop[tracks[lc]]),
                top_n,
            )
            out_r.append(tracks[lr[keep]])
            out_n.append(tracks[lc[keep]])
            out_c.append(co[keep])
    return (_cat(out_r, np.int32), _cat(out_n, np.int32), _cat(out_c, np.int32))


def same_artist_lists(
    art_indptr,
    art_ids,
    i_indptr,
    i_indices,
    pop,
    min_support,
    top_n,
    workers=None,
    log=None,
    block_cost=20_000_000,
    pair_budget=10_000_000,
):
    """Every track's exact top_n same-artist neighbours. For each artist
    with two or more tracks of min_support or more listeners, one sparse
    product (tracks x listeners) @ (listeners x tracks) counts every pair
    of its tracks exactly. A track with several artists merges its lists.
    Returns a dict of CSR arrays over item slots: indptr, neighbor,
    cosine, co."""
    log = log or _quiet
    i_indptr = np.asarray(i_indptr, dtype=np.int64)
    pop = np.asarray(pop, dtype=np.int64)
    eligible = pop >= min_support
    eligible[0] = False
    a_indptr, a_items = artist_tracks(art_indptr, art_ids, eligible)
    artists = np.flatnonzero(np.diff(a_indptr) >= 2)
    # Work per artist: the listener entries of its tracks.
    tcost = np.concatenate(([0], np.cumsum(pop[a_items])))
    cost = tcost[a_indptr[artists + 1]] - tcost[a_indptr[artists]]
    tasks = _ranges(cost, block_cost)
    costs = [int(cost[t0:t1].sum()) for t0, t1 in tasks]
    _CTX.update(
        a_indptr=a_indptr,
        a_items=a_items,
        artists=artists,
        pop=pop,
        i_indptr=i_indptr,
        i_indices=i_indices,
        min_support=int(min_support),
        top_n=int(top_n),
        pair_budget=int(pair_budget),
    )
    prog = Progress(
        "colisten: exact same-artist counts, one artist at a " "time",
        max(sum(costs), 1),
        log,
        unit=" listener entries",
    )
    try:
        parts = _run_tasks(_same_block, tasks, costs, _workers(workers), prog)
    finally:
        _CTX.clear()
    prog.finish()
    return rank_rows(
        _cat([p[0] for p in parts], np.int32),
        _cat([p[1] for p in parts], np.int32),
        _cat([p[2] for p in parts], np.int32),
        pop,
        len(pop),
        top_n,
        min_support,
        log=log,
        label="colisten: top same-artist neighbours per track",
    )


# Validation


def exact_lists(
    x, u_indptr, u_indices, i_indptr, i_indices, pop, min_support, chunk=1 << 25
):
    """Track x against every other track, from all of x's train listeners,
    with no sampling. Returns (neighbours, co, cosine) for every track
    with co >= min_support, sorted by cosine descending, then by id."""
    x = int(x)
    n_items = len(pop)
    users = np.asarray(i_indices[i_indptr[x] : i_indptr[x + 1]]).astype(np.int64)
    lens = u_indptr[users + 1] - u_indptr[users]
    acc = np.zeros(n_items, dtype=np.int64)
    for g0, g1 in _ranges(lens, chunk):
        idx, _ = _gather_index(u_indptr, users[g0:g1])
        acc += np.bincount(np.asarray(u_indices[idx]), minlength=n_items)
    acc[x] = 0
    cand = np.flatnonzero(acc >= min_support)
    co = acc[cand]
    cos = cosine(co, pop[x], pop[cand])
    order = np.lexsort((cand, -cos))
    return cand[order].astype(np.int64), co[order], cos[order]


def exact_topn(
    items,
    u_indptr,
    u_indices,
    i_indptr,
    i_indices,
    top_n,
    min_support,
    log=None,
    chunk=1 << 25,
    art=None,
):
    """The exact top_n of each item over the whole catalog, with the same
    cosine, tie rule and support floor as the stored lists. Without art:
    {item: (neighbors, co, cosine)}. With art (a credit CSR): {item:
    {"same": (...), "other": (...)}}, the top_n among the tracks that do
    and do not share an artist with the item."""
    log = log or _quiet
    i_indptr = np.asarray(i_indptr, dtype=np.int64)
    u_indptr = np.asarray(u_indptr, dtype=np.int64)
    pop = np.diff(i_indptr)
    out = {}
    total = int(sum(int(pop[x]) for x in items))
    prog = Progress(
        "colisten: exact whole-catalog neighbours for the " "validation tracks",
        max(total, 1),
        log,
        unit=" listeners",
    )
    for x in items:
        x = int(x)
        nb, co, cos = exact_lists(
            x, u_indptr, u_indices, i_indptr, i_indices, pop, min_support, chunk
        )
        if art is None:
            out[x] = (nb[:top_n], co[:top_n], cos[:top_n])
        else:
            same = share_artist(np.full(len(nb), x), nb, *art)
            out[x] = {
                kind: (nb[m][:top_n], co[m][:top_n], cos[m][:top_n])
                for kind, m in (("same", same), ("other", ~same))
            }
        prog.advance(int(pop[x]))
    prog.finish()
    return out


def knn_problems(knn, pop, top_n, min_support):
    """Why a stored neighbour graph breaks the rules of directed_topn() and
    rank_rows(), as sentences. An empty list means it keeps every rule.
    The support checks apply when the graph has support columns."""
    indptr = np.asarray(knn["indptr"], dtype=np.int64)
    nbr = np.asarray(knn["neighbor"], dtype=np.int64)
    cos, co = knn["cosine"], np.asarray(knn["co"], dtype=np.int64)
    n_items = len(pop)
    problems = []
    if (
        len(indptr) != n_items + 1
        or indptr[0] != 0
        or np.any(np.diff(indptr) < 0)
        or indptr[-1] != len(nbr)
    ):
        return ["indptr does not describe the stored rows"]
    row = np.repeat(np.arange(n_items), np.diff(indptr))
    if np.any(np.diff(indptr) > top_n):
        problems.append(f"a track holds more than {top_n} neighbours")
    if np.any((nbr < 1) | (nbr >= n_items)):
        problems.append("a neighbour id falls outside the item slots")
        return problems
    if np.any(nbr == row):
        problems.append("a track lists itself as a neighbour")
    same = row[1:] == row[:-1]
    in_order = (cos[:-1] > cos[1:]) | ((cos[:-1] == cos[1:]) & (nbr[:-1] < nbr[1:]))
    if np.any(same & ~in_order):
        problems.append(
            "a row is not sorted by cosine descending, then by " "neighbour id"
        )
    if np.any(co < min_support):
        problems.append(f"a pair has fewer than {min_support} co-listeners")
    if not np.array_equal(cos, cosine(co, pop[row], pop[nbr])):
        problems.append("a stored cosine differs from co / sqrt(pop_a pop_b)")
    if "support_most" in knn:
        sm, sl = knn["support_most"], knn["support_low"]
        if np.any(sm > co) or np.any(sl > co):
            problems.append("an in-sample support exceeds the exact count")
        if np.any(np.maximum(sm, sl) < min_support):
            problems.append("a pair came from no sample with enough support")
    return problems


def undirected_csr(a, b, n_items):
    """Both directions of undirected pairs (a, b) as CSR arrays over item
    slots, rows sorted by neighbour id."""
    rows = np.concatenate([a, b]).astype(np.int64)
    cols = np.concatenate([b, a]).astype(np.int64)
    order = np.lexsort((cols, rows))
    return _csr_from_rows(rows, n_items), cols[order]


def pick_hits(indptr, neighbors, u_indptr, u_indices, users, picks):
    """hits[k, j]: how many of pick j's neighbours are in user k's train
    history. -1 where the pick is missing (item 0). For a directed graph,
    a pick's neighbours are its own row."""
    hits = np.full(picks.shape, -1, dtype=np.int64)
    for k, u in enumerate(users):
        s, e = int(u_indptr[u]), int(u_indptr[u + 1])
        hist = np.asarray(u_indices[s:e]).astype(np.int64)
        if len(hist) > 1 and np.any(hist[1:] < hist[:-1]):
            hist = np.sort(hist)
        for j in range(picks.shape[1]):
            p = int(picks[k, j])
            if p <= 0:
                continue
            nb = np.asarray(neighbors[indptr[p] : indptr[p + 1]]).astype(np.int64)
            if len(nb) == 0 or len(hist) == 0:
                hits[k, j] = 0
                continue
            pos = np.minimum(np.searchsorted(hist, nb), len(hist) - 1)
            hits[k, j] = int(np.count_nonzero(hist[pos] == nb))
    return hits
