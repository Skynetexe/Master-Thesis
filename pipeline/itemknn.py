"""ItemKNN, the co-listening baseline: the item-based rule of Sarwar,
Karypis, Konstan & Riedl (2001) on P9b's two directed neighbour lists.

score(u, i) = sum over the history tracks h of u that are among track i's
neighbours of cosine(i, h) * w_h, with w_h = log1p(listens). P6 stores
round(1000 * log1p(listens)) as uint16, so w_h = data / 1000, the weight
of concepts.py and of the ALS baseline. Row i of each P9b list holds
track i's own neighbours with their exact co-listening cosine.

This score is the numerator of the GAM's two co-listening concepts
(colisten_same_artist and colisten_other_artists), summed. The concepts
divide by the user's total weight W. That does not change the order of
one user's candidates, so ItemKNN ranks with the numerator. ItemKNN thus
ranks by exactly the co-listening evidence that the GAM reads, without
learned shapes and without the other concepts. The gap between the two
measures what the GAM adds.

Ranking follows the P5 harness: every item slot except slot 0 and the
user's train items is a candidate, and held-out picks stay candidates.
Two tie rules:
- "id": equal scores rank by item_id ascending (P5's rule).
- "pop": equal scores rank by train listens descending, then by item_id.
A candidate without a neighbour in the history scores exactly 0, so
under "pop" such candidates follow in popularity order, a common
fallback.

A pick that is also a train item cannot be a candidate. It gets the rank
n_slots, the rule of als.rank_items(), so both baselines treat it alike.
"""

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .als import Progress

KINDS = ("same_artist", "other_artists")
LIST_FILES = {
    "same_artist": "colisten_same_artist.npz",
    "other_artists": "colisten_other_artists.npz",
}
WEIGHT_SCALE = 1000.0  # as concept_data.WEIGHT_SCALE
# (lists, tie rule) of each variant. "both" sums the two lists.
VARIANTS = (
    ("both", "id"),
    ("both", "pop"),
    ("same_artist", "id"),
    ("other_artists", "id"),
)
USERS_PER_TASK = 32


def variant_name(lists, tie):
    return f"{lists}/{tie}"


# Item side


def load_lists(model_dir, n_slots):
    """{kind: (indptr int64, neighbor int32, cosine float64)} from P9b's
    files, with their shapes checked against the item slots."""
    out = {}
    for kind in KINDS:
        z = np.load(Path(model_dir) / LIST_FILES[kind])
        indptr = np.asarray(z["indptr"], dtype=np.int64)
        nbr = np.asarray(z["neighbor"], dtype=np.int32)
        cos = np.asarray(z["cosine"], dtype=np.float64)
        if len(indptr) != n_slots + 1:
            raise ValueError(
                f"{LIST_FILES[kind]} has {len(indptr) - 1} rows, "
                f"the catalog has {n_slots} slots"
            )
        if len(nbr) != int(indptr[-1]) or len(cos) != int(indptr[-1]):
            raise ValueError(
                f"{LIST_FILES[kind]}: neighbor and cosine do not " f"match indptr"
            )
        if len(nbr) and (int(nbr.min()) < 1 or int(nbr.max()) >= n_slots):
            raise ValueError(
                f"{LIST_FILES[kind]}: a neighbour lies outside " f"the item slots"
            )
        if len(cos) and not (np.all(np.isfinite(cos)) and cos.min() > 0.0):
            raise ValueError(
                f"{LIST_FILES[kind]}: a cosine is not a finite " f"positive number"
            )
        out[kind] = (indptr, nbr, cos)
    return out


def _small_indptr(indptr, n_entries):
    """int32 when it fits, as concept_data._small_indptr."""
    if n_entries < np.iinfo(np.int32).max:
        return np.asarray(indptr, dtype=np.int32)
    return np.asarray(indptr, dtype=np.int64)


def transposed(lists, n_slots):
    """{kind: CSR (n_slots x n_slots)}. Row h lists the tracks i that hold
    h among their neighbours, with cosine(i, h). These are the matrices of
    concepts.ItemSide.colisten_t, built by the same steps, so a score here
    equals the GAM's co-listening numerator bit for bit."""
    out = {}
    for kind in KINDS:
        indptr, nbr, cos = lists[kind]
        # Copies: sort_indices() below works in place, and the caller's
        # lists stay as loaded (brute_scores() reads them).
        s = sp.csr_matrix(
            (
                np.array(cos, dtype=np.float64, copy=True),
                np.array(nbr, dtype=np.int32, copy=True),
                _small_indptr(indptr, len(nbr)),
            ),
            shape=(n_slots, n_slots),
        )
        s = sp.csr_matrix(s, dtype=np.float64)
        s.sum_duplicates()
        s.sort_indices()
        out[kind] = s.T.tocsr()
    return out


# Per user


def history(u_indptr, u_indices, u_data, user):
    """(items, weights) of one user from P6's user-major CSR, sorted by
    item, as concepts.build_user_state sorts them."""
    s, e = int(u_indptr[user]), int(u_indptr[user + 1])
    items = np.asarray(u_indices[s:e], dtype=np.int64)
    weights = np.asarray(u_data[s:e], dtype=np.float64) / WEIGHT_SCALE
    order = np.argsort(items, kind="stable")
    return items[order], weights[order]


def user_scores(t_mats, items, weights):
    """{kind: numerator for every item slot} for one history, and "both",
    their sum. items must be sorted and weights positive."""
    out = {}
    for kind in KINDS:
        out[kind] = np.asarray(
            t_mats[kind][items].T @ weights, dtype=np.float64
        ).ravel()
    out["both"] = out[KINDS[0]] + out[KINDS[1]]
    return out


def candidate_scores(score, train_items):
    """A copy of score with -inf on the non-candidates: slot 0 and the
    user's train items."""
    s = np.array(score, dtype=np.float64)
    s[0] = -np.inf
    s[np.asarray(train_items, dtype=np.int64)] = -np.inf
    return s


def pick_ranks(s, picks, tie="id", pop=None):
    """Rank of each pick among the candidates of s (-inf marks a
    non-candidate). "id": 1 + #(s > s_p) + #(s[:p] == s_p). "pop": ties
    by pop descending, then by item_id. A pick that is not a candidate
    gets len(s)."""
    out = np.empty(len(picks), dtype=np.int64)
    for k, p in enumerate(np.asarray(picks, dtype=np.int64)):
        p = int(p)
        sp_ = s[p]
        if not np.isfinite(sp_):
            out[k] = len(s)
            continue
        above = int(np.count_nonzero(s > sp_))
        if tie == "id":
            tied = int(np.count_nonzero(s[:p] == sp_))
        elif tie == "pop":
            eq = s == sp_
            pp = pop[p]
            tied = int(np.count_nonzero(eq & (pop > pp))) + int(
                np.count_nonzero(eq[:p] & (pop[:p] == pp))
            )
        else:
            raise ValueError(f"unknown tie rule {tie!r}")
        out[k] = 1 + above + tied
    return out


def user_ranks(t_mats, items, weights, picks, pop):
    """{variant: ranks of the picks} for one user, and the number of
    candidates with a positive "both" score."""
    scores = user_scores(t_mats, items, weights)
    out = {}
    positive = 0
    for lists in dict.fromkeys(v[0] for v in VARIANTS):
        s = candidate_scores(scores[lists], items)
        if lists == "both":
            positive = int(np.count_nonzero(s > 0.0))
        for tie in (v[1] for v in VARIANTS if v[0] == lists):
            out[(lists, tie)] = pick_ranks(s, picks, tie, pop)
    return out, positive


# All users
_CTX = {}


def _task(task):
    lo, hi = task
    c = _CTX
    ranks = {
        v: np.empty((hi - lo, c["picks"].shape[1]), dtype=np.int64) for v in VARIANTS
    }
    positive = np.empty(hi - lo, dtype=np.int64)
    for j, k in enumerate(range(lo, hi)):
        items, weights = history(
            c["u_indptr"], c["u_indices"], c["u_data"], int(c["users"][k])
        )
        if len(items) == 0:
            raise ValueError(f"user {int(c['users'][k])} has no train items")
        r, positive[j] = user_ranks(
            c["t_mats"], items, weights, c["picks"][k], c["pop"]
        )
        for v in VARIANTS:
            ranks[v][j] = r[v]
    return ranks, positive


def rank_all(
    t_mats,
    u_arrays,
    users,
    picks,
    pop,
    workers=None,
    log=None,
    label="itemknn: ranking the picks",
):
    """({variant: (n, k) ranks}, (n,) positive-candidate counts) for every
    user. The result does not depend on the worker count: each user is
    computed alone, and the parts join in task order."""
    users = np.asarray(users, dtype=np.int64)
    picks = np.asarray(picks, dtype=np.int64)
    if picks.ndim != 2 or len(picks) != len(users):
        raise ValueError("picks must hold one row per user")
    workers = int(workers or max(1, min(16, os.cpu_count() or 1)))
    _CTX.update(
        t_mats=t_mats,
        u_indptr=u_arrays[0],
        u_indices=u_arrays[1],
        u_data=u_arrays[2],
        users=users,
        picks=picks,
        pop=np.asarray(pop),
    )
    n = len(users)
    tasks = [(lo, min(lo + USERS_PER_TASK, n)) for lo in range(0, n, USERS_PER_TASK)]
    prog = Progress(label, n, log or (lambda msg: None), unit=" users")
    parts = []
    try:
        if workers <= 1 or len(tasks) <= 1:
            for task in tasks:
                parts.append(_task(task))
                prog.advance(task[1] - task[0])
        else:
            with ProcessPoolExecutor(
                max_workers=workers, mp_context=mp.get_context("fork")
            ) as pool:
                for task, part in zip(tasks, pool.map(_task, tasks)):
                    parts.append(part)
                    prog.advance(task[1] - task[0])
    finally:
        _CTX.clear()
    prog.finish()
    ranks = {
        v: (
            np.concatenate([p[0][v] for p in parts])
            if parts
            else np.empty((0, picks.shape[1]), dtype=np.int64)
        )
        for v in VARIANTS
    }
    positive = (
        np.concatenate([p[1] for p in parts]) if parts else np.empty(0, dtype=np.int64)
    )
    return ranks, positive


# Reference


def brute_scores(lists, items, weights, n_slots):
    """{kind: score for every slot} and "both", by a second route that
    shares no code with user_scores(): a dense weight per slot, gathered
    row by row over the untransposed lists. Sums run in a different
    order, so values agree to rounding, not bit for bit."""
    w = np.zeros(n_slots, dtype=np.float64)
    w[np.asarray(items, dtype=np.int64)] = np.asarray(weights, dtype=np.float64)
    out = {}
    for kind in KINDS:
        indptr, nbr, cos = lists[kind]
        vals = np.asarray(cos, dtype=np.float64) * w[np.asarray(nbr, dtype=np.int64)]
        score = np.zeros(n_slots, dtype=np.float64)
        rows = np.flatnonzero(np.diff(indptr) > 0)
        if len(rows):
            score[rows] = np.add.reduceat(vals, indptr[:-1][rows])
        out[kind] = score
    out["both"] = out[KINDS[0]] + out[KINDS[1]]
    return out


def rank_bounds(s, p, tol):
    """(lo, hi): the rank of pick p under any tie rule lies in [lo, hi]
    when scores within tol of s[p] count as possible ties. -inf marks a
    non-candidate."""
    sp_ = s[p]
    lo = 1 + int(np.count_nonzero(s > sp_ + tol))
    hi = int(np.count_nonzero(s >= sp_ - tol))
    return lo, hi


def sorted_rank(s, p, tie="id", pop=None):
    """The rank of pick p by a full sort of the candidates, a route that
    shares no code with pick_ranks()."""
    cand = np.flatnonzero(np.isfinite(s))
    if tie == "id":
        order = np.lexsort((cand, -s[cand]))
    else:
        order = np.lexsort((cand, -np.asarray(pop)[cand], -s[cand]))
    pos = np.flatnonzero(cand[order] == p)
    return int(pos[0]) + 1 if len(pos) else len(s)
