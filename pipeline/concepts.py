"""Concept layer: the named, evidence-bearing features the fusion model reads.

One definition per concept, two computation paths that must agree:

- full_catalog(): one user, every item slot. P14 ranks with this path.
- pairs(): one user, a short candidate list. P12 builds training rows
  with this path.

test_concepts.py checks both paths against an independent brute-force
implementation. It also checks that every evidence list sums to its
value (to floating-point rounding), so an explanation never cites
numbers that the model did not use.

Missing is not zero. Every continuous concept returns a value array and a
has_data mask. A value of 0.0 with has_data True means "the data exists
and shows no match". has_data False means "no data for this track", and
the model gives that state its own learned value.

Inputs are plain arrays, so this module has no DuckDB dependency and no
file paths. P12's loader builds ItemSide from P3, P4, P9b's two neighbour
lists and the feature tables, with the artist credits of
pipeline/artists.py for artist_affinity, so that "artist" means the same
in every concept. It builds each user's history from P6's user CSR.

Notation: w_h = log1p(listens) of history item h, W = sum over h of w_h.

- tag_affinity, artist_tag_affinity, genre_affinity: TF-IDF cosine.
  User side p(t) = sum_h w_h x_ht / W, the listening share that carries
  term t. Item side x_it. Both sides use smooth IDF. Evidence: one term
  per shared term, idf_t^2 x_it p(t) / (|v_i| |u|). The terms sum to the
  cosine (to rounding). Missing when the item has no terms or the user's
  history has none.
- artist_affinity, work_affinity: s(a) = sum_h w_h C_ha / W, the share of
  listening that credits artist (or work) a. Value: the largest s(a) over
  the item's own credits. Evidence: that credit and the history tracks
  that carry it. Their w_h / W sum to the value (to rounding). Missing when the
  item has no credit.
- colisten_same_artist, colisten_other_artists: sum_h S_ih w_h / W, where
  row i of S holds track i's own neighbours of that kind from P9b with
  their exact co-listening cosine (the item-based CF rule of Sarwar et
  al. 2001, scaled by W so users of different activity compare). A
  same-artist neighbour shares an artist credit with i, an other-artist
  neighbour shares none (pipeline/artists.py). S is directed: h can be
  among i's neighbours while i is not among h's. Evidence: one term per
  history track among i's neighbours of that kind. Missing when i has no
  neighbour of that kind. The same-artist concept ranks the tracks of an
  artist the user already plays, which artist_affinity cannot do. The
  other-artist concept is the only one that reaches past the user's
  artists through other listeners. One combined concept would mostly
  repeat artist_affinity: for 91 percent of tracks the closest
  co-listened track is by the same artist (measured on the real
  data).
- audio_fit: minus the mean, over the dimensions both sides have, of
  ((a_id - mu_d) / sigma_d)^2. mu and sigma are the user's
  listen-weighted audio mean and spread, with sigma floored at
  SIGMA_FLOOR. Evidence: one term per dimension. Missing when the item
  has no audio or the user's history has none.
- popularity: log1p(train listens). Never missing. No evidence terms.
- match_tier: the bridge tier (categorical). Never missing.

Slot 0 is the phantom item slot. It is missing for every concept.
"""

from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

TERM_CONCEPTS = {
    "tag_affinity": "tags_direct",
    "artist_tag_affinity": "tags_artist",
    "genre_affinity": "genres",
}
CREDIT_CONCEPTS = {
    "artist_affinity": "artists",
    "work_affinity": "works",
}
COLISTEN_CONCEPTS = {
    "colisten_same_artist": "same_artist",
    "colisten_other_artists": "other_artists",
}
CONTINUOUS = (
    list(TERM_CONCEPTS)
    + list(CREDIT_CONCEPTS)
    + list(COLISTEN_CONCEPTS)
    + ["audio_fit", "popularity"]
)
NEVER_MISSING = {"popularity"}
MATCH_TIERS = ["isrc", "isrc+url", "url", "fuzzy_dur", "fuzzy_name", "unmatched"]
AUDIO_DIMS = [
    "z_duration_ms",
    "z_tempo",
    "z_danceability",
    "z_energy",
    "z_loudness",
    "z_speechiness",
    "z_acousticness",
    "z_instrumentalness",
    "z_liveness",
    "z_valence",
]
# In z-units. Stops a very consistent listener's tiny spread from turning
# small audio differences into huge distances.
SIGMA_FLOOR = 0.25


def smooth_idf(m: sp.csr_matrix) -> np.ndarray:
    """ln((1 + N) / (1 + df)) + 1, the common smooth form. N counts the
    items that carry at least one term. df counts items per term."""
    df = np.bincount(m.indices, minlength=m.shape[1]).astype(np.float64)
    n_docs = float(np.count_nonzero(np.diff(m.indptr)))
    return np.log((1.0 + n_docs) / (1.0 + df)) + 1.0


def _row_max(values_at_nnz, indptr):
    """Maximum of each CSR row's gathered values. Rows without entries
    get 0.0. The caller masks them as missing."""
    n_rows = len(indptr) - 1
    out = np.zeros(n_rows, dtype=np.float64)
    nonempty = np.flatnonzero(np.diff(indptr) > 0)
    if len(nonempty):
        out[nonempty] = np.maximum.reduceat(values_at_nnz, indptr[:-1][nonempty])
    return out


def _row_sum(values_at_nnz, indptr):
    n_rows = len(indptr) - 1
    out = np.zeros(n_rows, dtype=np.float64)
    nonempty = np.flatnonzero(np.diff(indptr) > 0)
    if len(nonempty):
        out[nonempty] = np.add.reduceat(values_at_nnz, indptr[:-1][nonempty])
    return out


@dataclass
class ItemSide:
    """Everything item-side, built once and shared by every user.

    term_mats and credit_mats map a source name to a CSR matrix with one
    row per item slot. colisten maps each value of COLISTEN_CONCEPTS to a
    CSR matrix with one row per item slot, and row i holds track i's own
    neighbours of that kind (directed, as P9b stores them). audio holds
    NaN where a value is absent. tier holds codes into MATCH_TIERS.

    transposed=False skips the transposed co-listening matrices. Only
    full_catalog() reads them, and for P9b's lists they cost about 2 GB.
    pairs() and evidence() do not need them."""

    term_mats: dict
    credit_mats: dict
    colisten: dict
    audio: np.ndarray
    popularity: np.ndarray
    tier: np.ndarray
    idf: dict = field(default_factory=dict)
    transposed: bool = True

    def __post_init__(self):
        self.n_slots = int(self.audio.shape[0])
        for name, m in list(self.term_mats.items()):
            m = sp.csr_matrix(m, dtype=np.float64)
            m.sum_duplicates()
            m.sort_indices()
            self.term_mats[name] = m
            if name not in self.idf:
                self.idf[name] = smooth_idf(m)
        for name, m in list(self.credit_mats.items()):
            m = sp.csr_matrix(m, dtype=np.float64)
            m.sum_duplicates()
            m.sort_indices()
            m.data[:] = 1.0  # a credit is present or absent
            self.credit_mats[name] = m
        self.colisten = dict(self.colisten)
        missing = set(COLISTEN_CONCEPTS.values()) - set(self.colisten)
        if missing:
            raise ValueError(f"colisten lacks {sorted(missing)}")
        self.colisten_t = {}
        for name in COLISTEN_CONCEPTS.values():
            s = sp.csr_matrix(self.colisten[name], dtype=np.float64)
            s.sum_duplicates()
            s.sort_indices()
            if s.shape != (self.n_slots, self.n_slots):
                raise ValueError(
                    f"colisten {name} has shape {s.shape}, "
                    f"the catalog has {self.n_slots} slots"
                )
            self.colisten[name] = s
            # Row h of the transpose lists the tracks whose neighbour
            # lists hold h. full_catalog() sums those rows over the
            # history.
            if self.transposed:
                self.colisten_t[name] = s.T.tocsr()
        self.audio = np.asarray(self.audio, dtype=np.float64)
        self.popularity = np.asarray(self.popularity, dtype=np.float64)
        self.tier = np.asarray(self.tier, dtype=np.int64)
        for arr, what in [(self.popularity, "popularity"), (self.tier, "tier")]:
            if len(arr) != self.n_slots:
                raise ValueError(
                    f"{what} has {len(arr)} slots, audio has " f"{self.n_slots}"
                )

        # Derived arrays. Slot 0 is forced to "no data" everywhere.
        self.item_norm = {}
        self.has_terms = {}
        for name, m in self.term_mats.items():
            idf2 = self.idf[name] ** 2
            self.item_norm[name] = np.sqrt(m.multiply(m) @ idf2)
            has = np.diff(m.indptr) > 0
            has[0] = False
            self.has_terms[name] = has
        self.has_credit = {}
        for name, m in self.credit_mats.items():
            has = np.diff(m.indptr) > 0
            has[0] = False
            self.has_credit[name] = has
        self.has_colisten = {}
        for name, s in self.colisten.items():
            has = np.diff(s.indptr) > 0
            has[0] = False
            self.has_colisten[name] = has
        self.audio_mask = ~np.isnan(self.audio)
        self.audio_mask[0] = False
        self.audio_filled = np.where(self.audio_mask, self.audio, 0.0)


@dataclass
class UserState:
    """One user's train history, reduced to what the concepts read."""

    items: np.ndarray
    weights: np.ndarray
    total: float
    term_share: dict  # source -> dense p(t)
    term_norm: dict  # source -> |u|
    credit_cols: dict  # source -> sorted credit columns in the history
    credit_share: dict  # source -> s(a) for those columns
    audio_mu: np.ndarray
    audio_sigma: np.ndarray
    audio_dim_ok: np.ndarray


def build_user_state(items, weights, space: ItemSide) -> UserState:
    """items: the user's train item ids. weights: log1p(listens), > 0."""
    items = np.asarray(items, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if len(items) != len(weights):
        raise ValueError("items and weights differ in length")
    if len(items) == 0:
        raise ValueError("a user state needs at least one history item")
    order = np.argsort(items, kind="stable")
    items, weights = items[order], weights[order]
    if np.any(items[1:] == items[:-1]):
        raise ValueError("duplicate history items")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("history weights must be finite and positive")
    if items[0] < 1 or items[-1] >= space.n_slots:
        raise ValueError("a history item falls outside the slot range")
    total = float(weights.sum())

    term_share, term_norm = {}, {}
    for name, m in space.term_mats.items():
        p = (m[items].T @ weights) / total
        term_share[name] = np.asarray(p, dtype=np.float64).ravel()
        term_norm[name] = float(
            np.sqrt(np.sum((space.idf[name] * term_share[name]) ** 2))
        )

    credit_cols, credit_share = {}, {}
    for name, m in space.credit_mats.items():
        sub = m[items]
        row_w = np.repeat(weights, np.diff(sub.indptr))
        cols, inv = np.unique(sub.indices, return_inverse=True)
        credit_cols[name] = cols.astype(np.int64)
        credit_share[name] = (
            np.bincount(inv, weights=row_w, minlength=len(cols)) / total
        )

    mask = space.audio_mask[items]
    vals = space.audio_filled[items]
    w_d = weights @ mask  # (D,)
    ok = w_d > 0
    safe = np.where(ok, w_d, 1.0)
    mu = (weights @ (vals * mask)) / safe
    second = (weights @ (vals * vals * mask)) / safe
    sigma = np.sqrt(np.maximum(second - mu * mu, 0.0))
    sigma = np.maximum(sigma, SIGMA_FLOOR)
    return UserState(
        items,
        weights,
        total,
        term_share,
        term_norm,
        credit_cols,
        credit_share,
        np.where(ok, mu, 0.0),
        np.where(ok, sigma, 1.0),
        ok,
    )


def _credit_lookup(state: UserState, name, cols):
    """s(a) for arbitrary credit columns. 0.0 where the history lacks a."""
    ucols = state.credit_cols[name]
    shares = state.credit_share[name]
    if len(ucols) == 0:
        return np.zeros(len(cols), dtype=np.float64)
    pos = np.searchsorted(ucols, cols)
    pos_c = np.minimum(pos, len(ucols) - 1)
    found = ucols[pos_c] == cols
    return np.where(found, shares[pos_c], 0.0)


def _history_weight(state: UserState, item_ids):
    """w_h for arbitrary items. 0.0 where the item is not in the history."""
    pos = np.searchsorted(state.items, item_ids)
    pos_c = np.minimum(pos, len(state.items) - 1)
    found = state.items[pos_c] == item_ids
    return np.where(found, state.weights[pos_c], 0.0)


def _audio_fit(state: UserState, space: ItemSide, rows):
    """rows: an index into the slots, or slice(None) for all of them."""
    both = space.audio_mask[rows] & state.audio_dim_ok[None, :]
    z = (space.audio_filled[rows] - state.audio_mu[None, :]) / state.audio_sigma[
        None, :
    ]
    sq = np.where(both, z * z, 0.0)
    n = both.sum(axis=1)
    has = n > 0
    value = np.where(has, -sq.sum(axis=1) / np.maximum(n, 1), np.nan)
    return value, has


def full_catalog(state: UserState, space: ItemSide) -> dict:
    """Every concept for every item slot.

    Returns {concept: (value, has_data)} for CONTINUOUS, and
    {"match_tier": codes}. Values are float64 arrays of length n_slots,
    NaN where has_data is False."""
    if not space.transposed:
        raise ValueError(
            "full_catalog() needs an ItemSide built with " "transposed=True"
        )
    out = {}
    for concept, name in TERM_CONCEPTS.items():
        m = space.term_mats[name]
        coef = space.idf[name] ** 2 * state.term_share[name]
        numer = m @ coef
        has = space.has_terms[name] & (state.term_norm[name] > 0)
        denom = np.where(has, space.item_norm[name] * state.term_norm[name], 1.0)
        out[concept] = (np.where(has, numer / denom, np.nan), has)

    for concept, name in CREDIT_CONCEPTS.items():
        m = space.credit_mats[name]
        dense = np.zeros(m.shape[1], dtype=np.float64)
        dense[state.credit_cols[name]] = state.credit_share[name]
        best = _row_max(dense[m.indices], m.indptr)
        has = space.has_credit[name]
        out[concept] = (np.where(has, best, np.nan), has)

    for concept, name in COLISTEN_CONCEPTS.items():
        # sum_h S[i, h] w_h for every i, read through the transpose.
        numer = np.asarray(
            space.colisten_t[name][state.items].T @ state.weights, dtype=np.float64
        ).ravel()
        has = space.has_colisten[name]
        out[concept] = (np.where(has, numer / state.total, np.nan), has)

    out["audio_fit"] = _audio_fit(state, space, slice(None))

    has = np.ones(space.n_slots, dtype=bool)
    has[0] = False
    out["popularity"] = (np.where(has, space.popularity, np.nan), has)
    out["match_tier"] = space.tier.copy()
    return out


def pairs(state: UserState, space: ItemSide, candidates) -> dict:
    """The same concepts as full_catalog(), for a candidate list only.
    Must equal full_catalog() at the same slots. test_concepts.py
    checks this."""
    cand = np.asarray(candidates, dtype=np.int64)
    out = {}
    for concept, name in TERM_CONCEPTS.items():
        m = space.term_mats[name]
        coef = space.idf[name] ** 2 * state.term_share[name]
        numer = m[cand] @ coef
        has = space.has_terms[name][cand] & (state.term_norm[name] > 0)
        denom = np.where(has, space.item_norm[name][cand] * state.term_norm[name], 1.0)
        out[concept] = (np.where(has, numer / denom, np.nan), has)

    for concept, name in CREDIT_CONCEPTS.items():
        sub = space.credit_mats[name][cand]
        g = _credit_lookup(state, name, sub.indices.astype(np.int64))
        best = _row_max(g, sub.indptr)
        has = space.has_credit[name][cand]
        out[concept] = (np.where(has, best, np.nan), has)

    for concept, name in COLISTEN_CONCEPTS.items():
        sub = space.colisten[name][cand]
        w = _history_weight(state, sub.indices.astype(np.int64))
        numer = _row_sum(sub.data * w, sub.indptr)
        has = space.has_colisten[name][cand]
        out[concept] = (np.where(has, numer / state.total, np.nan), has)

    out["audio_fit"] = _audio_fit(state, space, cand)

    has = cand != 0
    out["popularity"] = (np.where(has, space.popularity[cand], np.nan), has)
    out["match_tier"] = space.tier[cand].copy()
    return out


def evidence(state: UserState, space: ItemSide, item: int) -> dict:
    """The exact terms behind every concept value for one item.

    Returns {concept: {"value", "has_data", "terms", ...}}. For every
    concept with has_data True and a terms list, the terms sum to the
    value (checked in test_concepts.py). Terms are ordered by size, then
    by id, so the order is deterministic."""
    item = int(item)
    vals = pairs(state, space, np.array([item]))
    out = {}

    for concept, name in TERM_CONCEPTS.items():
        value, has = vals[concept][0][0], bool(vals[concept][1][0])
        entry = {"value": value, "has_data": has, "unit": name, "terms": []}
        if has:
            m = space.term_mats[name]
            s, e = m.indptr[item], m.indptr[item + 1]
            cols, x = m.indices[s:e], m.data[s:e]
            p = state.term_share[name][cols]
            denom = space.item_norm[name][item] * state.term_norm[name]
            t = space.idf[name][cols] ** 2 * x * p / denom
            keep = p > 0
            rows = sorted(
                zip(cols[keep].tolist(), t[keep].tolist(), p[keep].tolist()),
                key=lambda r: (-r[1], r[0]),
            )
            entry["terms"] = [
                {"id": c, "term": tv, "listening_share": pv} for c, tv, pv in rows
            ]
        out[concept] = entry

    for concept, name in CREDIT_CONCEPTS.items():
        value, has = vals[concept][0][0], bool(vals[concept][1][0])
        entry = {
            "value": value,
            "has_data": has,
            "unit": name,
            "credit": None,
            "terms": [],
        }
        if has:
            m = space.credit_mats[name]
            s, e = m.indptr[item], m.indptr[item + 1]
            cols = m.indices[s:e].astype(np.int64)
            shares = _credit_lookup(state, name, cols)
            # Largest share wins. Ties go to the smallest column id.
            order = np.lexsort((cols, -shares))
            best = int(cols[order[0]])
            entry["credit"] = best
            hist_rows = m[state.items]
            carries = np.flatnonzero((hist_rows[:, [best]].toarray().ravel()) > 0)
            rows = sorted(
                (
                    (int(state.items[k]), float(state.weights[k] / state.total))
                    for k in carries
                ),
                key=lambda r: (-r[1], r[0]),
            )
            entry["terms"] = [{"id": h, "term": tv} for h, tv in rows]
        out[concept] = entry

    for concept, name in COLISTEN_CONCEPTS.items():
        value, has = vals[concept][0][0], bool(vals[concept][1][0])
        entry = {"value": value, "has_data": has, "unit": "tracks", "terms": []}
        if has:
            s_row = space.colisten[name][item]
            nbrs = s_row.indices.astype(np.int64)
            w = _history_weight(state, nbrs)
            t = s_row.data * w / state.total
            keep = w > 0
            rows = sorted(
                zip(nbrs[keep].tolist(), t[keep].tolist()), key=lambda r: (-r[1], r[0])
            )
            entry["terms"] = [{"id": h, "term": tv} for h, tv in rows]
        out[concept] = entry

    value = vals["audio_fit"][0][0]
    has = bool(vals["audio_fit"][1][0])
    entry = {"value": value, "has_data": has, "unit": "audio_dims", "terms": []}
    if has:
        both = space.audio_mask[item] & state.audio_dim_ok
        n = int(both.sum())
        z = (space.audio_filled[item] - state.audio_mu) / state.audio_sigma
        rows = []
        for d in np.flatnonzero(both):
            rows.append(
                {"id": AUDIO_DIMS[d], "term": float(-(z[d] ** 2) / n), "z": float(z[d])}
            )
        rows.sort(key=lambda r: (r["term"], r["id"]))  # worst fit first
        entry["terms"] = rows
    out["audio_fit"] = entry

    out["popularity"] = {
        "value": vals["popularity"][0][0],
        "has_data": bool(vals["popularity"][1][0]),
        "unit": "log1p_listens",
        "terms": [],
    }
    out["match_tier"] = {"value": MATCH_TIERS[int(vals["match_tier"][0])]}
    return out
