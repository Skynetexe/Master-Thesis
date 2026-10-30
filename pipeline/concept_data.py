"""The concept layer on the real data.

pipeline/concepts.py stays pure: arrays in, arrays out, no files. This
module reads the pipeline's files into Sources, builds concepts.ItemSide
from them, and holds the per-user step that P12 runs for training pairs
and P14 runs for ranking. One code path for both is the point: the
features the model learns from are computed exactly as the features it
ranks with. Breck et al. (2017) list this as an ML test: "training and
serving features compute the same values".

It also holds Reference, a second, slow implementation of every concept
with plain loops and dicts. It reads Sources directly and shares no code
with concepts.py. P12's verify() recomputes sampled pairs with it on the
real data.

Sources, as read by load_sources():
- terms: P4's item_tags_direct.npz, item_tags_artist.npz, item_genres.npz,
  binary item x term CSR arrays, and the term names from vocab_tags and
  vocab_genres.
- credits: "artists" from pipeline/artists.py (the one artist rule), and
  "works" from the features table item_works (work_id, a MusicBrainz
  work entity, so two works that share a title such as "Intro" stay
  apart, as stage_p7.py's docstring explains). Both are compacted to
  column indices 0..n-1, with credit_ids mapping each column back to its
  MusicBrainz id for the explanation text.
- colisten: P9b's two directed neighbour lists with their exact cosines.
- audio: P4's item_audio_std.parquet, the ten z-scored dimensions of
  concepts.AUDIO_DIMS, NaN where absent (tempo is NULL where Spotify
  reported 0).
- popularity: log1p(train listens) from stats_item_train.parquet.
- tier: P3's bridge tier, as codes into concepts.MATCH_TIERS.

History weights: P6 stores round(1000 * log1p(listens)) as uint16, so
w_h = data / 1000, the same weights the ALS baseline used.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

from . import artists as ar
from .common import connect, features_table
from .concepts import (
    AUDIO_DIMS,
    COLISTEN_CONCEPTS,
    CONTINUOUS,
    CREDIT_CONCEPTS,
    MATCH_TIERS,
    SIGMA_FLOOR,
    TERM_CONCEPTS,
    ItemSide,
    build_user_state,
    pairs,
)
from .config import MODEL_DIR

WEIGHT_SCALE = 1000.0
TERM_FILES = {
    "tags_direct": "item_tags_direct.npz",
    "tags_artist": "item_tags_artist.npz",
    "genres": "item_genres.npz",
}
TERM_VOCAB = {
    "tags_direct": ("vocab_tags.parquet", "tag_name", "tag_idx"),
    "tags_artist": ("vocab_tags.parquet", "tag_name", "tag_idx"),
    "genres": ("vocab_genres.parquet", "genre", "genre_idx"),
}
COLISTEN_FILES = {
    "same_artist": "colisten_same_artist.npz",
    "other_artists": "colisten_other_artists.npz",
}


@dataclass
class Sources:
    """The raw item-side arrays, before ItemSide normalizes them.
    terms and credits map a name to (indptr, indices, n_cols), colisten
    maps a kind to (indptr, neighbor, cosine). Rows are item slots."""

    n_slots: int
    terms: dict
    credits: dict
    credit_ids: dict
    colisten: dict
    audio: np.ndarray
    popularity: np.ndarray
    tier: np.ndarray
    term_names: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)


# Builders


def compact(indptr, ids):
    """CSR rows of arbitrary ids -> (indptr, cols, n_cols, id_of_col).
    Columns are the sorted distinct ids."""
    id_of_col, cols = np.unique(np.asarray(ids, dtype=np.int64), return_inverse=True)
    return (
        np.asarray(indptr, dtype=np.int64),
        cols.astype(np.int64),
        int(len(id_of_col)),
        id_of_col,
    )


def tier_codes(item_ids, tier_names, n_slots):
    """Codes into MATCH_TIERS per item slot. Slots without a row get
    "unmatched". An unknown tier name raises."""
    lookup = {name: k for k, name in enumerate(MATCH_TIERS)}
    out = np.full(n_slots, lookup["unmatched"], dtype=np.int64)
    names = [str(t) for t in tier_names]
    bad = sorted(set(names) - set(lookup))
    if bad:
        raise ValueError(f"unknown tier names {bad[:5]}")
    out[np.asarray(item_ids, dtype=np.int64)] = [lookup[t] for t in names]
    return out


def audio_array(item_ids, columns, n_slots):
    """(n_slots, len(AUDIO_DIMS)) float64 with NaN where a value is
    absent. columns maps each AUDIO_DIMS name to a possibly masked
    array (DuckDB masks NULLs)."""
    out = np.full((n_slots, len(AUDIO_DIMS)), np.nan)
    rows = np.asarray(item_ids, dtype=np.int64)
    for d, name in enumerate(AUDIO_DIMS):
        col = columns[name]
        out[rows, d] = np.ma.filled(np.ma.asarray(col).astype(np.float64), np.nan)
    return out


def popularity_array(item_ids, listens, n_slots):
    """log1p(train listens) per item slot, 0 for items without any."""
    out = np.zeros(n_slots, dtype=np.float64)
    out[np.asarray(item_ids, dtype=np.int64)] = np.log1p(
        np.asarray(listens, dtype=np.float64)
    )
    return out


def _small_indptr(indptr, n_entries):
    """int32 indptr when it fits, so scipy keeps int32 indices and does
    not copy them."""
    if n_entries < np.iinfo(np.int32).max:
        return np.asarray(indptr, dtype=np.int32)
    return np.asarray(indptr, dtype=np.int64)


def item_side(src, transposed=False):
    """concepts.ItemSide from Sources. Terms and credits are binary: a
    pair listed twice still counts once, as in Reference.

    Memory: the co-listening matrices share their arrays with
    src.colisten, and ItemSide sorts each row's neighbours by id in
    place. The rows keep their content, and Reference sums whole rows,
    so the order does not matter to it. Sharing saves about 2 GB on
    P9b's 165 million neighbours."""
    n = src.n_slots

    def binary(indptr, indices, n_cols):
        indices = np.asarray(indices)
        idx = (
            indices.astype(np.int32)
            if (len(indices) == 0 or int(indices.max()) < 2**31 - 1)
            else indices
        )
        m = sp.csr_matrix(
            (np.ones(len(idx)), idx, _small_indptr(indptr, len(idx))), shape=(n, n_cols)
        )
        m.sum_duplicates()
        m.data[:] = 1.0
        return m

    colisten = {}
    for kind, (indptr, nbr, cos) in src.colisten.items():
        colisten[kind] = sp.csr_matrix(
            (
                np.asarray(cos, dtype=np.float64),
                np.asarray(nbr, np.int32),
                _small_indptr(indptr, len(nbr)),
            ),
            shape=(n, n),
        )
    return ItemSide(
        term_mats={k: binary(*v) for k, v in src.terms.items()},
        credit_mats={k: binary(*v[:3]) for k, v in src.credits.items()},
        colisten=colisten,
        audio=src.audio,
        popularity=src.popularity,
        tier=src.tier,
        transposed=transposed,
    )


def _fetch(sql):
    con = connect(4)
    try:
        return con.execute(sql).fetchnumpy()
    finally:
        con.close()


def load_sources(n_slots, log):
    """Sources from the pipeline's files. Each step logs its size."""
    terms, term_names = {}, {}
    for name, fname in TERM_FILES.items():
        z = np.load(MODEL_DIR / fname)
        shape = tuple(int(x) for x in z["shape"])
        if shape[0] != n_slots:
            raise ValueError(
                f"{fname} has {shape[0]} rows, the catalog " f"has {n_slots} slots"
            )
        terms[name] = (
            np.asarray(z["indptr"], dtype=np.int64),
            np.asarray(z["indices"], dtype=np.int32),
            shape[1],
        )
        vfile, vname, vidx = TERM_VOCAB[name]
        cols = _fetch(
            f"SELECT v.{vname} AS n, v.{vidx} AS i FROM "
            f"read_parquet('{MODEL_DIR / vfile}') AS v ORDER BY i"
        )
        names = [str(x) for x in cols["n"]]
        if (
            not np.array_equal(
                np.asarray(cols["i"], dtype=np.int64), np.arange(len(names))
            )
            or len(names) != shape[1]
        ):
            raise ValueError(f"{vfile} does not number its terms " f"0..{shape[1] - 1}")
        term_names[name] = names
        log(
            f"concept data: {name}: {len(terms[name][1]):,} item-term "
            f"pairs over {shape[1]:,} terms"
        )

    credits, credit_ids, stats = {}, {}, {}
    a_indptr, a_ids, a_stats = ar.load_credits(n_slots)
    stats["artists"] = a_stats
    c = compact(a_indptr, a_ids)
    credits["artists"], credit_ids["artists"] = c[:3], c[3]
    w = _fetch(
        f"SELECT DISTINCT w.item_id AS item_id, "
        f"CAST(w.work_id AS BIGINT) AS work_id FROM read_parquet("
        f"'{features_table('item_works')}') AS w "
        f"WHERE w.item_id IS NOT NULL AND w.work_id IS NOT NULL"
    )
    w_indptr, w_ids = ar.credit_csr(w["item_id"], w["work_id"], [], n_slots)
    c = compact(w_indptr, w_ids)
    credits["works"], credit_ids["works"] = c[:3], c[3]
    for name in credits:
        log(
            f"concept data: {name}: {len(credits[name][1]):,} credits, "
            f"{credits[name][2]:,} distinct"
        )

    colisten = {}
    for kind, fname in COLISTEN_FILES.items():
        z = np.load(MODEL_DIR / fname)
        colisten[kind] = (
            np.asarray(z["indptr"], dtype=np.int64),
            np.asarray(z["neighbor"]),
            np.asarray(z["cosine"], dtype=np.float64),
        )
        if len(colisten[kind][0]) != n_slots + 1:
            raise ValueError(f"{fname} does not cover the item slots")
        log(
            f"concept data: co-listening {kind}: "
            f"{len(colisten[kind][1]):,} neighbours"
        )

    cols = _fetch(
        f"SELECT a.item_id AS item_id, "
        f"{', '.join(f'a.{d} AS {d}' for d in AUDIO_DIMS)}"
        f" FROM read_parquet("
        f"'{MODEL_DIR / 'item_audio_std.parquet'}') AS a"
    )
    audio = audio_array(cols["item_id"], cols, n_slots)
    log(f"concept data: audio for {len(cols['item_id']):,} items")
    cols = _fetch(
        f"SELECT s.item_id AS item_id, s.listens AS listens "
        f"FROM read_parquet("
        f"'{MODEL_DIR / 'stats_item_train.parquet'}') AS s"
    )
    popularity = popularity_array(cols["item_id"], cols["listens"], n_slots)
    cols = _fetch(
        f"SELECT c.item_id AS item_id, c.tier AS tier "
        f"FROM read_parquet("
        f"'{MODEL_DIR / 'item_covariates.parquet'}') AS c"
    )
    tier = tier_codes(cols["item_id"], cols["tier"], n_slots)
    return Sources(
        n_slots,
        terms,
        credits,
        credit_ids,
        colisten,
        audio,
        popularity,
        tier,
        term_names,
        stats,
    )


# Per user


def history_of(u_indptr, u_indices, u_data, user):
    """(items, weights) of one user from P6's user-major CSR."""
    s, e = int(u_indptr[user]), int(u_indptr[user + 1])
    items = np.asarray(u_indices[s:e], dtype=np.int64)
    weights = np.asarray(u_data[s:e], dtype=np.float64) / WEIGHT_SCALE
    return items, weights


def user_rows(space, items, weights, candidates):
    """Every concept for one user's candidates, through concepts.pairs().
    Returns (values (n, len(CONTINUOUS)) float64 with NaN where missing,
    has (n, len(CONTINUOUS)) bool, tier (n,) int64). Column j is
    CONTINUOUS[j]."""
    state = build_user_state(items, weights, space)
    out = pairs(state, space, candidates)
    n = len(candidates)
    values = np.empty((n, len(CONTINUOUS)), dtype=np.float64)
    has = np.empty((n, len(CONTINUOUS)), dtype=bool)
    for j, c in enumerate(CONTINUOUS):
        values[:, j], has[:, j] = out[c]
    return values, has, np.asarray(out["match_tier"], dtype=np.int64)


class Reference:
    """A second implementation of every concept, for checks only. Plain
    loops and dicts over the raw Sources, one (user, item) at a time. It
    shares no code with concepts.py."""

    def __init__(self, src):
        self.src = src
        self.idf = {}
        for name, (indptr, indices, n_cols) in src.terms.items():
            rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
            pair_key = np.unique(
                rows.astype(np.int64) * n_cols + np.asarray(indices, dtype=np.int64)
            )
            cols = pair_key % n_cols
            df = np.zeros(n_cols)
            np.add.at(df, cols, 1.0)
            n_docs = len(np.unique(pair_key // n_cols))
            self.idf[name] = [math.log((1 + n_docs) / (1 + d)) + 1 for d in df.tolist()]

    def user(self, items, weights):
        """Everything the concepts read from one history."""
        src = self.src
        items = [int(x) for x in items]
        weights = [float(x) for x in weights]
        total = math.fsum(weights)
        hist = dict(zip(items, weights))
        prof = {}
        for name, (indptr, indices, _) in src.terms.items():
            p = defaultdict(float)
            for h, w in hist.items():
                for t in set(indices[indptr[h] : indptr[h + 1]].tolist()):
                    p[t] += w
            prof[name] = {t: v / total for t, v in p.items()}
        shares = {}
        for name, (indptr, cols, _) in src.credits.items():
            s = defaultdict(float)
            for h, w in hist.items():
                for a in set(cols[indptr[h] : indptr[h + 1]].tolist()):
                    s[a] += w
            shares[name] = {a: v / total for a, v in s.items()}
        mu, sigma, ok = [], [], []
        for d in range(len(AUDIO_DIMS)):
            num = num2 = den = 0.0
            for h, w in hist.items():
                x = src.audio[h, d]
                if not math.isnan(x):
                    num += w * x
                    num2 += w * x * x
                    den += w
            if den > 0:
                m = num / den
                sd = math.sqrt(max(num2 / den - m * m, 0.0))
                mu.append(m)
                sigma.append(max(sd, SIGMA_FLOOR))
                ok.append(True)
            else:
                mu.append(0.0)
                sigma.append(1.0)
                ok.append(False)
        return {
            "total": total,
            "hist": hist,
            "prof": prof,
            "shares": shares,
            "mu": mu,
            "sigma": sigma,
            "ok": ok,
        }

    def values(self, u, item):
        """{concept: (value, has_data)} and "match_tier" for one item."""
        src = self.src
        item = int(item)
        out = {}
        for concept, name in TERM_CONCEPTS.items():
            indptr, indices, _ = src.terms[name]
            idf, p = self.idf[name], u["prof"][name]
            item_terms = set(indices[indptr[item] : indptr[item + 1]].tolist())
            u_norm = math.sqrt(math.fsum((idf[t] * v) ** 2 for t, v in p.items()))
            if item == 0 or not item_terms or u_norm == 0:
                out[concept] = (math.nan, False)
                continue
            numer = math.fsum(idf[t] ** 2 * p.get(t, 0.0) for t in item_terms)
            i_norm = math.sqrt(math.fsum(idf[t] ** 2 for t in item_terms))
            out[concept] = (numer / (i_norm * u_norm), True)
        for concept, name in CREDIT_CONCEPTS.items():
            indptr, cols, _ = src.credits[name]
            own = set(cols[indptr[item] : indptr[item + 1]].tolist())
            if item == 0 or not own:
                out[concept] = (math.nan, False)
                continue
            out[concept] = (max(u["shares"][name].get(a, 0.0) for a in own), True)
        for concept, kind in COLISTEN_CONCEPTS.items():
            indptr, nbr, cos = src.colisten[kind]
            s, e = int(indptr[item]), int(indptr[item + 1])
            if item == 0 or s == e:
                out[concept] = (math.nan, False)
                continue
            out[concept] = (
                math.fsum(
                    float(c) * u["hist"].get(int(h), 0.0)
                    for h, c in zip(nbr[s:e].tolist(), cos[s:e].tolist())
                )
                / u["total"],
                True,
            )
        terms = []
        for d in range(len(AUDIO_DIMS)):
            x = src.audio[item, d]
            if item != 0 and u["ok"][d] and not math.isnan(x):
                terms.append(((x - u["mu"][d]) / u["sigma"][d]) ** 2)
        out["audio_fit"] = (
            (-math.fsum(terms) / len(terms), True) if terms else (math.nan, False)
        )
        out["popularity"] = (
            (float(src.popularity[item]), True) if item else (math.nan, False)
        )
        out["match_tier"] = int(src.tier[item])
        return out
