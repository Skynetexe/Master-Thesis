"""Tracks that MLHD lacks, for the second list of the ListenBrainz demo.
The rules are in the docstring of pipeline/lb_demo.py, part "Outside
list", written before any run on real data. This module holds the steps
that read no file:
P4's term rules for a track outside the catalog, an ItemSide in the
catalog's columns, the concept values, the reasons and the evidence.

Why an ItemSide of its own: concepts.pairs() computes every concept of a
listener for rows that share the catalog's columns (the vocabularies, the
credit columns, the audio dimensions) and the catalog's IDF. So a track
outside the catalog gets the values of the same expressions that the model
learned from, with the listener's state from the catalog history. The IDF
stays the catalog's: one more track never changes it.

P4's rules, as stage_p4.py and SpotifyBridge.py apply them:
- tags (direct and artist): binary. The raw tag names fold onto their
  canonical names through curated/tag_aliases.txt before the vocabulary
  join (a variant listed twice gives each canonical, as P4's LEFT JOIN
  does). A name outside vocab_tags gives no column.
- genres: binary, the exact genre string of vocab_genres.
- credits: binary, the catalog's credit columns (concept_data.compact over
  pipeline/artists.py's rule, so a special-purpose artist has no column).
- audio: (x - mean) / sd with P4's unweighted mean and sd of
  audio_std_stats.json, tempo 0 as missing (stage_p4._std_sql).
"""

import re
import unicodedata

import numpy as np
import scipy.sparse as sp

from . import concepts as cn
from . import p13c as pc

TITLE_CUT = " - "  # Spotify's version suffix: "Song - Remastered"
ISRC_LEN = 12
# lb_demo.py, part 3, 2: a bracketed part with one of these words marks
# a version of a song ("take" only with a number: "(Take Me Home)" stays)
VERSION_RE = re.compile(
    r"\b(remaster(?:ed|s)?|re-master(?:ed)?|mix(?:es|ed)?|remix(?:es|ed)?|"
    r"live|demo|takes?\s*\d+|version|edit|mono|stereo|acoustic|"
    r"instrumental|rehearsal|outtakes?|sessions?|single|radio|bonus|"
    r"alternate|alternative|alt|early|rough|unplugged|karaoke|feat|ft|"
    r"featuring|anniversary|edition)\b",
    re.IGNORECASE,
)
BRACKET_RE = re.compile(r"\s*[\(\[]([^\)\]]*)(?:[\)\]]|$)")
QUOTED_RE = re.compile(
    '["\u201c\u201d\u201e]([^"\u201c\u201d\u201e]+)' '["\u201c\u201d\u201e]'
)
QUOTES = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u02bc": "'",
        "`": "'",
        "\u00b4": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u2033": '"',
    }
)


# Selection


def norm_isrc(s):
    """The ISRC as SpotifyBridge.py compares it: upper case without the
    characters outside A-Z and 0-9. None unless 12 characters remain."""
    if not isinstance(s, str):
        return None
    t = re.sub(r"[^A-Za-z0-9]", "", s).upper()
    return t if len(t) == ISRC_LEN else None


def title_key(name):
    """The title up to the first " - ", in lower case with single
    spaces, so "Song - Remastered 2011" and "Song" share a key."""
    if not isinstance(name, str):
        return ""
    return " ".join(name.split(TITLE_CUT, 1)[0].lower().split())


def norm_name(name):
    """A name for comparisons (lb_demo.py, part 2, known artists): NFKC,
    case folded, typographic quotes folded, single spaces. "" for a
    non-string."""
    if not isinstance(name, str):
        return ""
    s = unicodedata.normalize("NFKC", name).translate(QUOTES).casefold()
    return " ".join(s.split())


def _is_version(inner):
    """True when a bracketed part marks a version (lb_demo.py, part 3,
    2): a word of VERSION_RE, or a first word "with"."""
    return bool(VERSION_RE.search(inner) or inner.strip().lower().startswith("with "))


def cut_versions(name):
    """name without its bracketed parts that mark a version (lb_demo.py,
    part 3, 2): a part in "(...)" or "[...]", or from an open bracket to
    the end, for which _is_version() holds."""
    if not isinstance(name, str):
        return name
    return BRACKET_RE.sub(lambda m: "" if _is_version(m.group(1)) else m.group(0), name)


def song_key(name):
    """The song key of lb_demo.py, parts 2 and 3: title_key() of the
    name without its bracketed version parts, with typographic quotes folded
    and every character that is no letter, digit or space removed, so
    "Don\u2019t Stop Me Now - 2011 Mix", "Don't Stop Me Now (Remastered)"
    and "Don't Stop Me Now" share a key. "" for a title without letters or
    digits."""
    s = norm_name(title_key(cut_versions(name)))
    s = "".join(c if (c.isalnum() or c.isspace()) else "" for c in s)
    return " ".join(s.split())


def song_keys(name):
    """The keys of an outside track (lb_demo.py, part 3, 2): song_key()
    of the name, then the song key of every quoted title inside a bracketed
    version part, so "Dusty ("Poor Twisted Me" Take 12)" also has the key
    of "Poor Twisted Me". Each key once, empty keys left out."""
    if not isinstance(name, str):
        return []
    keys = [song_key(name)]
    for m in BRACKET_RE.finditer(name):
        if _is_version(m.group(1)):
            keys += [song_key(q) for q in QUOTED_RE.findall(m.group(1))]
    out = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def dedupe_tracks(tracks):
    """One track per ISRC, then one per (song key, Spotify artists): the
    highest Spotify popularity first, ties by the smallest Spotify id
    (lb_demo.py, part 1, with part 3's song key in place of the title key).
    tracks: dicts with sp_id, isrc (normalized or None), name, popularity
    and sp_artists. Returns the kept tracks sorted by Spotify id."""

    def pop(t):
        p = t.get("popularity")
        return -1 if p is None else int(p)

    seen_isrc, seen_title, kept = set(), set(), []
    for t in sorted(tracks, key=lambda t: (-pop(t), str(t["sp_id"]))):
        key = (song_key(t.get("name")), tuple(sorted(t.get("sp_artists") or ())))
        if (t.get("isrc") and t["isrc"] in seen_isrc) or key in seen_title:
            continue
        if t.get("isrc"):
            seen_isrc.add(t["isrc"])
        seen_title.add(key)
        kept.append(t)
    return sorted(kept, key=lambda t: str(t["sp_id"]))


# P4's rules


def alias_map(pairs):
    """{variant: sorted canonicals} from stage_p4._read_tag_aliases()."""
    out = {}
    for variant, canonical in pairs:
        out.setdefault(variant, set()).add(canonical)
    return {v: sorted(c) for v, c in out.items()}


def fold(names, aliases):
    """The distinct tag names after P4's alias fold."""
    out = set()
    for n in names:
        if isinstance(n, str):
            out.update(aliases.get(n, (n,)))
    return out


def term_csr(rows, index):
    """(indptr, cols) of the distinct vocabulary columns of each track.
    rows: per track an iterable of names. index: {name: column}."""
    indptr, cols = [0], []
    for names in rows:
        c = sorted({int(index[n]) for n in names if n in index})
        cols.extend(c)
        indptr.append(len(cols))
    return np.asarray(indptr, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def credit_csr(rows, id_of_col):
    """(indptr, cols) of the catalog's credit columns of each track. rows:
    per track an iterable of MusicBrainz artist ids. id_of_col: the
    catalog's sorted credit ids. An id without a column is left out."""
    id_of_col = np.asarray(id_of_col, dtype=np.int64)
    indptr, cols = [0], []
    for ids in rows:
        a = np.unique(np.asarray(list(ids), dtype=np.int64))
        if len(a) and len(id_of_col):
            pos = np.minimum(np.searchsorted(id_of_col, a), len(id_of_col) - 1)
            cols.extend(int(p) for p in pos[id_of_col[pos] == a])
        indptr.append(len(cols))
    return np.asarray(indptr, dtype=np.int64), np.asarray(cols, dtype=np.int64)


def audio_z(raw, stats):
    """(n, len(AUDIO_DIMS)) z-scores as stage_p4._std_sql writes them.
    raw: {column: (n,) float64, NaN where NULL}. stats: the "columns" of
    audio_std_stats.json."""
    n = len(raw[cn.AUDIO_DIMS[0][2:]])
    out = np.full((n, len(cn.AUDIO_DIMS)), np.nan)
    for d, dim in enumerate(cn.AUDIO_DIMS):
        col = dim[2:]
        x = np.asarray(raw[col], dtype=np.float64)
        z = (x - float(stats[col]["mean"])) / float(stats[col]["sd"])
        if col == "tempo":
            z = np.where(x == 0.0, np.nan, z)
        out[:, d] = z
    return out


# Item side


def _slot_csr(indptr, cols, n_cols):
    """CSR over slots 0..n with slot 0 empty, from per-track rows."""
    ip = np.concatenate(([0], np.asarray(indptr, dtype=np.int64)))
    cols = np.asarray(cols, dtype=np.int64)
    return sp.csr_matrix(
        (np.ones(len(cols)), cols, ip), shape=(len(ip) - 1, int(n_cols))
    )


def outside_space(space, terms, credits, audio, tier_code):
    """concepts.ItemSide of n outside tracks in slots 1..n (slot 0 is the
    phantom slot), in the catalog's columns and with the catalog's IDF.
    terms: {source: (indptr, cols)} for every term matrix of space.
    credits: {name: (indptr, cols)}, a credit name left out is empty.
    audio: (n, len(AUDIO_DIMS)) with NaN where absent. No co-listening."""
    audio = np.asarray(audio, dtype=np.float64)
    n = len(audio)
    if set(terms) != set(space.term_mats):
        raise ValueError(
            f"the outside terms {sorted(terms)} differ from the "
            f"catalog's {sorted(space.term_mats)}"
        )
    empty_rows = (np.zeros(n + 1, dtype=np.int64), np.zeros(0, np.int64))
    term_mats = {
        k: _slot_csr(*terms[k], space.term_mats[k].shape[1]) for k in space.term_mats
    }
    credit_mats = {
        k: _slot_csr(*credits.get(k, empty_rows), space.credit_mats[k].shape[1])
        for k in space.credit_mats
    }
    for k, m in list(term_mats.items()) + list(credit_mats.items()):
        if m.shape[0] != n + 1:
            raise ValueError(f"{k}: {m.shape[0] - 1} rows for {n} tracks")
    a = np.full((n + 1, len(cn.AUDIO_DIMS)), np.nan)
    a[1:] = audio
    empty = sp.csr_matrix((n + 1, n + 1), dtype=np.float64)
    return cn.ItemSide(
        term_mats=term_mats,
        credit_mats=credit_mats,
        colisten={k: empty.copy() for k in cn.COLISTEN_CONCEPTS.values()},
        audio=a,
        popularity=np.zeros(n + 1),
        tier=np.full(n + 1, int(tier_code), dtype=np.int64),
        idf={k: np.array(v, dtype=np.float64, copy=True) for k, v in space.idf.items()},
        transposed=False,
    )


def concepts_of(state, space_out):
    """{concept: (value, has)} of every outside track (slots 1..n), through
    concepts.pairs(), with the listener's state from the catalog."""
    return cn.pairs(state, space_out, np.arange(1, space_out.n_slots))


def evidence(state, space, space_out, slot):
    """concepts.evidence() for one outside slot. Term concepts and audio
    read the outside rows. The credit's history tracks are catalog tracks,
    so they come from the catalog's credit matrix (space). No co-listening
    neighbour exists for an outside track."""
    slot = int(slot)
    vals = cn.pairs(state, space_out, np.array([slot]))
    out = {}
    for concept, name in cn.TERM_CONCEPTS.items():
        value, has = vals[concept][0][0], bool(vals[concept][1][0])
        entry = {"value": value, "has_data": has, "unit": name, "terms": []}
        if has:
            m = space_out.term_mats[name]
            s, e = m.indptr[slot], m.indptr[slot + 1]
            cols, x = m.indices[s:e], m.data[s:e]
            p = state.term_share[name][cols]
            denom = space_out.item_norm[name][slot] * state.term_norm[name]
            t = space_out.idf[name][cols] ** 2 * x * p / denom
            keep = p > 0
            rows = sorted(
                zip(cols[keep].tolist(), t[keep].tolist(), p[keep].tolist()),
                key=lambda r: (-r[1], r[0]),
            )
            entry["terms"] = [
                {"id": c, "term": tv, "listening_share": pv} for c, tv, pv in rows
            ]
        out[concept] = entry
    for concept, name in cn.CREDIT_CONCEPTS.items():
        value, has = vals[concept][0][0], bool(vals[concept][1][0])
        entry = {
            "value": value,
            "has_data": has,
            "unit": name,
            "credit": None,
            "terms": [],
        }
        if has:
            m = space_out.credit_mats[name]
            cols = m.indices[m.indptr[slot] : m.indptr[slot + 1]].astype(np.int64)
            shares = cn._credit_lookup(state, name, cols)
            best = int(cols[np.lexsort((cols, -shares))[0]])
            entry["credit"] = best
            hist = space.credit_mats[name][state.items]
            carries = np.flatnonzero(hist[:, [best]].toarray().ravel() > 0)
            rows = sorted(
                (
                    (int(state.items[k]), float(state.weights[k] / state.total))
                    for k in carries
                ),
                key=lambda r: (-r[1], r[0]),
            )
            entry["terms"] = [{"id": h, "term": tv} for h, tv in rows]
        out[concept] = entry
    for concept in cn.COLISTEN_CONCEPTS:
        out[concept] = {
            "value": vals[concept][0][0],
            "has_data": bool(vals[concept][1][0]),
            "unit": "tracks",
            "terms": [],
        }
    value, has = vals["audio_fit"][0][0], bool(vals["audio_fit"][1][0])
    entry = {"value": value, "has_data": has, "unit": "audio_dims", "terms": []}
    if has:
        both = space_out.audio_mask[slot] & state.audio_dim_ok
        n = int(both.sum())
        z = (space_out.audio_filled[slot] - state.audio_mu) / state.audio_sigma
        rows = [
            {"id": cn.AUDIO_DIMS[d], "term": float(-(z[d] ** 2) / n), "z": float(z[d])}
            for d in np.flatnonzero(both)
        ]
        rows.sort(key=lambda r: (r["term"], r["id"]))
        entry["terms"] = rows
    out["audio_fit"] = entry
    return out


# Reasons


def shown_top(val, cand, k=pc.K_TOP):
    """The reasons of the outside list: a candidate reason with a positive
    value among the k largest of its track (ties by the reason order, as
    p13c.ranks_of orders them). val, cand: (G, n)."""
    r = pc.ranks_of(np.asarray(val, dtype=np.float64), np.asarray(cand, bool))
    return np.asarray(cand, bool) & (np.asarray(val) > 0) & (r <= k)
