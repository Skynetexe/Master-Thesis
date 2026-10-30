"""Artist credits per track, under one rule for every stage that uses
them: P9b (same-artist and other-artist co-listening) and P12
(artist_affinity). One rule matters: with two rules, a track could count
as "same artist" in one concept and as another artist in the other.

Source: the features table item_artists (BuildFeatures.py), one row per
credit (item_id, artist_id) of the track's canonical recording.

Rule: special-purpose artists are not artists. MusicBrainz uses them for
missing or collective credits: "[unknown]", "[anonymous]",
"[traditional]", "[no artist]", "Various Artists" and a few more. Two
tracks credited only to "[unknown]" have no artist in common. A track
keeps its other credits. The rule matches the names in SPECIAL_NAMES
exactly (case-insensitive) and the two MBIDs below. It does not match
every bracketed name, because real artists such as "[dunkelbunt]" use
brackets too. An artist id counts as special when any of its credited
names matches.

A credit prints the name on the release, and MusicBrainz lets that name
differ from the artist's own name. So a credit to [unknown] can print
any name. special_summary() keeps the two apart for the reports: the
part of the rule that matched each special artist, and the names that
its credits print.
"""

from collections import Counter

import numpy as np

from .common import connect, features_table

VARIOUS_ARTISTS_MBID = "89ad4ac3-39f7-470e-963a-56509c546377"
UNKNOWN_MBID = "125ec42a-7229-4250-afc5-e057484327fe"
SPECIAL_MBIDS = {VARIOUS_ARTISTS_MBID, UNKNOWN_MBID}
MBID_NAMES = {VARIOUS_ARTISTS_MBID: "Various Artists", UNKNOWN_MBID: "[unknown]"}
SPECIAL_NAMES = {
    "various artists",
    "[unknown]",
    "[anonymous]",
    "[traditional]",
    "[no artist]",
    "[data]",
    "[dialogue]",
    "[theatre]",
    "[church chimes]",
    "[language instruction]",
    "[nature sounds]",
    "[christmas music]",
    "[disney]",
    "[audiobook]",
    "[classical]",
}


def _norm(s):
    """s without outer spaces and in lower case. None when s is not a
    string, and None matches nothing."""
    return s.strip().lower() if isinstance(s, str) else None


def _strings(col):
    """A fetched string column as a list, with None for SQL NULL. DuckDB
    returns a masked array when a column holds NULLs."""
    data, miss = np.ma.getdata(col), np.ma.getmaskarray(col)
    return [None if m or v is None else str(v) for v, m in zip(data, miss)]


def special_artist_ids(artist_ids, names, mbids):
    """Sorted artist ids that count as special-purpose artists. names and
    mbids run along artist_ids."""
    artist_ids = np.asarray(artist_ids, dtype=np.int64)
    hit = np.fromiter(
        (
            _norm(n) in SPECIAL_NAMES or _norm(m) in SPECIAL_MBIDS
            for n, m in zip(names, mbids)
        ),
        dtype=bool,
        count=len(artist_ids),
    )
    return np.unique(artist_ids[hit])


def special_summary(artist_ids, names, mbids, special, top_names=3):
    """One dict per special artist id, most credit rows first:
    - artist_id
    - matched_by: the parts of the rule that its rows match, such as
      'name "[unknown]"' or 'MBID of [unknown]'
    - rows: its credit rows
    - distinct_names: how many different names its credits print
    - names: its top_names most printed names, as [name, rows] pairs.
    The printed names are data, not rule names (see the module
    docstring). A credit without a name shows as "(no name)"."""
    artist_ids = np.asarray(artist_ids, dtype=np.int64)
    rows = np.flatnonzero(np.isin(artist_ids, np.asarray(special, dtype=np.int64)))
    per = {}
    for k in rows:
        rules, printed = per.setdefault(int(artist_ids[k]), (set(), Counter()))
        n, m = _norm(names[k]), _norm(mbids[k])
        if n in SPECIAL_NAMES:
            rules.add(f'name "{n}"')
        if m in SPECIAL_MBIDS:
            rules.add(f"MBID of {MBID_NAMES[m]}")
        printed["(no name)" if names[k] is None else names[k]] += 1
    out = []
    for a, (rules, printed) in per.items():
        top = sorted(printed.items(), key=lambda t: (-t[1], t[0]))
        out.append(
            {
                "artist_id": a,
                "matched_by": sorted(rules),
                "rows": int(sum(printed.values())),
                "distinct_names": len(printed),
                "names": [[s, int(c)] for s, c in top[:top_names]],
            }
        )
    return sorted(out, key=lambda e: (-e["rows"], e["artist_id"]))


def credit_csr(item_ids, artist_ids, special_ids, n_items):
    """Each track's distinct artist ids without the special ones, as CSR
    arrays over item slots: (indptr, ids), each row sorted. Rows outside
    1..n_items-1 or with a negative artist id are dropped."""
    item_ids = np.asarray(item_ids, dtype=np.int64)
    artist_ids = np.asarray(artist_ids, dtype=np.int64)
    ok = (
        (item_ids >= 1)
        & (item_ids < n_items)
        & (artist_ids >= 0)
        & ~np.isin(artist_ids, np.asarray(special_ids, dtype=np.int64))
    )
    item_ids, artist_ids = item_ids[ok], artist_ids[ok]
    base = int(artist_ids.max()) + 1 if len(artist_ids) else 1
    key = np.unique(item_ids * base + artist_ids)
    indptr = np.concatenate(
        ([0], np.cumsum(np.bincount(key // base, minlength=n_items)))
    ).astype(np.int64)
    return indptr, (key % base).astype(np.int64)


def load_credits(n_items):
    """(indptr, ids, stats) from item_artists under the rule. stats holds
    credit_rows, special_artists (how many ids the rule matched),
    special_rows (their credit rows), special (special_summary() of
    them) and tracks_with_credit (tracks left with a credit)."""
    con = connect(4)
    try:
        cols = con.execute(
            f"SELECT item_id, artist_id, artist_name, "
            f"CAST(artist_mbid AS VARCHAR) AS artist_mbid "
            f"FROM read_parquet('{features_table('item_artists')}') "
            f"WHERE item_id IS NOT NULL AND artist_id IS NOT NULL"
        ).fetchnumpy()
    finally:
        con.close()
    item_ids = np.asarray(cols["item_id"], dtype=np.int64)
    artist_ids = np.asarray(cols["artist_id"], dtype=np.int64)
    names = _strings(cols["artist_name"])
    mbids = _strings(cols["artist_mbid"])
    special = special_artist_ids(artist_ids, names, mbids)
    indptr, ids = credit_csr(item_ids, artist_ids, special, n_items)
    stats = {
        "credit_rows": int(len(item_ids)),
        "special_artists": int(len(special)),
        "special_rows": int(np.isin(artist_ids, special).sum()),
        "special": special_summary(artist_ids, names, mbids, special),
        "tracks_with_credit": int(np.count_nonzero(np.diff(indptr))),
    }
    return indptr, ids, stats
