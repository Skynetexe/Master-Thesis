"""Builds a tiny, hand-designed fixture for BuildFeatures.py: two MLHD-shaped
listen files, a canonical layer, and the raw MusicBrainz tables BuildFeatures.py
actually reads. Every value below was chosen to exercise one specific branch
in BuildFeatures.py (see the comment on each recording). Run this once, then
point BuildFeatures.py at its output (the block at the bottom of this file
gives the commands).

Requires duckdb. No network, no pandas.
"""

import os
import sys

import duckdb

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/thesis_fixture"
MLHD = f"{ROOT}/parquet/mlhd"
CANON = f"{ROOT}/parquet/canonical"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
MBD = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump-derived"

for d in (MLHD, CANON, MB, MBD):
    os.makedirs(d, exist_ok=True)

con = duckdb.connect()


def write(path, columns, rows, types=None):
    """COPY a hand-built VALUES table to parquet with an explicit column list."""
    types = types or ["VARCHAR"] * len(columns)
    collist = ", ".join(f"{c} {t}" for c, t in zip(columns, types))
    con.execute(f"CREATE OR REPLACE TEMP TABLE _t ({collist})")
    placeholders = ", ".join(["?"] * len(columns))
    con.executemany(f"INSERT INTO _t VALUES ({placeholders})", rows)
    con.execute(f"COPY _t TO '{path}' (FORMAT PARQUET)")
    print(f"  wrote {path}  ({len(rows)} rows)")


print("MLHD listen files")
# File 0: U1 only. rec-0001 x3 (tests many-listens + first/last_ts spread and,
# together with rec-0002 sharing the same canonical id, the many-to-one
# interaction collapse). rec-0002 x1 (self-canonical). rec-0005 x2 (the
# multi-artist recording).
write(
    f"{MLHD}/mlhd_complete_0.parquet",
    ["user_mbid", "recording_mbid", "listened_at_unix"],
    [
        ("user-aaaa", "rec-0001", 1000),
        ("user-aaaa", "rec-0001", 1001),
        ("user-aaaa", "rec-0001", 1002),
        ("user-aaaa", "rec-0002", 1010),
        ("user-aaaa", "rec-0005", 1020),
        ("user-aaaa", "rec-0005", 1021),
    ],
    ["VARCHAR", "VARCHAR", "BIGINT"],
)

# File 1: proves U1 unions correctly across files (distinct-user scan), and
# carries rec-0004 (the never-resolved recording, must vanish downstream),
# rec-0006/0007/0008 (tags and work relations), and U2's one listen of
# rec-0003 (the gid-redirect-chain recording, and the only "light listener").
write(
    f"{MLHD}/mlhd_complete_1.parquet",
    ["user_mbid", "recording_mbid", "listened_at_unix"],
    [
        ("user-aaaa", "rec-0006", 1030),
        ("user-aaaa", "rec-0004", 1040),
        ("user-aaaa", "rec-0007", 1050),
        ("user-aaaa", "rec-0008", 1060),
        ("user-bbbb", "rec-0003", 2000),
    ],
    ["VARCHAR", "VARCHAR", "BIGINT"],
)

print("\ncanonical layer")
# rec-0001 redirects to rec-0002's canonical id: two raw recordings, one
# canonical item. This is the many-to-one collapse case.
write(
    f"{CANON}/canonical_recording_redirect.parquet",
    ["recording_mbid", "canonical_recording_mbid"],
    [("rec-0001", "rec-0002")],
)

# rec-0002, 0005, 0006, 0007, 0008, and the chain target rec-0003-new are all
# self-canonical: each already is the canonical form of itself. rec-0001,
# rec-0003, and rec-0004 are deliberately absent from this table.
write(
    f"{CANON}/canonical_musicbrainz_data.parquet",
    [
        "recording_mbid",
        "artist_credit_name",
        "artist_mbids",
        "recording_name",
        "release_mbid",
        "release_name",
    ],
    [
        (
            "rec-0002",
            "Artist Two",
            "artist-mbid-201",
            "Song Two Canonical",
            "rel-0002",
            "Album Two",
        ),
        (
            "rec-0003-new",
            "Artist Nine",
            "artist-mbid-209",
            "Song Three Canonical",
            "rel-0003",
            "Album Three",
        ),
        (
            "rec-0005",
            "Lead Artist feat. Feature Artist",
            "artist-mbid-205",
            "Song Five Canonical",
            "rel-0005",
            "Album Five",
        ),
        (
            "rec-0006",
            "Feature Artist",
            "artist-mbid-206",
            "Song Six Canonical",
            "rel-0006",
            "Album Six",
        ),
        (
            "rec-0007",
            "Artist Seven",
            "artist-mbid-207",
            "Song Seven Canonical",
            "rel-0007",
            "Album Seven",
        ),
        (
            "rec-0008",
            "Artist Eight",
            "artist-mbid-208",
            "Song Eight Canonical",
            "rel-0008",
            "Album Eight",
        ),
    ],
)

print("\nraw MusicBrainz tables")
# id, gid, artist_credit, name. rec-0004 gets its own row here despite never
# being canonicalized. This is what makes it "orphaned from canonicalization"
# rather than "absent from MusicBrainz entirely", and is what exercises the
# COALESCE(cd.recording_name, rec.name) fallback in item_catalog.
write(
    f"{MB}/recording.parquet",
    ["id", "gid", "artist_credit", "name"],
    [
        (2, "rec-0002", 2, "Song Two Raw"),
        (4, "rec-0004", 4, "Orphan Track"),
        (501, "rec-0003-new", 501, "Song Three Raw"),
        (5, "rec-0005", 5, "Song Five Raw"),
        (6, "rec-0006", 6, "Song Six Raw"),
        (7, "rec-0007", 7, "Song Seven Raw"),
        (8, "rec-0008", 8, "Song Eight Raw"),
    ],
    ["BIGINT", "VARCHAR", "BIGINT", "VARCHAR"],
)

# artist_credit, position, name, artist. rec-0005's artist_credit=5 gets two
# rows, at positions 0 and 1. This is the multi-artist-credit test.
write(
    f"{MB}/artist_credit_name.parquet",
    ["artist_credit", "position", "name", "artist"],
    [
        (2, 0, "Artist Two", 201),
        (4, 0, "Orphan Artist", 204),
        (501, 0, "Artist Nine", 209),
        (5, 0, "Lead Artist", 205),
        (5, 1, "Feature Artist", 206),
        (6, 0, "Feature Artist", 206),
        (7, 0, "Artist Seven", 207),
        (8, 0, "Artist Eight", 208),
    ],
    ["BIGINT", "INTEGER", "VARCHAR", "BIGINT"],
)

write(
    f"{MB}/artist.parquet",
    ["id", "gid"],
    [
        (201, "artist-gid-201"),
        (204, "artist-gid-204"),
        (205, "artist-gid-205"),
        (206, "artist-gid-206"),
        (207, "artist-gid-207"),
        (208, "artist-gid-208"),
        (209, "artist-gid-209"),
    ],
    ["BIGINT", "VARCHAR"],
)

# The gid-redirect chain: rec-0003 (raw) points at internal id 501, which is
# rec-0003-new above. This is the two-hop resolution path, the one path
# BuildFeatures.py has a fan-out guard for and nothing else checks directly.
write(
    f"{MB}/recording_gid_redirect.parquet",
    ["gid", "new_id"],
    [("rec-0003", 501)],
    ["VARCHAR", "BIGINT"],
)

write(
    f"{MB}/l_recording_work.parquet",
    ["link", "entity0", "entity1"],
    [(10, 7, 70), (11, 8, 71)],
    ["BIGINT", "BIGINT", "BIGINT"],
)
write(
    f"{MB}/link.parquet",
    ["id", "link_type"],
    [(10, 100), (11, 100)],
    ["BIGINT", "BIGINT"],
)
write(
    f"{MB}/link_type.parquet", ["id", "name"], [(100, "cover")], ["BIGINT", "VARCHAR"]
)
write(
    f"{MB}/work.parquet",
    ["id", "name"],
    [(70, "Original Work A"), (71, "Original Work B")],
    ["BIGINT", "VARCHAR"],
)
# Only link 10 (rec-0007's relation) gets an attribute. Link 11 (rec-0008's)
# deliberately gets none, to prove the LEFT JOIN leaves attributes NULL
# instead of dropping the row.
write(
    f"{MB}/link_attribute.parquet",
    ["link", "attribute_type"],
    [(10, 1000)],
    ["BIGINT", "BIGINT"],
)
write(
    f"{MB}/link_attribute_type.parquet",
    ["id", "name"],
    [(1000, "live")],
    ["BIGINT", "VARCHAR"],
)

print("\nmbdump-derived tables")
write(f"{MBD}/tag.parquet", ["id", "name"], [(1, "synthpop")], ["BIGINT", "VARCHAR"])
write(
    f"{MBD}/recording_tag.parquet",
    ["recording", "tag", "count"],
    [(6, 1, 5)],
    ["BIGINT", "BIGINT", "BIGINT"],
)
# Artist 206 is credited on both rec-0005 (position 1) and rec-0006 (position
# 0). One artist_tag row here should surface on both items in item_tags_artist.
write(
    f"{MBD}/artist_tag.parquet",
    ["artist", "tag", "count"],
    [(206, 1, 3)],
    ["BIGINT", "BIGINT", "BIGINT"],
)

print(f"\nFixture written under {ROOT}")

r"""
USE

1. BuildFeatures.py reads its data folder from THESIS_DATA_ROOT.

2. Write the fixture (this script):

     python3 build_fixture.py /tmp/thesis_fixture

3. Run BuildFeatures.py on it instead of the real data:

     THESIS_DATA_ROOT=/tmp/thesis_fixture python3 BuildFeatures.py

4. Compare the console output with the expected values below. If a value
   differs, check `/tmp/thesis_fixture/parquet/features/*.parquet` and
   `/tmp/thesis_fixture/parquet/interactions/*.parquet` directly.

EXPECTED CONSOLE OUTPUT, derived by hand from the fixture above:

    popularity: 8 rows, 11 listens
    map (rows = popularity rows; resolved ~99.9%): 8 rows
    item_listens (items = item_ids rows; listens = matrix total): 6 rows, 10 listens

    Final identity checks
    matrix: 6 pairs, 10 listens (must equal the item_listens total exactly)
    item_ids: 6  item_listens items: 6  (must match)

EXPECTED TABLE SHAPES:

    item_ids.parquet            6 rows   (rec-0002, rec-0003-new, rec-0005,
                                           rec-0006, rec-0007, rec-0008:
                                           rec-0001 collapses into rec-0002,
                                           rec-0004 has no canonical id at all)
    gid_redirect_resolution.parquet   1 row    (rec-0003 -> rec-0003-new only;
                                                 rec-0004 resolves to NULL and
                                                 is filtered out)
    item_catalog.parquet         8 rows   (one per raw recording_mbid, incl.
                                            rec-0001 and rec-0004 separately)
    item_artists.parquet         7 rows   (rec-0005 contributes 2, everyone
                                            else contributes 1)
    item_tags.parquet            1 row    (rec-0006 only)
    item_tags_artist.parquet     2 rows   (artist 206, once via rec-0005's
                                            item, once via rec-0006's item)
    item_works.parquet           2 rows   (rec-0007 with attributes="live",
                                            rec-0008 with attributes=NULL)
    user_ids.parquet             2 rows   (user-aaaa, user-bbbb)
    interactions_0.parquet       2 rows   (user-aaaa/rec-0002's item: 4
                                            listens, ts 1000-1010, from
                                            rec-0001 x3 + rec-0002 x1 folding
                                            into one row; user-aaaa/rec-0005's
                                            item: 2 listens, ts 1020-1021)
    interactions_1.parquet       4 rows   (rec-0004's listen has no item_id
                                            and must not appear anywhere)

When every number above matches, the script's logic holds end to end on
the fixture. A difference is cheap to diagnose here, much cheaper than
three hours into the rebuild of 21 billion rows.
"""
