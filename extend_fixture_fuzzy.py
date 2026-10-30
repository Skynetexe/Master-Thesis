"""Extends the shared fixture for FuzzyBridge.py. Run it after
build_fixture.py and extend_fixture_spotify.py, against the same root.

FuzzyBridge.py only ever looks at items absent from item_spotify_map.parquet.
In this fixture, those are exactly items 5 (rec-0007) and 6 (rec-0008), the
two that SpotifyBridge.py's fixture left without a Spotify match. This script
gives each of them one Spotify candidate:

    item 5 (rec-0007, length 205000ms): a candidate at 205300ms (diff
        300ms, inside the 2500ms window), so tier 'name+duration'
    item 6 (rec-0008, length 198000ms): a candidate with no duration at
        all on the Spotify side, so it is excluded from pass 1 and falls
        through to the 'name_only' fallback

Rewrites tracks.parquet and artists.parquet (Kaggle side) a second time,
adding name (both tables) and duration_ms (tracks). FuzzyBridge.py
reads both directly off tracks.parquet, not off track_audio_features.parquet,
and neither SpotifyBridge.py nor its own discovery check ever required
them. This does not touch anything SpotifyBridge.py reads (id, rowid,
external_id_isrc, popularity, explicit, fetched_at all stay), so it changes
no result of SpotifyBridge.py on that fixture.
"""

import os
import sys

import duckdb

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/thesis_fixture"
KAG = f"{ROOT}/kaggle/spotify-catalogue/spotify_clean_parquet"
os.makedirs(KAG, exist_ok=True)  # don't assume extend_fixture_spotify.py already ran

con = duckdb.connect()


def write(path, columns, rows, types):
    collist = ", ".join(f"{c} {t}" for c, t in zip(columns, types))
    con.execute(f"CREATE OR REPLACE TEMP TABLE _t ({collist})")
    con.executemany(f"INSERT INTO _t VALUES ({', '.join(['?'] * len(columns))})", rows)
    con.execute(f"COPY _t TO '{path}' (FORMAT PARQUET, COMPRESSION 'zstd')")
    print(f"  wrote {path}  ({len(rows)} rows)")


print("rewriting tracks.parquet with name + duration_ms")
write(
    f"{KAG}/tracks.parquet",
    [
        "id",
        "rowid",
        "name",
        "external_id_isrc",
        "popularity",
        "explicit",
        "fetched_at",
        "duration_ms",
    ],
    [
        (
            "TRACKID000000000000001",
            101,
            "Placeholder One",
            "US1234567890",
            "50",
            "false",
            "2025-05-01",
            "200500",
        ),
        (
            "TRACKID000000000000002",
            102,
            "Placeholder Two A",
            None,
            "80",
            "false",
            "2025-05-01",
            "199000",
        ),
        (
            "TRACKID000000000000004",
            103,
            "Placeholder Three",
            "GB9876543210",
            "60",
            "true",
            "2025-05-01",
            "184800",
        ),
        (
            "TRACKID000000000000005",
            104,
            "Placeholder Four",
            None,
            "40",
            "false",
            "2025-05-01",
            None,
        ),
        # item 5's candidate: name matches exactly, duration inside the window
        (
            "TRACKID000000000000006",
            105,
            "Song Seven Canonical",
            None,
            "30",
            "false",
            "2025-05-01",
            "205300",
        ),
        # item 6's candidate: name matches, no duration at all on this side
        (
            "TRACKID000000000000007",
            106,
            "Song Eight Canonical",
            None,
            "20",
            "false",
            "2025-05-01",
            None,
        ),
    ],
    [
        "VARCHAR",
        "BIGINT",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
        "VARCHAR",
    ],
)

print("\nrewriting artists.parquet (Kaggle side) with name")
write(
    f"{KAG}/artists.parquet",
    ["rowid", "id", "name"],
    [
        (2001, "ARTISTID00000000000002", "Path B Artist"),
        (2099, "ARTISTID00000000000099", "Path A Final Artist"),
        (2005, "ARTISTID00000000000005", "Artist Seven"),
        (2006, "ARTISTID00000000000006", "Artist Eight"),
    ],
    ["BIGINT", "VARCHAR", "VARCHAR"],
)

print("\nadding the two new track_artists links")
con.execute(
    f"CREATE OR REPLACE TEMP TABLE _ta (track_rowid BIGINT, artist_rowid BIGINT)"
)
con.executemany(
    "INSERT INTO _ta VALUES (?, ?)", [(101, 2001), (105, 2005), (106, 2006)]
)
con.execute(
    f"COPY _ta TO '{KAG}/track_artists.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')"
)
print(f"  wrote {KAG}/track_artists.parquet  (3 rows, was 1, item 1's path-B row kept)")

print(f"\nExtension written under {ROOT}")

r"""
USE

    python3 extend_fixture_fuzzy.py /tmp/thesis_fixture
    THESIS_DATA_ROOT=/tmp/thesis_fixture python3 FuzzyBridge.py --fresh

FuzzyBridge.py reads its data folder from THESIS_DATA_ROOT. --fresh matters
here: the script's steps are file-backed under parquet/scratch/ and skip the
outputs that exist, so without it a stale scratch file would be reused and
nothing would be tested.

EXPECTED OUTPUT, in the order it prints:

    fuzzy universe: 2 items (2 with MB length)
    distinct (artist,track) keys: 2
    Spotify candidate tracks: 2
    duration-verified candidates: 1 rows over 1 items
    items with duration-verified match: 1
    name-only fallback matches: 1

    basis            n    avg_diff_ms
    name+duration    1    300.0
    name_only        1    NULL

    items_any  distinct_pct  listen_pct
    6          100.0         100.0

    tier        items  avg_listens_per_item  median_listens
    map         4      2.0                   1.5
    fuzzy_dur   1      1.0                   1.0
    fuzzy_name  1      1.0                   1.0
    (fuzzy_dur and fuzzy_name are tied at 1.0, so either may print first)

The 100.0 / 100.0 coverage line is expected: with six items, each designed
to have a Spotify counterpart somewhere in this fixture, the map tier and
the fuzzy tier together cover all six. On the real catalogue the map tier
alone reaches 15.37% of items and 68.65% of listens. The run shows that an
item that SpotifyBridge.py could not place is picked up by name+duration
when the durations agree, and falls through to name_only when they do not:
the hand-over between the two scripts, which neither script shows alone.
"""
