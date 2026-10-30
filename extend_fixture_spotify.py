"""Extends the BuildFeatures.py fixture so SpotifyBridge.py can run against
the same /tmp/thesis_fixture root. Run this after build_fixture.py, pointed
at the same root. It reuses the six canonical items that fixture already
created (item_ids 1-6, mapped to rec-0002, rec-0003-new, rec-0005, rec-0006,
rec-0007, rec-0008) rather than inventing a second, disconnected fixture.

Two tables get rewritten, not just added to:
  - recording.parquet gains a length column (SpotifyBridge.py's discovery
    check requires it; BuildFeatures.py never needed it) and one new row for
    rec-0003. Its raw gid never got its own row before, because rec-0003
    resolved entirely through its canonical partner rec-0003-new. It needs
    one now to carry a Spotify URL, which links through the raw gid, not the
    canonical one.
  - artist.parquet gains a name column, for the same reason: BuildFeatures
    reads artist names off artist_credit_name, never off artist.name, but
    SpotifyBridge's artist_spotify_map does read artist.name directly.

Neither rewrite changes a result of BuildFeatures.py on the fixture.
rec-0003 already resolved via rec-0003-new regardless of whether its own raw
recording.parquet row exists, and nothing behind BuildFeatures.py's counts
(8/11, 8, 6/10, matrix 6/10, item_ids 6) reads artist.name.

Every new row below is designed to land on one of the six existing items,
chosen to hit one match-method tier each:
    item 1 (rec-0002):     ISRC only
    item 2 (rec-0003-new): URL only, with a second, losing URL candidate
    item 3 (rec-0005):     ISRC and URL, the same track, so isrc+url. The
                           recorded run gives ISRC alone, because of the
                           URL defect (see the end of this file)
    item 4 (rec-0006):     ISRC is ambiguous (shared with item 5) and
                           gets thrown out; wins via URL instead
    item 5 (rec-0007):     shares item 4's ambiguous ISRC and has no other
                           candidate, so it ends up with no Spotify match
    item 6 (rec-0008):     no ISRC and no URL, so no Spotify match either
Artist 206 (credited on both item 3 and item 4) gets a Spotify genre via a
redirected artist URL, deliberately with a chained redirect elsewhere in the
same JSON file, to exercise the "chained redirects" warning without
affecting artist 206's own (single-hop, successfully resolving) case.
"""

import os
import sys
import json

import duckdb

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/thesis_fixture"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
KAG = f"{ROOT}/kaggle/spotify-catalogue/spotify_clean_parquet"
KAGAF = f"{ROOT}/kaggle/spotify-catalogue/spotify_clean_audio_features_parquet"

for d in (MB, KAG, KAGAF, f"{ROOT}/kaggle/spotify-catalogue"):
    os.makedirs(d, exist_ok=True)

con = duckdb.connect()


def write(path, columns, rows, types):
    collist = ", ".join(f"{c} {t}" for c, t in zip(columns, types))
    con.execute(f"CREATE OR REPLACE TEMP TABLE _t ({collist})")
    con.executemany(f"INSERT INTO _t VALUES ({', '.join(['?'] * len(columns))})", rows)
    con.execute(f"COPY _t TO '{path}' (FORMAT PARQUET)")
    print(f"  wrote {path}  ({len(rows)} rows)")


print("rewriting recording.parquet and artist.parquet with the extra columns")
write(
    f"{MB}/recording.parquet",
    ["id", "gid", "artist_credit", "name", "length"],
    [
        (2, "rec-0002", 2, "Song Two Raw", 200000),
        (3, "rec-0003", 3, "Song Three Raw", 190000),
        (4, "rec-0004", 4, "Orphan Track", 210000),
        (501, "rec-0003-new", 501, "Song Three Raw", 195000),
        (5, "rec-0005", 5, "Song Five Raw", 185000),
        (6, "rec-0006", 6, "Song Six Raw", 220000),
        (7, "rec-0007", 7, "Song Seven Raw", 205000),
        (8, "rec-0008", 8, "Song Eight Raw", 198000),
    ],
    ["BIGINT", "VARCHAR", "BIGINT", "VARCHAR", "BIGINT"],
)

write(
    f"{MB}/artist.parquet",
    ["id", "gid", "name"],
    [
        (201, "artist-gid-201", "Artist Two"),
        (204, "artist-gid-204", "Orphan Artist"),
        (205, "artist-gid-205", "Lead Artist"),
        (206, "artist-gid-206", "Feature Artist"),
        (207, "artist-gid-207", "Artist Seven"),
        (208, "artist-gid-208", "Artist Eight"),
        (209, "artist-gid-209", "Artist Nine"),
    ],
    ["BIGINT", "VARCHAR", "VARCHAR"],
)

print("\nMB url / link tables (new)")
write(
    f"{MB}/url.parquet",
    ["id", "url"],
    [
        (9001, "https://open.spotify.com/track/TRACKID000000000000002"),
        (9002, "https://open.spotify.com/track/TRACKID000000000000003"),
        (9003, "spotify:track:TRACKID000000000000004"),
        (9004, "https://open.spotify.com/track/TRACKID000000000000005"),
        (9005, "https://open.spotify.com/artist/ARTISTID00000000000001"),
    ],
    ["BIGINT", "VARCHAR"],
)
write(
    f"{MB}/l_recording_url.parquet",
    ["entity0", "entity1"],
    [
        (3, 9001),  # item 2, winning URL candidate (has AF, higher pop)
        (3, 9002),  # item 2, losing URL candidate (no AF row at all)
        (5, 9003),  # item 3, the same track as its ISRC. The URL defect skips it.
        (6, 9004),
    ],  # item 4, its only candidate, after ISRC is thrown out
    ["BIGINT", "BIGINT"],
)
write(
    f"{MB}/l_artist_url.parquet",
    ["entity0", "entity1"],
    [(206, 9005)],
    ["BIGINT", "BIGINT"],
)
write(
    f"{MB}/isrc.parquet",
    ["recording", "isrc"],
    [
        (2, "US1234567890"),  # item 1
        (5, "GB9876543210"),  # item 3, the same track as its URL
        (6, "XX0000000000"),  # item 4 claims this ISRC too
        (7, "XX0000000000"),
    ],  # item 5, the same ISRC as item 4: ambiguous
    ["BIGINT", "VARCHAR"],
)

print("\nKaggle catalogue tables (new): numerics written as VARCHAR on purpose,")
print("matching the real source: BuildFeatures.py's own warning is that these")
print("columns are all-VARCHAR and a plain CAST is required, never TRY_CAST.")
TRACKS = [
    ("TRACKID000000000000001", 101, "US1234567890", "50", "false", "2025-05-01"),
    ("TRACKID000000000000002", 102, None, "80", "false", "2025-05-01"),
    ("TRACKID000000000000004", 103, "GB9876543210", "60", "true", "2025-05-01"),
    ("TRACKID000000000000005", 104, None, "40", "false", "2025-05-01"),
]
write(
    f"{KAG}/tracks.parquet",
    ["id", "rowid", "external_id_isrc", "popularity", "explicit", "fetched_at"],
    TRACKS,
    ["VARCHAR", "BIGINT", "VARCHAR", "VARCHAR", "VARCHAR", "VARCHAR"],
)

AF_COLS = [
    "track_id",
    "null_response",
    "duration_ms",
    "tempo",
    "key",
    "mode",
    "time_signature",
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
]
AF_ROWS = [
    (
        "TRACKID000000000000001",
        "false",
        "200500",
        "120.0",
        "5",
        "1",
        "4",
        "0.70",
        "0.80",
        "-5.0",
        "0.05",
        "0.10",
        "0.00",
        "0.15",
        "0.60",
    ),
    (
        "TRACKID000000000000002",
        "false",
        "199000",
        "118.5",
        "7",
        "1",
        "4",
        "0.65",
        "0.75",
        "-6.0",
        "0.04",
        "0.12",
        "0.01",
        "0.12",
        "0.55",
    ),
    (
        "TRACKID000000000000004",
        "false",
        "184800",
        "95.0",
        "2",
        "0",
        "4",
        "0.50",
        "0.60",
        "-7.5",
        "0.03",
        "0.30",
        "0.05",
        "0.20",
        "0.40",
    ),
    # item 4's track: flagged empty. Null_response=true and every feature
    # column genuinely null: the probe in SpotifyBridge.py checks exactly
    # this alignment (f1 == f2) and should print clean, no warning.
    (
        "TRACKID000000000000005",
        "true",
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    ),
]
write(
    f"{KAGAF}/track_audio_features.parquet",
    AF_COLS,
    AF_ROWS,
    ["VARCHAR"] + ["VARCHAR"] * 14,
)

write(
    f"{KAG}/track_artists.parquet",
    ["track_rowid", "artist_rowid"],
    [(101, 2001)],
    ["BIGINT", "BIGINT"],
)
write(
    f"{KAG}/artist_genres.parquet",
    ["artist_rowid", "genre"],
    [(2001, "electropop"), (2099, "electronic")],
    ["BIGINT", "VARCHAR"],
)
write(
    f"{KAG}/artists.parquet",
    ["rowid", "id"],
    [(2001, "ARTISTID00000000000002"), (2099, "ARTISTID00000000000099")],
    ["BIGINT", "VARCHAR"],
)

redirect_path = f"{ROOT}/kaggle/spotify-catalogue/spotify_artist_redirects.json"
redirects = [
    {"from_id": "ARTISTID00000000000001", "to_id": "ARTISTID00000000000099"},
    # an inert 2-hop chain that never touches any of the 6 items. It is only
    # here to make n_chained > 0 and prove the warning fires correctly,
    # without dragging artist 206's own (clean, single-hop) case into it.
    {"from_id": "ARTISTID00000000000090", "to_id": "ARTISTID00000000000091"},
    {"from_id": "ARTISTID00000000000091", "to_id": "ARTISTID00000000000092"},
]
with open(redirect_path, "w") as f:
    json.dump(redirects, f)
print(f"  wrote {redirect_path}  ({len(redirects)} rows)")

print(f"\nExtension written under {ROOT}")

r"""
USE

1. SpotifyBridge.py reads its data folder from THESIS_DATA_ROOT, as
   BuildFeatures.py does.

2. Run this script on the same fixture root that build_fixture.py wrote:

     python3 extend_fixture_spotify.py /tmp/thesis_fixture

3. Run SpotifyBridge.py on it:

     THESIS_DATA_ROOT=/tmp/thesis_fixture python3 SpotifyBridge.py --counts

EXPECTED OUTPUT, in the order it prints. These are the values of a recorded
run on this fixture.

    discovery: every "OK" line prints, RED file OK, no assertion failure.

    MB spotify track urls: 3 rows, 3 distinct tracks
    recording-track pairs: 3

    MB side: 4 items carry ISRCs (3 distinct, 4 pairs)
    ambiguous ISRCs (>1 item claims them): 1 (excluded and counted, not dropped silently)
    Spotify side: 2 MB ISRCs found on 2 Spotify tracks
    url/isrc disagreements: 0 (resolved to ISRC by precedence)

    null_response probe: 1 flagged (1 with null features), 0 unflagged-with-null-features
    (no "!!" warning line: f1 and f2 both equal 1)

    path B (via track artists): 1 items
    path A: MB artist links: 1 MB artists
    artist redirects: 3 (chained: 1)
    !! chained redirects exist: the one-hop alias table cannot resolve them,
       so genre coverage will silently miss the chained artists
    (this warning is expected: the inert 90/91/92 chain fires it, not a
    fault in artist 206's own resolution)

    Final identity checks (always run)
    map: 4 rows / 4 distinct items (must be equal; expect 1,020,242)
      method mix: isrc 2, url 2
    map coverage: 4 items, 66.67% distinct, 80.0% of listens

    features: 3 rows / 3 distinct items (must be equal; expect 1,017,150)
    features coverage: 3 items, 50.0% distinct, 70.0% of listens

    duration validation: n=3, median=500.0 ms, p90=7300.0 ms, 66.7% within 2s

    genres: 3 items, 2 genres (expect 2,814,416 / 684)
    genres coverage: 3 items, 50.0% distinct, 70.0% of listens

    combined vocabulary (MB tags + Spotify genres): 3 items, 50.0% distinct, 70.0% of listens

The recorded run differs from the design above in one place, because of a
defect in the URL step of SpotifyBridge.py. The step takes the track id
from the web form of a link (open.spotify.com or play.spotify.com) and is
meant to fall back to the spotify:track: form. But regexp_extract returns
an empty string, not NULL, when its pattern does not match, so coalesce()
never reaches the second pattern. The artist step has the same defect for
the spotify:artist: form. So the URL table holds 3 track URLs, not 4, and
item 3 matches by ISRC alone, not by ISRC and URL. The defect was found
after the evaluation, so SpotifyBridge.py keeps the code as it ran.

features coverage, genres coverage and the combined vocabulary line all
print the same 50.0% / 70.0%. This follows from which items land in each
set (features on items 1/2/3, genres on 1/3/4, tags only on item 4): items
2 and 4 carry exactly 1 listen each in this fixture, so two different
3-item sets sum to the same total. With six items such coincidences are
common.
"""
