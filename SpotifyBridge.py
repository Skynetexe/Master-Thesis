#!/usr/bin/env python3
"""Rebuild the entire Spotify integration layer: the URL-tier candidates,
the ISRC crosswalk, the typed audio features, and the two-path genres with
the artist map.

Status: it brings together queries that were each run and checked on their
own. It has not run end to end as one script. With all outputs present, it
only runs the identity checks (a dry run), as BuildFeatures.py does.

Idempotent: every step checks for its output first; delete an output file
to rebuild just that step. Deleting item_spotify_map.parquet also rebuilds
item_spotify_isrc_candidates.parquet (same step). Deleting
item_spotify_genres.parquet also rebuilds artist_spotify_map.parquet.

Prerequisites (asserted at start): MB-phase outputs in features/
(recording_canonical_map, item_ids, item_listens, item_artists,
item_tags, item_tags_artist); MB tables url / l_recording_url /
l_artist_url / recording / artist / isrc; the Kaggle catalogue.
The Hugging Face copy is not used: this script never reads or writes
huggingface/.

Kaggle numerics in the audio-features table are VARCHAR: all casts are
built in and fail loudly on garbage (plain CAST, never TRY_CAST).

Frozen-snapshot expectations (the MusicBrainz snapshot of 2026, the Kaggle
scrape of 2025): map 1,020,242 items (isrc 910,895 / isrc+url 99,559 /
url 9,788); features 1,017,150 items, duration median 160 ms;
genres 2,814,416 items / 684 genres; coverage 15.37/68.65, 15.32/67.54,
42.4/74.11; combined vocabulary 67.3/95.93. A data refresh re-derives all
expectations from the printed components. Investigate any failure.

Usage:
    python3 SpotifyBridge.py             # dry run / incremental build
    python3 SpotifyBridge.py --counts    # also count source-table rows
"""

import argparse, duckdb, os, time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THESIS_DATA_ROOT", os.path.join(CODE_DIR, "data"))
KAG = f"{ROOT}/kaggle/spotify-catalogue"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
FEAT = f"{ROOT}/parquet/features"
TMP = f"{ROOT}/duckdb_tmp"
for d in (FEAT, TMP):
    os.makedirs(d, exist_ok=True)

TRK = f"{KAG}/spotify_clean_parquet/tracks.parquet"
AF = f"{KAG}/spotify_clean_audio_features_parquet/track_audio_features.parquet"
TA = f"{KAG}/spotify_clean_parquet/track_artists.parquet"
AG = f"{KAG}/spotify_clean_parquet/artist_genres.parquet"
ART = f"{KAG}/spotify_clean_parquet/artists.parquet"
RED = f"{KAG}/spotify_artist_redirects.json"

MAP = f"{FEAT}/item_spotify_map.parquet"
CAND = f"{FEAT}/item_spotify_candidates.parquet"
ISRC = f"{FEAT}/item_spotify_isrc_candidates.parquet"
FEATS = f"{FEAT}/item_audio_features.parquet"
GEN = f"{FEAT}/item_spotify_genres.parquet"
AMAP = f"{FEAT}/artist_spotify_map.parquet"

ap = argparse.ArgumentParser()
ap.add_argument(
    "--counts",
    action="store_true",
    help="also count source-table rows in discovery (~1 min)",
)
ARGS = ap.parse_args()

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='24GB'")
con.execute(f"SET temp_directory='{TMP}'")
con.execute("SET preserve_insertion_order=false")
con.execute(
    "SET max_temp_directory_size='150GiB'"
)  # runaway queries error, not fill the disk


def ladder(sql, out, label):
    for th in (4, 2, 1):
        try:
            con.execute(f"SET threads={th}")
            con.execute(sql)
            return th
        except duckdb.Error as e:
            if "Out of Memory" not in str(e):
                raise
            if os.path.exists(out):
                os.remove(out)
            print(f"  [retry] OOM at {th} threads: {label}")
    raise SystemExit(f"OOM at 1 thread: {label}")


def copy_to(path, sql, label):
    if os.path.exists(path):
        print(f"[skip] {label}")
        return
    t0 = time.time()
    ladder(
        f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, COMPRESSION 'zstd')", path, label
    )
    print(f"[done] {label} ({time.time()-t0:,.0f}s)")


def need(path, cols):
    have = {
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
    }
    missing = set(cols) - have
    assert not missing, f"{path} missing columns {missing}"


def coverage(side_sql, label, expect=None):
    """The standard coverage shape: the concentration-law measurement."""
    row = con.execute(f"""
        SELECT count(*) FILTER (WHERE x.item_id IS NOT NULL),
               round(100.0 * count(*) FILTER (WHERE x.item_id IS NOT NULL) / count(*), 2),
               round(100.0 * sum(l.listens) FILTER (WHERE x.item_id IS NOT NULL) / sum(l.listens), 2)
        FROM read_parquet('{FEAT}/item_ids.parquet') i
        JOIN read_parquet('{FEAT}/item_listens.parquet') l ON l.canonical_recording_mbid = i.canonical_recording_mbid
        LEFT JOIN ({side_sql}) x ON x.item_id = i.item_id
    """).fetchone()
    print(
        f"  {label}: {row[0]:,} items, {row[1]}% distinct, {row[2]}% of listens"
        + (f"  (expect {expect})" if expect else "")
    )
    return row


# Discovery
print("Discovery (schema; --counts adds row counts)")
for path, want in [
    (f"{MB}/url.parquet", {"id", "url"}),
    (f"{MB}/l_recording_url.parquet", {"entity0", "entity1"}),
    (f"{MB}/l_artist_url.parquet", {"entity0", "entity1"}),
    (f"{MB}/recording.parquet", {"id", "gid", "length"}),
    (f"{MB}/artist.parquet", {"id", "gid", "name"}),
    (f"{MB}/isrc.parquet", {"recording", "isrc"}),
    (TRK, {"id", "rowid", "external_id_isrc", "popularity", "explicit", "fetched_at"}),
    (
        AF,
        {
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
        },
    ),
    (TA, {"track_rowid", "artist_rowid"}),
    (AG, {"artist_rowid", "genre"}),
    (ART, {"rowid", "id"}),
    (
        f"{FEAT}/recording_canonical_map.parquet",
        {"recording_mbid", "canonical_recording_mbid"},
    ),
    (f"{FEAT}/item_ids.parquet", {"item_id", "canonical_recording_mbid"}),
    (f"{FEAT}/item_listens.parquet", {"canonical_recording_mbid", "listens"}),
    (f"{FEAT}/item_artists.parquet", {"item_id", "artist_id"}),
    (f"{FEAT}/item_tags.parquet", {"item_id"}),
    (f"{FEAT}/item_tags_artist.parquet", {"item_id"}),
]:
    need(path, want)
    print(f"  OK {os.path.basename(path)}")
assert os.path.exists(RED), f"missing {RED}"
print(f"  OK {os.path.basename(RED)}")

# print the types of the feature table, not just its column names
af_types = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{AF}')").fetchall()
print("  AF types: " + ", ".join(f"{c[0]}:{c[1]}" for c in af_types))

if ARGS.counts:
    for path in (TRK, AF, TA, AG, ART):
        n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
        print(f"  {os.path.basename(path)}: {n:,} rows")

# 1. URL-tier candidates (1:N)
if not os.path.exists(CAND):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sp_url AS
    SELECT url_id, sp_track FROM (
        SELECT id AS url_id,
               coalesce(
                   regexp_extract(url, 'https?://(?:open|play)\\.spotify\\.com/(?:intl-[^/]+/)?track/([A-Za-z0-9]{{22}})', 1),
                   regexp_extract(url, 'spotify:track:([A-Za-z0-9]{{22}})', 1)
               ) AS sp_track
        FROM read_parquet('{MB}/url.parquet')
        WHERE url LIKE '%spotify%'
    ) t WHERE sp_track <> ''
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE rec_sp AS
    SELECT DISTINCT l.entity0 AS rec_id, s.sp_track
    FROM read_parquet('{MB}/l_recording_url.parquet') l
    JOIN sp_url s ON s.url_id = l.entity1
    """)
    n, d = con.execute(
        "SELECT count(*), count(DISTINCT sp_track) FROM sp_url"
    ).fetchone()
    print(f"  MB spotify track urls: {n:,} rows, {d:,} distinct tracks")
    print(
        f"  recording-track pairs: {con.execute('SELECT count(*) FROM rec_sp').fetchone()[0]:,}"
    )
    copy_to(
        CAND,
        f"""
        SELECT it.item_id, rec.gid AS recording_mbid, x.sp_track
        FROM (SELECT rec_id, sp_track FROM rec_sp) x
        JOIN read_parquet('{MB}/recording.parquet') rec ON rec.id = x.rec_id
        JOIN read_parquet('{FEAT}/recording_canonical_map.parquet') m ON m.recording_mbid = rec.gid
        JOIN read_parquet('{FEAT}/item_ids.parquet') it ON it.canonical_recording_mbid = m.canonical_recording_mbid
    """,
        "URL-tier candidates",
    )
else:
    print("[skip] URL-tier candidates")

# 2. ISRC chain + map assembly (1:1 by item)
if not os.path.exists(MAP):
    # uniqueness guard: the map's LEFT JOINs fan out silently if these
    # ids ever stop being unique (peek verified rows == distinct once; assert every build)
    for path, col in ((TRK, "id"), (AF, "track_id")):
        n, d = con.execute(
            f'SELECT count(*), count(DISTINCT "{col}") FROM read_parquet("{path}")'
        ).fetchone()
        assert (
            n == d
        ), f"{path}: duplicate {col} ({n:,} vs {d:,}), so the map assembly would fan out"

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE mb_item_isrc AS
    SELECT DISTINCT it.item_id, norm.isrc
    FROM (
        SELECT recording, upper(regexp_replace(isrc, '[^A-Za-z0-9]', '', 'g')) AS isrc
        FROM read_parquet('{MB}/isrc.parquet')
        WHERE regexp_replace(isrc, '[^A-Za-z0-9]', '', 'g') ~ '^[A-Za-z0-9]{{12}}$'
    ) norm
    JOIN read_parquet('{MB}/recording.parquet') rec ON rec.id = norm.recording
    JOIN read_parquet('{FEAT}/recording_canonical_map.parquet') m ON m.recording_mbid = rec.gid
    JOIN read_parquet('{FEAT}/item_ids.parquet') it ON it.canonical_recording_mbid = m.canonical_recording_mbid
    """)
    a, b, c = con.execute(
        "SELECT count(DISTINCT item_id), count(DISTINCT isrc), count(*) FROM mb_item_isrc"
    ).fetchone()
    print(f"  MB side: {a:,} items carry ISRCs ({b:,} distinct, {c:,} pairs)")

    con.execute(
        "CREATE OR REPLACE TEMP TABLE ambiguous_isrc AS "
        "SELECT isrc FROM mb_item_isrc GROUP BY isrc HAVING count(DISTINCT item_id) > 1"
    )
    n_amb = con.execute("SELECT count(*) FROM ambiguous_isrc").fetchone()[0]
    print(
        f"  ambiguous ISRCs (>1 item claims them): {n_amb:,} (excluded and counted, not dropped silently)"
    )

    # corpus discipline: semi-join the 256M Spotify isrc column against the
    # small MB key set before any DISTINCT
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sp_isrc_hits AS
    SELECT DISTINCT s.sp_track, s.isrc
    FROM (
        SELECT id AS sp_track, upper(regexp_replace(external_id_isrc, '[^A-Za-z0-9]', '', 'g')) AS isrc
        FROM read_parquet('{TRK}')
        WHERE external_id_isrc IS NOT NULL
    ) s
    JOIN (SELECT DISTINCT isrc FROM mb_item_isrc
          WHERE isrc NOT IN (SELECT isrc FROM ambiguous_isrc)) k ON k.isrc = s.isrc
    """)
    h, k = con.execute(
        "SELECT count(DISTINCT sp_track), count(DISTINCT isrc) FROM sp_isrc_hits"
    ).fetchone()
    print(f"  Spotify side: {k:,} MB ISRCs found on {h:,} Spotify tracks")

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE allp AS
    SELECT item_id, sp_track, 'isrc' AS m FROM (
        SELECT DISTINCT mi.item_id, hi.sp_track
        FROM mb_item_isrc mi JOIN sp_isrc_hits hi ON hi.isrc = mi.isrc
        WHERE mi.isrc NOT IN (SELECT isrc FROM ambiguous_isrc)
    ) x
    UNION ALL
    SELECT DISTINCT item_id, sp_track, 'url' FROM read_parquet('{CAND}')
    """)
    n_dis = con.execute("""
        SELECT count(*) FROM (SELECT item_id FROM allp GROUP BY item_id
        HAVING count(DISTINCT m) = 2 AND count(*) = count(DISTINCT sp_track))
    """).fetchone()[0]
    print(f"  url/isrc disagreements: {n_dis:,} (resolved to ISRC by precedence)")

    # precedence: isrc+url > isrc > url, then has-features, then popularity
    copy_to(
        MAP,
        f"""
        SELECT p.item_id, p.sp_track, p.match_method,
               count(*) OVER (PARTITION BY p.item_id) - 1 AS n_alt_tracks
        FROM (
            SELECT item_id, sp_track,
                   CASE WHEN count(*) = 2 THEN 'isrc+url' ELSE min(m) END AS match_method
            FROM allp GROUP BY item_id, sp_track
        ) p
        LEFT JOIN read_parquet('{AF}') af ON af.track_id = p.sp_track
        LEFT JOIN read_parquet('{TRK}') tr ON tr.id = p.sp_track
        QUALIFY row_number() OVER (
            PARTITION BY p.item_id
            ORDER BY CASE p.match_method WHEN 'isrc+url' THEN 3 WHEN 'isrc' THEN 2 ELSE 1 END DESC,
                     (af.track_id IS NOT NULL) DESC,
                     CAST(tr.popularity AS DOUBLE) DESC NULLS LAST
        ) = 1
    """,
        "Spotify crosswalk map",
    )

    copy_to(
        ISRC,
        "SELECT DISTINCT item_id, sp_track FROM allp WHERE m = 'isrc'",
        "ISRC-tier candidates (provenance)",
    )
else:
    print("[skip] Spotify crosswalk map (+ ISRC candidates)")

# 3. typed audio features (1:1 by item)
if not os.path.exists(FEATS):
    # null_response semantics probe: the filter below is only correct while
    # flagged rows are exactly the empty-feature rows
    f1, f2, f3 = con.execute(f"""
        SELECT count(*) FILTER (WHERE CAST(null_response AS BOOLEAN)),
               count(*) FILTER (WHERE CAST(null_response AS BOOLEAN) AND danceability IS NULL),
               count(*) FILTER (WHERE NOT CAST(null_response AS BOOLEAN) AND danceability IS NULL)
        FROM read_parquet('{AF}')
    """).fetchone()
    print(
        f"  null_response probe: {f1:,} flagged ({f2:,} with null features), "
        f"{f3:,} unflagged-with-null-features"
    )
    if f2 != f1:
        print("  !! null_response does not mark exactly the empty-feature rows, so the")
        print(
            "     filter below is not correct for this data. Inspect it before you trust the build."
        )

    copy_to(
        FEATS,
        f"""
        SELECT
            m.item_id, m.sp_track, m.match_method,
            CAST(af.duration_ms      AS BIGINT)  AS duration_ms,
            CAST(af.tempo            AS DOUBLE)  AS tempo,
            CAST(af."key"            AS INTEGER) AS "key",
            CAST(af.mode             AS INTEGER) AS "mode",
            CAST(af.time_signature   AS INTEGER) AS time_signature,
            CAST(af.danceability     AS DOUBLE)  AS danceability,
            CAST(af.energy           AS DOUBLE)  AS energy,
            CAST(af.loudness         AS DOUBLE)  AS loudness,
            CAST(af.speechiness      AS DOUBLE)  AS speechiness,
            CAST(af.acousticness     AS DOUBLE)  AS acousticness,
            CAST(af.instrumentalness AS DOUBLE)  AS instrumentalness,
            CAST(af.liveness         AS DOUBLE)  AS liveness,
            CAST(af.valence          AS DOUBLE)  AS valence,
            CAST(tr.popularity       AS INTEGER) AS sp_popularity,
            CAST(tr.explicit         AS BOOLEAN) AS explicit
        FROM read_parquet('{MAP}') m
        JOIN read_parquet('{AF}') af
             ON af.track_id = m.sp_track
            AND NOT COALESCE(CAST(af.null_response AS BOOLEAN), FALSE)
        LEFT JOIN read_parquet('{TRK}') tr ON tr.id = m.sp_track
    """,
        "typed audio features",
    )
else:
    print("[skip] typed audio features")

# 4. two-path genres + artist map
if not os.path.exists(GEN):
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE item_sp_genres_track AS
    SELECT DISTINCT m.item_id, g.genre
    FROM read_parquet('{MAP}') m
    JOIN (SELECT rowid, id FROM read_parquet('{TRK}') ) t ON t.id = m.sp_track
    JOIN read_parquet('{TA}') ta ON ta.track_rowid = t.rowid
    JOIN read_parquet('{AG}') g ON g.artist_rowid = ta.artist_rowid
    """)
    print(
        f"  path B (via track artists): "
        f"{con.execute('SELECT count(DISTINCT item_id) FROM item_sp_genres_track').fetchone()[0]:,} items"
    )

    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sp_artist_url AS
    SELECT url_id, sp_artist FROM (
        SELECT id AS url_id,
               coalesce(
                   regexp_extract(url, 'https?://(?:open|play)\\.spotify\\.com/(?:intl-[^/]+/)?artist/([A-Za-z0-9]{{22}})', 1),
                   regexp_extract(url, 'spotify:artist:([A-Za-z0-9]{{22}})', 1)
               ) AS sp_artist
        FROM read_parquet('{MB}/url.parquet')
        WHERE url LIKE '%spotify%'
    ) t WHERE sp_artist <> ''
    """)
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE artist_sp AS
    SELECT DISTINCT l.entity0 AS mb_artist_id, s.sp_artist
    FROM read_parquet('{MB}/l_artist_url.parquet') l
    JOIN sp_artist_url s ON s.url_id = l.entity1
    """)
    print(
        f"  path A: MB artist links: "
        f"{con.execute('SELECT count(DISTINCT mb_artist_id) FROM artist_sp').fetchone()[0]:,} MB artists"
    )

    con.execute(
        f"CREATE OR REPLACE TEMP TABLE sp_redirect AS "
        f"SELECT from_id, to_id FROM read_json_auto('{RED}')"
    )
    n_red, n_chained = con.execute("""
        SELECT count(*), (SELECT count(*) FROM sp_redirect a JOIN sp_redirect b ON a.to_id = b.from_id)
        FROM sp_redirect
    """).fetchone()
    print(f"  artist redirects: {n_red:,} (chained: {n_chained:,})")
    if n_chained > 0:
        print(
            "  !! chained redirects exist: the one-hop alias table cannot resolve them,"
        )
        print("     so genre coverage will silently miss the chained artists")

    # alias table: every catalogue artist's own id + each merged-away from_id
    # pointing at its successor's row
    con.execute(f"""
    CREATE OR REPLACE TEMP TABLE sp_alias AS
    SELECT ar.id AS sp_artist, ar.rowid
    FROM read_parquet('{ART}') ar
    UNION
    SELECT r.from_id AS sp_artist, ar.rowid
    FROM sp_redirect r
    JOIN read_parquet('{ART}') ar ON ar.id = r.to_id
    """)

    copy_to(
        GEN,
        f"""
        SELECT item_id, genre FROM item_sp_genres_track
        UNION
        SELECT DISTINCT ia.item_id, g.genre
        FROM read_parquet('{FEAT}/item_artists.parquet') ia
        JOIN artist_sp a ON a.mb_artist_id = ia.artist_id
        JOIN sp_alias al ON al.sp_artist = a.sp_artist
        JOIN read_parquet('{AG}') g ON g.artist_rowid = al.rowid
    """,
        "Spotify genres (two paths)",
    )

    copy_to(
        AMAP,
        f"""
        SELECT a.mb_artist_id, mb.gid AS artist_mbid, mb.name AS mb_name,
               a.sp_artist, coalesce(r.to_id, a.sp_artist) AS sp_artist_resolved
        FROM artist_sp a
        LEFT JOIN sp_redirect r ON r.from_id = a.sp_artist
        JOIN read_parquet('{MB}/artist.parquet') mb ON mb.id = a.mb_artist_id
    """,
        "artist map",
    )
else:
    print("[skip] Spotify genres (+ artist map)")

# Final identity checks
print("\nFinal identity checks (always run)")
rows, items = con.execute(
    f"SELECT count(*), count(DISTINCT item_id) FROM read_parquet('{MAP}')"
).fetchone()
mm = con.execute(
    f"SELECT match_method, count(*) FROM read_parquet('{MAP}') GROUP BY 1 ORDER BY 2 DESC"
).fetchall()
print(
    f"map: {rows:,} rows / {items:,} distinct items (must be equal; expect 1,020,242)"
)
print(
    "  method mix: "
    + ", ".join(f"{m} {n:,}" for m, n in mm)
    + "  (expect isrc 910,895 / isrc+url 99,559 / url 9,788)"
)
assert (
    rows == items and sum(n for _, n in mm) == rows
), "map is not 1:1 or method mix does not sum"
coverage(
    f"SELECT DISTINCT item_id FROM read_parquet('{MAP}')",
    "map coverage",
    "15.37% / 68.65%",
)

frows, fitems = con.execute(
    f"SELECT count(*), count(DISTINCT item_id) FROM read_parquet('{FEATS}')"
).fetchone()
print(
    f"features: {frows:,} rows / {fitems:,} distinct items (must be equal; expect 1,017,150)"
)
assert frows == fitems, "features is not 1:1 by item"
assert fitems <= items, "features items exceed map items (impossible by construction)"
coverage(
    f"SELECT DISTINCT item_id FROM read_parquet('{FEATS}')",
    "features coverage",
    "15.32% / 67.54%",
)

# physical validation of the crosswalk: MB length vs Spotify duration
n, med, p90, p2 = con.execute(f"""
    WITH d AS (
        -- one raw gid per item, deterministically: smallest recording_mbid.
        -- (DISTINCT ON without ORDER BY picks an arbitrary row per item, so
        -- n would change from run to run.)
        SELECT abs(CAST(rec.length AS BIGINT) - f.duration_ms) AS diff_ms
        FROM read_parquet('{FEATS}') f
        JOIN (
            SELECT DISTINCT ON (item_id) item_id, recording_mbid
            FROM (SELECT m.recording_mbid, it.item_id
                  FROM read_parquet('{FEAT}/recording_canonical_map.parquet') m
                  JOIN read_parquet('{FEAT}/item_ids.parquet') it
                       ON it.canonical_recording_mbid = m.canonical_recording_mbid)
            ORDER BY item_id, recording_mbid
        ) r ON r.item_id = f.item_id
        JOIN read_parquet('{MB}/recording.parquet') rec ON rec.gid = r.recording_mbid
        WHERE rec.length IS NOT NULL
    )
    SELECT count(*), median(diff_ms), quantile_cont(diff_ms, 0.90),
           round(100.0 * count(*) FILTER (WHERE diff_ms < 2000) / count(*), 1)
    FROM d
""").fetchone()
print(
    f"duration validation: n={n:,}, median={med} ms, p90={p90} ms, "
    f"{p2}% within 2s (expect 1,008,446 / 160 / 2,253 / 87.3 with the deterministic smallest-gid pick)"
)

gitems = con.execute(
    f"SELECT count(DISTINCT item_id) FROM read_parquet('{GEN}')"
).fetchone()[0]
gvocab = con.execute(
    f"SELECT count(DISTINCT genre) FROM read_parquet('{GEN}')"
).fetchone()[0]
print(f"genres: {gitems:,} items, {gvocab:,} genres (expect 2,814,416 / 684)")
coverage(
    f"SELECT DISTINCT item_id FROM read_parquet('{GEN}')",
    "genres coverage",
    "42.4% / 74.11%",
)

vocab = coverage(
    f"""
    SELECT DISTINCT item_id FROM read_parquet('{FEAT}/item_tags.parquet')
    UNION SELECT DISTINCT item_id FROM read_parquet('{FEAT}/item_tags_artist.parquet')
    UNION SELECT DISTINCT item_id FROM read_parquet('{GEN}')
""",
    "combined vocabulary (MB tags + Spotify genres)",
    "67.3% / 95.93%",
)

print(
    "\nOutputs:",
    ", ".join(os.path.basename(p) for p in (CAND, MAP, ISRC, FEATS, GEN, AMAP)),
)
print("Rebuild one artifact: delete its file and re-run. Full rebuild: delete all six.")
