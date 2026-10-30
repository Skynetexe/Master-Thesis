#!/usr/bin/env python3
"""Extend the typed audio-feature table through the
fuzzy tier (basis='name+duration' only; name_only is display-tier and
contributes nothing).

Output: features/item_audio_features_ext.parquet = map rows (match_method
preserved) plus the fuzzy name+duration rows (match_method='fuzzy'). The
map-only table stays as it is, so a comparison of map-only and extended
features needs only another file name.

Expected (frozen-snapshot arithmetic): map rows 1,017,150 (unchanged);
fuzzy rows <= 2,393,709 (derived at run time, and fewer after the
null_response filter); total items, distinct % and listen % are printed
and compared against 15.32/67.54 (map-only). The differences are the
contribution of the fuzzy tier.
"""

import duckdb, os


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THESIS_DATA_ROOT", os.path.join(CODE_DIR, "data"))
FEAT = f"{ROOT}/parquet/features"
KAG = f"{ROOT}/kaggle/spotify-catalogue"
AF = f"{KAG}/spotify_clean_audio_features_parquet/track_audio_features.parquet"
TRK = f"{KAG}/spotify_clean_parquet/tracks.parquet"
OUT = f"{FEAT}/item_audio_features_ext.parquet"

con = duckdb.connect()
con.execute("SET threads=14")
con.execute("SET memory_limit='24GB'")
con.execute(f"SET temp_directory='{ROOT}/duckdb_tmp'")
con.execute("SET preserve_insertion_order=false")
con.execute("SET max_temp_directory_size='150GiB'")

# precondition 1: disjointness of the two bridges (measured, not assumed)
overlap = con.execute(f"""
    SELECT count(*) FROM read_parquet('{FEAT}/item_spotify_map.parquet') m
    JOIN read_parquet('{FEAT}/item_spotify_fuzzy.parquet') f USING (item_id)
""").fetchone()[0]
check(
    overlap == 0,
    f"bridges overlap on {overlap} items, so a UNION would count them twice",
)
print(f"[OK] bridges disjoint (0 shared items)")

# precondition 2: the existing table is what we think it is
n_map = con.execute(
    f"SELECT count(*) FROM read_parquet('{FEAT}/item_audio_features.parquet')"
).fetchone()[0]
check(
    n_map == 1_017_150, f"map-only features table has {n_map} rows, expected 1,017,150"
)
print(f"[OK] existing feature table: {n_map:,} rows")

if os.path.exists(OUT):
    print(f"[skip] {os.path.basename(OUT)} exists")
else:
    # fuzzy feature rows: name+duration only, typed casts, null_response filtered
    con.execute(f"""
        COPY (
            SELECT
                f.item_id, f.sp_track, 'fuzzy' AS match_method,
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
            FROM read_parquet('{FEAT}/item_spotify_fuzzy.parquet') f
            JOIN read_parquet('{AF}') af
                 ON af.track_id = f.sp_track
                AND NOT COALESCE(CAST(af.null_response AS BOOLEAN), FALSE)
            LEFT JOIN read_parquet('{TRK}') tr ON tr.id = f.sp_track
            WHERE f.basis = 'name+duration'
        ) TO '{FEAT}/fuzzy_features_tmp.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    n_fuzzy = con.execute(
        f"SELECT count(*) FROM read_parquet('{FEAT}/fuzzy_features_tmp.parquet')"
    ).fetchone()[0]
    n_total = con.execute(
        f"SELECT count(*) FROM read_parquet('{FEAT}/item_spotify_fuzzy.parquet') "
        f"WHERE basis = 'name+duration'"
    ).fetchone()[0]
    print(
        f"[done] fuzzy feature rows: {n_fuzzy:,} of {n_total:,} name+duration matches "
        f"({n_total - n_fuzzy:,} lacked usable AF rows)"
    )

    # final table: map rows (match_method as stored) UNION fuzzy rows
    con.execute(f"""
        COPY (
            SELECT item_id, sp_track, match_method, duration_ms, tempo,
                   "key", "mode", time_signature, danceability, energy,
                   loudness, speechiness, acousticness, instrumentalness,
                   liveness, valence, sp_popularity, explicit
            FROM read_parquet('{FEAT}/item_audio_features.parquet')
            UNION ALL
            SELECT item_id, sp_track, 'fuzzy', duration_ms, tempo,
                   "key", "mode", time_signature, danceability, energy,
                   loudness, speechiness, acousticness, instrumentalness,
                   liveness, valence, sp_popularity, explicit
            FROM read_parquet('{FEAT}/fuzzy_features_tmp.parquet')
        ) TO '{OUT}' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)
    os.remove(f"{FEAT}/fuzzy_features_tmp.parquet")

# Always-run identity checks
rows, items = con.execute(
    f"SELECT count(*), count(DISTINCT item_id) FROM read_parquet('{OUT}')"
).fetchone()
check(rows == items, f"not 1:1 by item: {rows:,} rows, {items:,} items")
print(f"[OK] extended table is 1:1 by item: {items:,} rows")

mm = con.execute(
    f"SELECT match_method, count(*) FROM read_parquet('{OUT}') GROUP BY 1 ORDER BY 2 DESC"
).fetchall()
print("  method mix: " + ", ".join(f"{m} {n:,}" for m, n in mm))

con.sql(f"""
SELECT count(*) AS items,
       count(*) FILTER (WHERE x.item_id IS NOT NULL) AS with_features,
       round(100.0 * count(*) FILTER (WHERE x.item_id IS NOT NULL) / count(*), 2) AS distinct_pct,
       round(100.0 * sum(l.listens) FILTER (WHERE x.item_id IS NOT NULL) / sum(l.listens), 2) AS listen_pct
FROM read_parquet('{FEAT}/item_ids.parquet') i
JOIN read_parquet('{FEAT}/item_listens.parquet') l ON l.canonical_recording_mbid = i.canonical_recording_mbid
LEFT JOIN (SELECT DISTINCT item_id FROM read_parquet('{OUT}')) x ON x.item_id = i.item_id
""").show()  # map-only baseline: 15.32% / 67.54%; bridge ceiling: 61.04/94.52
