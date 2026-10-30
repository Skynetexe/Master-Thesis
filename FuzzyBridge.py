#!/usr/bin/env python3
"""Tier 3 of the Spotify bridge: a name and duration match for the corpus
items that the ISRC/URL map missed.

Additive by design: it builds only over items absent from item_spotify_map
(verified tiers are never overwritten or merged). Outputs a separate
item_spotify_fuzzy.parquet with match_method/basis provenance columns:
  basis = 'name+duration' (both sides have a duration; |diff| <= 2.5 s)
  basis = 'name_only'     (no duration agreement; lower confidence)
Use: JOIN with item_spotify_map at query time (never UNION rows blindly;
the items are disjoint anyway, because the build universe excludes mapped
items).

Intermediate stages are file-backed under parquet/scratch/, so a crashed
run resumes where it left off. Re-run costs nothing already written.

Obscurity note: match_method across tiers is a proxy for item obscurity
(ISRC < fuzzy name+duration < name_only < unmatched). Keep it as a column
into feature views; it is both a reweighting lever and an ablation switch.
"""

import argparse, duckdb, os, time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THESIS_DATA_ROOT", os.path.join(CODE_DIR, "data"))
FEAT = f"{ROOT}/parquet/features"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
KAG = f"{ROOT}/kaggle/spotify-catalogue"
SCR = f"{ROOT}/parquet/scratch"
FUZ = f"{FEAT}/item_spotify_fuzzy.parquet"
for d in (SCR,):
    os.makedirs(d, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument(
    "--fresh",
    action="store_true",
    help="rebuild even if outputs exist (deletes scratch + fuzzy files)",
)
ARGS = ap.parse_args()
if ARGS.fresh:
    import glob as _g

    for p in (
        f"{SCR}/mb_fuzzy.parquet",
        f"{SCR}/mb_keys.parquet",
        f"{SCR}/sp_stage2.parquet",
        FUZ,
        f"{SCR}/item_match.parquet",
        f"{SCR}/pick_dur.parquet",
    ):
        if os.path.exists(p):
            os.remove(p)
    for pat in (f"{SCR}/item_match_*.parquet", f"{SCR}/pick_dur_*.parquet"):
        for f in _g.glob(pat):
            os.remove(f)

con = duckdb.connect()
con.execute("SET threads=14")
con.execute("SET memory_limit='24GB'")
con.execute(f"SET temp_directory='{ROOT}/duckdb_tmp'")
con.execute("SET preserve_insertion_order=false")
con.execute("SET max_temp_directory_size='150GiB'")


def run(sql, label, heavy=False):
    """threads=14 by default; heavy steps step down to 4 and then 1 thread on OOM."""
    for th in ((14, 4, 1) if heavy else (14,)):
        try:
            con.execute(f"SET threads={th}")
            con.execute(sql)
            return
        except duckdb.Error as e:
            if "Out of Memory" not in str(e) or th == 1:
                raise
    raise SystemExit(f"OOM at 1 thread: {label}")


NORM = """regexp_replace(regexp_replace(regexp_replace(regexp_replace(
         lower(%s),
         '\\s+-\\s+.*$', '', 'g'),
         '\\([^)]*\\)|\\[[^]]*\\]', '', 'g'),
         '(feat\\.?|featuring|ft\\.)\\s*', '', 'g'),
         '[^a-z0-9]+', '', 'g')"""


def copy_if_missing(path, sql, label, heavy=False):
    if os.path.exists(path):
        print(f"[skip] {label}")
        return
    t0 = time.time()
    run(f"COPY ({sql}) TO '{path}' (FORMAT PARQUET, COMPRESSION 'zstd')", label, heavy)
    print(f"[done] {label} ({time.time()-t0:,.0f}s)")


# 0. MB fuzzy universe: corpus items not in the map, with names
MF = f"{SCR}/mb_fuzzy.parquet"
copy_if_missing(
    MF,
    f"""
    SELECT it.item_id, d.listens, c.artist_name, c.recording_name, L.length_ms
    FROM read_parquet('{FEAT}/item_ids.parquet') it
    JOIN read_parquet('{FEAT}/item_listens.parquet') d
         ON d.canonical_recording_mbid = it.canonical_recording_mbid
    LEFT JOIN (SELECT DISTINCT item_id FROM read_parquet('{FEAT}/item_spotify_map.parquet')) smp
         ON smp.item_id = it.item_id
    JOIN (
        SELECT canonical_recording_mbid, artist_name, recording_name
        FROM read_parquet('{FEAT}/item_catalog.parquet')
        QUALIFY row_number() OVER (PARTITION BY canonical_recording_mbid
                                   ORDER BY recording_name IS NOT NULL DESC,
                                            artist_name IS NOT NULL DESC, listens DESC) = 1
    ) c ON c.canonical_recording_mbid = it.canonical_recording_mbid
    LEFT JOIN (
        SELECT DISTINCT ON (item_id) item_id, CAST(rec.length AS BIGINT) AS length_ms
        FROM (SELECT m.recording_mbid, it2.item_id
              FROM read_parquet('{FEAT}/recording_canonical_map.parquet') m
              JOIN read_parquet('{FEAT}/item_ids.parquet') it2
                   ON it2.canonical_recording_mbid = m.canonical_recording_mbid) x
        JOIN read_parquet('{MB}/recording.parquet') rec ON rec.gid = x.recording_mbid
        ORDER BY item_id, recording_mbid
    ) L ON L.item_id = it.item_id
    WHERE smp.item_id IS NULL
      AND c.recording_name IS NOT NULL AND c.artist_name IS NOT NULL
""",
    "mb_fuzzy",
    heavy=True,
)
n_items, n_len = con.execute(
    f"SELECT count(*), count(length_ms) FROM read_parquet('{MF}')"
).fetchone()
print(f"  fuzzy universe: {n_items:,} items ({n_len:,} with MB length)")

# 1. normalized keys
MK = f"{SCR}/mb_keys.parquet"
copy_if_missing(
    MK,
    f"""
    SELECT item_id, length_ms,
           {NORM % 'artist_name'} AS akey, {NORM % 'recording_name'} AS tkey
    FROM read_parquet('{MF}')
          WHERE {NORM % 'artist_name'} <> '' AND {NORM % 'recording_name'} <> ''
""",
    "mb_keys",
)
n_dk = con.execute(
    f"SELECT count(DISTINCT akey || tkey) FROM read_parquet('{MK}')"
).fetchone()[0]
print(f"  distinct (artist,track) keys: {n_dk:,}")

# 2. Spotify candidates (the 256M scan; file-backed, ~3-5 min)
S2 = f"{SCR}/sp_stage2.parquet"
copy_if_missing(
    S2,
    f"""
    SELECT sp_track, duration_ms, popularity, tkey,
           {NORM % "string_agg(name, ' ' ORDER BY name)"} AS akey
    FROM (SELECT s.id AS sp_track, CAST(s.duration_ms AS BIGINT) AS duration_ms,
                 CAST(s.popularity AS INTEGER) AS popularity,
                 {NORM % 's.name'} AS tkey, ar.name
          FROM read_parquet('{KAG}/spotify_clean_parquet/tracks.parquet') s
          JOIN read_parquet('{KAG}/spotify_clean_parquet/track_artists.parquet') ta
               ON ta.track_rowid = s.rowid
          JOIN read_parquet('{KAG}/spotify_clean_parquet/artists.parquet') ar
               ON ar.rowid = ta.artist_rowid
          WHERE {NORM % 's.name'} <> '' AND {NORM % 's.name'} IN (SELECT tkey FROM read_parquet('{MK}'))) sub
    GROUP BY sp_track, duration_ms, popularity, tkey
""",
    "sp_stage2",
    heavy=True,
)
n_s1 = con.execute(f"SELECT count(*) FROM read_parquet('{S2}')").fetchone()[0]
print(f"  Spotify candidate tracks: {n_s1:,}")

# 3. pass 1: duration-verified candidates, in batches
# The naive single join explodes: key collisions on the Spotify side (108M
# candidates, names like 'home') expand to >150 GiB of intermediate before
# any filter applies, and a WHERE on the join output does not bound the
# expansion. So the MB items go in chunks: per chunk, the candidates are
# semi-joined to that chunk's keys before the join, and each batch stays small.
IM = f"{SCR}/item_match.parquet"
if os.path.exists(IM):
    print("[skip] item_match")
else:
    t0 = time.time()
    # rank items by id for stable, contiguous chunks
    n_total = con.execute(f"SELECT count(*) FROM read_parquet('{MK}')").fetchone()[0]
    CHUNK = 250_000
    first = True
    done = 0
    lo = 0
    while lo < n_total:
        hi = min(lo + CHUNK, n_total)
        sql = f"""
        COPY (
            SELECT k.item_id, s.sp_track,
                   abs(k.length_ms - s.duration_ms) AS duration_diff_ms,
                   s.popularity
            FROM (
                SELECT item_id, length_ms, akey, tkey
                FROM read_parquet('{MK}')
                ORDER BY item_id
                LIMIT {hi - lo} OFFSET {lo}
            ) k
            JOIN (
                SELECT sp_track, duration_ms, popularity, akey, tkey
                FROM read_parquet('{S2}')
                WHERE (akey, tkey) IN (
                    SELECT akey, tkey FROM (
                        SELECT item_id, length_ms, akey, tkey
                        FROM read_parquet('{MK}')
                        ORDER BY item_id
                        LIMIT {hi - lo} OFFSET {lo}
                    )
                )
            ) s ON s.akey = k.akey AND s.tkey = k.tkey
            WHERE k.length_ms IS NOT NULL AND s.duration_ms IS NOT NULL
              AND abs(k.length_ms - s.duration_ms) <= 2500
        ) TO '{SCR}/item_match_{lo}.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
        run(sql, f"item_match chunk {lo}", heavy=True)
        done = con.execute(
            f"SELECT count(*) FROM read_parquet('{SCR}/item_match_{lo}.parquet')"
        ).fetchone()[0]
        print(
            f"    chunk {lo:>9,}-{hi:>9,}: {done:,} candidate rows ({time.time()-t0:,.0f}s elapsed)"
        )
        lo = hi
    # union the chunks into one file
    run(
        f"COPY (SELECT * FROM read_parquet('{SCR}/item_match_*.parquet')) "
        f"TO '{IM}' (FORMAT PARQUET, COMPRESSION 'zstd')",
        "item_match union",
    )
    import glob as _g

    for f in _g.glob(f"{SCR}/item_match_*.parquet"):
        if f != IM:
            os.remove(f)
n_rows, n_items_m = con.execute(
    f"SELECT count(*), count(DISTINCT item_id) FROM read_parquet('{IM}')"
).fetchone()
print(f"  duration-verified candidates: {n_rows:,} rows over {n_items_m:,} items")

# 4. pass 2a: best verified per item
# batched pick: the window over 8.3B rows can't sort-materialize even into
# 150 GiB of spill. item_match is item_id-ordered by construction (chunks
# were cut on ORDER BY item_id ranges), so partition by item_id BETWEEN:
# each batch's window is small, and each item lives in exactly one batch, so
# the winners concatenate directly with no dedup across batches.
IMF = f"{SCR}/item_match.parquet"
PD = f"{SCR}/pick_dur.parquet"
if os.path.exists(PD):
    print("[skip] pick_dur")
else:
    lo_item, hi_item = con.execute(
        f"SELECT min(item_id), max(item_id) FROM read_parquet('{IMF}')"
    ).fetchone()
    BATCH = 250_000
    t0 = time.time()
    first = True
    lo = lo_item
    while lo <= hi_item:
        hi = min(lo + BATCH - 1, hi_item)
        sql = f"""
        COPY (
            SELECT item_id, sp_track, duration_diff_ms, popularity, TRUE AS verified
            FROM read_parquet('{IMF}')
            WHERE item_id BETWEEN {lo} AND {hi}
            QUALIFY row_number() OVER (PARTITION BY item_id
                ORDER BY duration_diff_ms ASC, popularity DESC NULLS LAST) = 1
        ) TO '{SCR}/pick_dur_{lo}.parquet' (FORMAT PARQUET, COMPRESSION 'zstd')
        """
        run(sql, f"pick_dur batch {lo}", heavy=True)
        n = con.execute(
            f"SELECT count(*) FROM read_parquet('{SCR}/pick_dur_{lo}.parquet')"
        ).fetchone()[0]
        print(
            f"    items {lo:>9,}-{hi:>9,}: {n:,} winners ({time.time()-t0:,.0f}s elapsed)"
        )
        lo = hi + 1
    import glob as _g

    parts = sorted(_g.glob(f"{SCR}/pick_dur_*.parquet"))
    run(
        f"COPY (SELECT * FROM read_parquet('{SCR}/pick_dur_*.parquet')) "
        f"TO '{PD}' (FORMAT PARQUET, COMPRESSION 'zstd')",
        "pick_dur union",
    )
    for f in parts:
        os.remove(f)
con.execute(
    f"CREATE OR REPLACE TEMP TABLE pick_dur AS " f"SELECT * FROM read_parquet('{PD}')"
)
n_dur = con.execute("SELECT count(*) FROM pick_dur").fetchone()[0]
print(f"  items with duration-verified match: {n_dur:,}")

# 5. pass 2b: name-only fallback (best candidate per key, anti-join)
con.execute(f"""
CREATE OR REPLACE TEMP TABLE pick_name AS
WITH key_best AS (
    SELECT akey, tkey, sp_track, popularity
    FROM (SELECT akey, tkey, sp_track, popularity FROM read_parquet('{S2}'))
    QUALIFY row_number() OVER (PARTITION BY akey, tkey
        ORDER BY popularity DESC NULLS LAST) = 1
)
SELECT k.item_id, s.sp_track, CAST(NULL AS BIGINT) AS duration_diff_ms,
       s.popularity, FALSE AS verified
FROM read_parquet('{MK}') k
JOIN key_best s ON s.akey = k.akey AND s.tkey = k.tkey
WHERE NOT EXISTS (SELECT 1 FROM pick_dur WHERE pick_dur.item_id = k.item_id)
""")
n_nm = con.execute("SELECT count(*) FROM pick_name").fetchone()[0]
print(f"  name-only fallback matches: {n_nm:,}")

# 6. output: separate table, provenance columns intact
con.execute(f"""
COPY (
    SELECT item_id, sp_track, 'fuzzy' AS match_method,
           CASE WHEN verified THEN 'name+duration' ELSE 'name_only' END AS basis,
           duration_diff_ms, popularity AS sp_popularity
    FROM (SELECT item_id, sp_track, duration_diff_ms, popularity, verified FROM pick_dur
          UNION ALL
          SELECT item_id, sp_track, duration_diff_ms, popularity, verified FROM pick_name)
) TO '{FUZ}' (FORMAT PARQUET, COMPRESSION 'zstd')
""")

# 7. always-run diagnostics
con.sql(f"""SELECT basis, count(*) AS n,
       round(avg(duration_diff_ms),0) AS avg_diff_ms
FROM read_parquet('{FUZ}') GROUP BY 1""").show()

con.sql(f"""WITH mapped AS (
    SELECT item_id FROM read_parquet('{FEAT}/item_spotify_map.parquet')
    UNION
    SELECT item_id FROM read_parquet('{FUZ}')
)
SELECT count(*) FILTER (WHERE m.item_id IS NOT NULL) AS items_any,
       round(100.0 * count(*) FILTER (WHERE m.item_id IS NOT NULL) / count(*), 2) AS distinct_pct,
       round(100.0 * sum(l.listens) FILTER (WHERE m.item_id IS NOT NULL) / sum(l.listens), 2) AS listen_pct
FROM read_parquet('{FEAT}/item_ids.parquet') i
JOIN read_parquet('{FEAT}/item_listens.parquet') l
     ON l.canonical_recording_mbid = i.canonical_recording_mbid
LEFT JOIN mapped m ON m.item_id = i.item_id
""").show()  # baseline to beat: 15.37 / 68.65

# obscurity gradient: listens per item by tier (the de-biasing measurement)
con.sql(f"""WITH tier AS (
    SELECT item_id, 'map' AS t FROM read_parquet('{FEAT}/item_spotify_map.parquet')
    UNION ALL
    SELECT item_id, CASE WHEN basis = 'name+duration' THEN 'fuzzy_dur' ELSE 'fuzzy_name' END
    FROM read_parquet('{FUZ}')
)
SELECT t.t AS tier, count(*) AS items,
       round(avg(l.listens),1) AS avg_listens_per_item,
       quantile_cont(l.listens, 0.5) AS median_listens
FROM tier t
JOIN read_parquet('{FEAT}/item_ids.parquet') i ON i.item_id = t.item_id
JOIN read_parquet('{FEAT}/item_listens.parquet') l
     ON l.canonical_recording_mbid = i.canonical_recording_mbid
GROUP BY 1 ORDER BY 3 DESC
""").show()

print(f"\nOutput: {FUZ}")
print("Disjoint from item_spotify_map by construction; join, never merge rows.")
