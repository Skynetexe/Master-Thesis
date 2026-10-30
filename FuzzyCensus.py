#!/usr/bin/env python3
"""The census of the fuzzy matcher of FuzzyBridge.py, and the arithmetic
of the two earlier census populations.

Part 1: run the dual-verified census (99,559 isrc+url items) against the
matcher of FuzzyBridge.py, with its empty-key guard and the hyphen cut in
NORM. It turns the expected purity grades into a measurement, and it
gives the recall of the candidate pool at item grain.

Part 2: the earlier census rates came from two populations with two
denominators (the raw pool and the top-5 pool). The arithmetic shows that
neither set of figures reproduces on the other denominator, so each set
carries the label of its own population.

Expectations, declared before the run:
  - census keys: 99,559 minus empty-key items (the guard excludes them),
    expect about 90-97K
  - verified-tier certified: >= 90.1% (the earlier census, of the matcher
    without the guard and the hyphen cut), small or no shift
  - fallback-tier certified: about 16-19%
  - end-to-end recall (vs keyed_items): higher than the earlier 70.0%,
    because the guard removed unmatchable items from the denominator
"""

import os

import duckdb

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THESIS_DATA_ROOT", os.path.join(CODE_DIR, "data"))
FEAT = f"{ROOT}/parquet/features"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
KAG = f"{ROOT}/kaggle/spotify-catalogue"
TRK = f"{KAG}/spotify_clean_parquet/tracks.parquet"
TA = f"{KAG}/spotify_clean_parquet/track_artists.parquet"
AR = f"{KAG}/spotify_clean_parquet/artists.parquet"

con = duckdb.connect()
con.execute("SET threads=14")
con.execute("SET memory_limit='24GB'")
con.execute(f"SET temp_directory='{ROOT}/duckdb_tmp'")
con.execute("SET preserve_insertion_order=false")
con.execute("SET max_temp_directory_size='150GiB'")

NORM = """regexp_replace(regexp_replace(regexp_replace(regexp_replace(
         lower(%s),
         '\\s+-\\s+.*$', '', 'g'),
         '\\([^)]*\\)|\\[[^]]*\\]', '', 'g'),
         '(feat\\.?|featuring|ft\\.)\\s*', '', 'g'),
         '[^a-z0-9]+', '', 'g')"""

# Part 1: census
print("Part 1: census purity\n")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE census AS
SELECT h.item_id, h.sp_track AS true_track, c.artist_name, c.recording_name,
       L.length_ms
FROM (SELECT item_id, sp_track FROM read_parquet('{FEAT}/item_spotify_map.parquet')
      WHERE match_method = 'isrc+url') h
JOIN read_parquet('{FEAT}/item_ids.parquet') i ON i.item_id = h.item_id
JOIN (
    SELECT canonical_recording_mbid, artist_name, recording_name
    FROM read_parquet('{FEAT}/item_catalog.parquet')
    QUALIFY row_number() OVER (PARTITION BY canonical_recording_mbid
                               ORDER BY recording_name IS NOT NULL DESC,
                                        artist_name IS NOT NULL DESC, listens DESC) = 1
) c ON c.canonical_recording_mbid = i.canonical_recording_mbid
LEFT JOIN (
    SELECT DISTINCT ON (item_id) item_id, CAST(rec.length AS BIGINT) AS length_ms
    FROM (SELECT m.recording_mbid, it2.item_id
          FROM read_parquet('{FEAT}/recording_canonical_map.parquet') m
          JOIN read_parquet('{FEAT}/item_ids.parquet') it2
               ON it2.canonical_recording_mbid = m.canonical_recording_mbid) x
    JOIN read_parquet('{MB}/recording.parquet') rec ON rec.gid = x.recording_mbid
    ORDER BY item_id, recording_mbid
) L ON L.item_id = h.item_id
WHERE c.artist_name IS NOT NULL AND c.recording_name IS NOT NULL
""")
print(f"census: {con.execute('SELECT count(*) FROM census').fetchone()[0]:,} items")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE census_keys AS
SELECT item_id, true_track, length_ms,
       {NORM % 'artist_name'} AS akey, {NORM % 'recording_name'} AS tkey
FROM census
WHERE {NORM % 'artist_name'} <> '' AND {NORM % 'recording_name'} <> ''
""")
n_keys = con.execute("SELECT count(*) FROM census_keys").fetchone()[0]
print(
    f"census keys (guarded, non-empty): {n_keys:,} "
    f"({99559 - n_keys:,} empty-key census items excluded by the guard)"
)

# candidate pool: aggregate inside the subquery (ar is in scope there),
# then normalize the aggregate outside
con.execute(f"""
CREATE OR REPLACE TEMP TABLE cands AS
SELECT sp_track, duration_ms, popularity, tkey,
       {NORM % "artist_concat"} AS akey
FROM (
    SELECT s.id AS sp_track, CAST(s.duration_ms AS BIGINT) AS duration_ms,
           CAST(s.popularity AS INTEGER) AS popularity,
           {NORM % 's.name'} AS tkey, string_agg(ar.name, ' ' ORDER BY ar.name) AS artist_concat
    FROM read_parquet('{TRK}') s
    JOIN read_parquet('{TA}') ta ON ta.track_rowid = s.rowid
    JOIN read_parquet('{AR}') ar ON ar.rowid = ta.artist_rowid
    WHERE {NORM % 's.name'} <> ''
      AND {NORM % 's.name'} IN (SELECT tkey FROM census_keys)
    GROUP BY s.id, s.duration_ms, s.popularity, tkey
) sub
""")
print(
    f"candidate pool (guarded): {con.execute('SELECT count(*) FROM cands').fetchone()[0]:,} tracks"
)

con.execute("""
CREATE OR REPLACE TEMP TABLE cands_top AS
SELECT sp_track, duration_ms, popularity, tkey, akey
FROM cands
QUALIFY row_number() OVER (
    PARTITION BY akey, tkey
    ORDER BY (duration_ms IS NOT NULL) DESC, popularity DESC NULLS LAST
) <= 5
""")

# the census matcher (top-5 pool): the closest duration wins, as in
# production, and popularity breaks ties
con.execute(f"""
CREATE OR REPLACE TEMP TABLE picks AS
SELECT k.item_id, s.sp_track AS fuzzy_track,
       CASE WHEN k.length_ms IS NOT NULL AND s.duration_ms IS NOT NULL
            AND abs(k.length_ms - s.duration_ms) <= 2500
            THEN TRUE ELSE FALSE END AS dur_ok
FROM census_keys k
JOIN cands_top s ON s.akey = k.akey AND s.tkey = k.tkey
QUALIFY row_number() OVER (PARTITION BY k.item_id
    ORDER BY abs(CASE WHEN k.length_ms IS NOT NULL AND s.duration_ms IS NOT NULL
                      THEN k.length_ms - s.duration_ms END) ASC NULLS LAST,
             s.popularity DESC NULLS LAST) = 1
""")

n_picks, n_dur = con.execute(
    "SELECT count(*), count(*) FILTER (WHERE dur_ok) FROM picks"
).fetchone()
print(f"picks: {n_picks:,} attempted, {n_dur:,} duration-verified\n")

print("Tier purity")
con.sql(f"""
WITH iso AS (
    SELECT p.dur_ok,
           (p.fuzzy_track = c.true_track) AS exact,
           upper(regexp_replace(fi.external_id_isrc, '[^A-Za-z0-9]', '', 'g'))
         = upper(regexp_replace(ti.external_id_isrc, '[^A-Za-z0-9]', '', 'g')) AS same_isrc
    FROM picks p
    JOIN census c ON c.item_id = p.item_id
    JOIN (SELECT id, external_id_isrc FROM read_parquet('{TRK}')
          WHERE id IN (SELECT fuzzy_track FROM picks)) fi ON fi.id = p.fuzzy_track
    JOIN (SELECT id, external_id_isrc FROM read_parquet('{TRK}')
          WHERE id IN (SELECT true_track FROM census_keys)) ti ON ti.id = c.true_track
    WHERE fi.external_id_isrc IS NOT NULL AND ti.external_id_isrc IS NOT NULL
)
SELECT dur_ok AS duration_verified_pick,
       count(*) AS checkable,
       round(100.0 * count(*) FILTER (WHERE exact) / count(*), 1) AS pct_exact,
       round(100.0 * count(*) FILTER (WHERE same_isrc) / count(*), 1) AS pct_same_isrc,
       round(100.0 * count(*) FILTER (WHERE exact OR same_isrc) / count(*), 1) AS pct_certified,
       count(*) FILTER (WHERE exact) AS n_exact
FROM iso GROUP BY 1
""").show()

print("Pool recall (item grain, guarded)")
con.sql(f"""
WITH per_item AS (
    SELECT k.item_id, k.true_track,
           bool_or(s.sp_track = k.true_track) AS in_pool
    FROM census_keys k
    JOIN cands_top s ON s.akey = k.akey AND s.tkey = k.tkey
    GROUP BY k.item_id, k.true_track
)
SELECT (SELECT count(*) FROM census_keys) AS keyed_items,
       count(*) AS with_any_candidate,
       count(*) FILTER (WHERE in_pool) AS in_pool,
       round(100.0 * count(*) FILTER (WHERE in_pool)
             / (SELECT count(*) FROM census_keys), 1) AS pct_recall_vs_keyed,
       round(100.0 * count(*) FILTER (WHERE in_pool) / 99559, 1) AS pct_recall_vs_census
FROM per_item
""").show()

# Part 2: the arithmetic of the two populations
print("\nPart 2: the two census populations\n")
D, N = 14511, 2694
r_exact_d, r_exact_n = 0.677, 0.147
r_cert_d, r_cert_n = 0.881, 0.179
exact_d = round(r_exact_d * D)
exact_n = round(r_exact_n * N)
cert_d = round(r_cert_d * D)
cert_n = round(r_cert_n * N)
print(
    f"top-5 population (recorded in the earlier census): attempted 17,206 "
    f"(14,511 verified + 2,694 fallback + 1 tie-order row)"
)
print(f"  verified tier: exact {exact_d:,}, certified {cert_d:,} (of {D:,})")
print(f"  fallback tier: exact {exact_n:,}, certified {cert_n:,} (of {N:,})")
print(
    f"  totals: exact {exact_d + exact_n:,} "
    f"({100.0 * (exact_d + exact_n) / (D + N):.1f}%), "
    f"certified {cert_d + cert_n:,} "
    f"({100.0 * (cert_d + cert_n) / (D + N):.1f}%)"
)
print(f"""
raw-pool population (recorded in the first census): attempted 15,956
  exact 8,550 = 53.6%; certified 12,524 = 78.5%

Conclusion (demonstrated, not asserted): the tier rates applied to the
top-5 denominators produce overall rates (~{100.0 * (exact_d + exact_n) / (D + N):.1f}% exact) that
do not reproduce the first census figures against 15,956, and those
figures do not reproduce against 17,206 either (8,550/17,206 = 49.7%).
Two different populations, two different denominators: each set of
figures carries the label of its own population, and each figure is
internally consistent with its own denominator.
""")
