#!/usr/bin/env python3
"""Rebuild the entire feature layer and the interaction matrix from the
converted parquet. Idempotent: every step checks for its output
first; delete an output file to rebuild just that step. Prints identity
checks; investigate any line that doesn't match its expectation.

It brings together queries that were each run and checked on their own. It
has not run end to end as one script. With its outputs present, it only runs
the identity checks (a dry run)."""

import duckdb, glob, os, shutil, time


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("THESIS_DATA_ROOT", os.path.join(CODE_DIR, "data"))
MLHD, FEAT, INT = (
    f"{ROOT}/parquet/mlhd",
    f"{ROOT}/parquet/features",
    f"{ROOT}/parquet/interactions",
)
CANON = f"{ROOT}/parquet/canonical"
MB = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump"
MBD = f"{ROOT}/parquet/musicbrainz/fullexport/mbdump-derived"
TMP = f"{ROOT}/duckdb_tmp"
for d in (FEAT, INT, TMP):
    os.makedirs(d, exist_ok=True)

con = duckdb.connect()
con.execute("SET threads=4")
con.execute("SET memory_limit='24GB'")
con.execute(f"SET temp_directory='{TMP}'")
con.execute("SET preserve_insertion_order=false")


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


def stat(path, label, expect=None):
    """Count rows; also sum 'listens' only if the table actually has that
    column (the canonical map has none, and a sum over a missing column
    fails)."""
    cols = {
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{path}')"
        ).fetchall()
    }
    n = con.execute(f"SELECT count(*) FROM read_parquet('{path}')").fetchone()[0]
    extra = ""
    if "listens" in cols:
        s = con.execute(
            f"SELECT coalesce(sum(listens),0) FROM read_parquet('{path}')"
        ).fetchone()[0]
        extra = f", {int(s):,} listens"
    print(f"  {label}: {n:,} rows{extra}" + (f"  (expect {expect})" if expect else ""))


files = sorted(glob.glob(f"{MLHD}/mlhd_complete_*.parquet"))

# 1. popularity (per-file corpus scan)
POP = f"{FEAT}/complete_recording_popularity.parquet"
if not os.path.exists(POP):
    pd_ = f"{INT}/pop_partials"
    os.makedirs(pd_, exist_ok=True)
    for f in files:
        out = f"{pd_}/{os.path.basename(f).replace('.parquet','_agg.parquet')}"
        copy_to(
            out,
            f"SELECT recording_mbid, count(*) AS listens FROM read_parquet('{f}') "
            f"WHERE recording_mbid IS NOT NULL GROUP BY 1",
            f"agg {os.path.basename(f)}",
        )
    con.execute(f"SET threads=4")
    copy_to(
        POP,
        f"SELECT recording_mbid, sum(listens) AS listens FROM read_parquet('{pd_}/*_agg.parquet') GROUP BY 1",
        "popularity",
    )
    shutil.rmtree(pd_)
stat(POP, "popularity")

# 2. canonical map (three-tier chain)
MAP = f"{FEAT}/recording_canonical_map.parquet"
if not os.path.exists(MAP):
    UNR = f"{INT}/unresolved_recordings.parquet"
    copy_to(
        UNR,
        f"SELECT d.recording_mbid, d.listens FROM read_parquet('{POP}') d "
        f"LEFT JOIN read_parquet('{CANON}/canonical_recording_redirect.parquet') r ON r.recording_mbid = d.recording_mbid "
        f"LEFT JOIN read_parquet('{CANON}/canonical_musicbrainz_data.parquet') c ON c.recording_mbid = d.recording_mbid "
        f"WHERE r.canonical_recording_mbid IS NULL AND c.recording_mbid IS NULL",
        "unresolved",
    )
    gr = {
        r[0]
        for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{MB}/recording_gid_redirect.parquet')"
        ).fetchall()
    }
    G, N = ("gid", "new_id") if {"gid", "new_id"} <= gr else ("c0", "c1")
    con.execute(
        f"""CREATE OR REPLACE TEMP TABLE chain AS
        SELECT u.recording_mbid AS old_mbid,
               COALESCE(cd.recording_mbid, cr.canonical_recording_mbid) AS canon
        FROM read_parquet('{UNR}') u
        LEFT JOIN read_parquet('{MB}/recording_gid_redirect.parquet') gr2 ON gr2.{G} = u.recording_mbid
        LEFT JOIN read_parquet('{MB}/recording.parquet') rec ON rec.id = gr2.{N}
        LEFT JOIN read_parquet('{CANON}/canonical_musicbrainz_data.parquet') cd ON cd.recording_mbid = rec.gid
        LEFT JOIN read_parquet('{CANON}/canonical_recording_redirect.parquet') cr ON cr.recording_mbid = rec.gid"""
    )
    # fan-out guard: the resolver must not multiply rows
    n_unres = con.execute(f"SELECT count(*) FROM read_parquet('{UNR}')").fetchone()[0]
    n_chain = con.execute("SELECT count(*) FROM chain").fetchone()[0]
    check(
        n_chain == n_unres,
        f"resolver fan-out: chain {n_chain:,} vs unresolved {n_unres:,}",
    )
    copy_to(
        f"{FEAT}/gid_redirect_resolution.parquet",
        "SELECT old_mbid, canon AS canonical_recording_mbid FROM chain WHERE canon IS NOT NULL",
        "gid resolution",
    )
    copy_to(
        MAP,
        f"SELECT d.recording_mbid, COALESCE(r.canonical_recording_mbid, c.recording_mbid, g.canonical_recording_mbid) "
        f"AS canonical_recording_mbid FROM read_parquet('{POP}') d "
        f"LEFT JOIN read_parquet('{CANON}/canonical_recording_redirect.parquet') r ON r.recording_mbid = d.recording_mbid "
        f"LEFT JOIN read_parquet('{CANON}/canonical_musicbrainz_data.parquet') c ON c.recording_mbid = d.recording_mbid "
        f"LEFT JOIN read_parquet('{FEAT}/gid_redirect_resolution.parquet') g ON g.old_mbid = d.recording_mbid",
        "canonical map",
    )
stat(MAP, "map (rows = popularity rows; resolved ~99.9%)")

# 3. id maps
copy_to(
    f"{FEAT}/item_ids.parquet",
    f"SELECT row_number() OVER (ORDER BY canonical_recording_mbid) AS item_id, canonical_recording_mbid "
    f"FROM (SELECT DISTINCT canonical_recording_mbid FROM read_parquet('{MAP}') WHERE canonical_recording_mbid IS NOT NULL)",
    "item_ids",
)
UID = f"{FEAT}/user_ids.parquet"
if not os.path.exists(UID):
    up = f"{INT}/user_partials"
    os.makedirs(up, exist_ok=True)
    for f in files:
        copy_to(
            f"{up}/{os.path.basename(f)}",
            f"SELECT DISTINCT user_mbid FROM read_parquet('{f}')",
            f"users {os.path.basename(f)}",
        )
    con.execute("SET threads=4")
    copy_to(
        UID,
        f"SELECT row_number() OVER (ORDER BY user_mbid) AS user_id, user_mbid "
        f"FROM (SELECT DISTINCT user_mbid FROM read_parquet('{up}/*.parquet'))",
        "user_ids",
    )
    shutil.rmtree(up)

# 4. catalog (keyed by raw recording_mbid by design: dedupe it before a join
#    to item-level tables)
copy_to(
    f"{FEAT}/item_catalog.parquet",
    f"""
    SELECT m.recording_mbid, m.canonical_recording_mbid, d.listens,
           COALESCE(cd.artist_credit_name, acn.name) AS artist_name,
           cd.artist_mbids, COALESCE(cd.recording_name, rec.name) AS recording_name,
           cd.release_mbid, cd.release_name
    FROM read_parquet('{MAP}') m
    JOIN read_parquet('{POP}') d ON d.recording_mbid = m.recording_mbid
    LEFT JOIN read_parquet('{CANON}/canonical_musicbrainz_data.parquet') cd ON cd.recording_mbid = m.canonical_recording_mbid
    LEFT JOIN read_parquet('{MB}/recording.parquet') rec ON rec.gid = m.recording_mbid
    LEFT JOIN read_parquet('{MB}/artist_credit_name.parquet') acn ON acn.artist_credit = rec.artist_credit AND acn."position" = 0""",
    "item_catalog",
)

# 5. item_listens (canonical-level popularity)
copy_to(
    f"{FEAT}/item_listens.parquet",
    f"""
    SELECT m.canonical_recording_mbid, sum(p.listens) AS listens
    FROM read_parquet('{POP}') p JOIN read_parquet('{MAP}') m ON m.recording_mbid = p.recording_mbid
    WHERE m.canonical_recording_mbid IS NOT NULL GROUP BY 1""",
    "item_listens",
)
stat(
    f"{FEAT}/item_listens.parquet",
    "item_listens (items = item_ids rows; listens = matrix total)",
)

# 6-9. metadata tables
copy_to(
    f"{FEAT}/item_artists.parquet",
    f"""
    SELECT it.item_id, a.id AS artist_id, a.gid AS artist_mbid,
           min(acn.name) AS artist_name, CAST(min(acn."position") AS INTEGER) AS position
    FROM read_parquet('{FEAT}/item_ids.parquet') it
    JOIN read_parquet('{MB}/recording.parquet') rec ON rec.gid = it.canonical_recording_mbid
    JOIN read_parquet('{MB}/artist_credit_name.parquet') acn ON acn.artist_credit = rec.artist_credit
    JOIN read_parquet('{MB}/artist.parquet') a ON a.id = acn.artist GROUP BY 1, 2, 3""",
    "item_artists",
)
copy_to(
    f"{FEAT}/item_tags.parquet",
    f"""
    SELECT it.item_id, t.name AS tag_name, sum(CAST(rt."count" AS BIGINT)) AS tag_votes
    FROM read_parquet('{MBD}/recording_tag.parquet') rt
    JOIN read_parquet('{MB}/recording.parquet') rec ON rec.id = rt.recording
    JOIN read_parquet('{MAP}') m ON m.recording_mbid = rec.gid
    JOIN read_parquet('{FEAT}/item_ids.parquet') it ON it.canonical_recording_mbid = m.canonical_recording_mbid
    JOIN read_parquet('{MBD}/tag.parquet') t ON t.id = rt.tag GROUP BY 1, 2""",
    "item_tags",
)
copy_to(
    f"{FEAT}/item_tags_artist.parquet",
    f"""
    SELECT ia.item_id, t.name AS tag_name, sum(CAST(atg."count" AS BIGINT)) AS tag_votes
    FROM read_parquet('{FEAT}/item_artists.parquet') ia
    JOIN read_parquet('{MBD}/artist_tag.parquet') atg ON atg.artist = ia.artist_id
    JOIN read_parquet('{MBD}/tag.parquet') t ON t.id = atg.tag GROUP BY 1, 2""",
    "item_tags_artist",
)
copy_to(
    f"{FEAT}/item_works.parquet",
    f"""
    SELECT it.item_id, w.id AS work_id, w.name AS work_name, lt.name AS relation,
           string_agg(DISTINCT lat.name, ', ' ORDER BY lat.name) AS attributes
    FROM read_parquet('{MB}/l_recording_work.parquet') lrw
    JOIN read_parquet('{MB}/link.parquet') lnk ON lnk.id = lrw.link
    JOIN read_parquet('{MB}/link_type.parquet') lt ON lt.id = lnk.link_type
    JOIN read_parquet('{MB}/recording.parquet') rec ON rec.id = lrw.entity0
    JOIN read_parquet('{MB}/work.parquet') w ON w.id = lrw.entity1
    JOIN read_parquet('{FEAT}/item_ids.parquet') it ON it.canonical_recording_mbid = rec.gid
    LEFT JOIN read_parquet('{MB}/link_attribute.parquet') la ON la.link = lrw.link
    LEFT JOIN read_parquet('{MB}/link_attribute_type.parquet') lat ON lat.id = la.attribute_type
    GROUP BY 1, 2, 3, 4""",
    "item_works",
)

# 10. interaction matrix (users are bucket-local, so the 16 files are the matrix)
for f in files:
    sfx = os.path.basename(f).replace("mlhd_complete_", "").replace(".parquet", "")
    copy_to(
        f"{INT}/interactions_{sfx}.parquet",
        f"""
        SELECT u.user_id, it.item_id, count(*) AS listens,
               min(l.listened_at_unix) AS first_ts, max(l.listened_at_unix) AS last_ts
        FROM read_parquet('{f}') l
        JOIN read_parquet('{UID}') u ON u.user_mbid = l.user_mbid
        JOIN read_parquet('{MAP}') m ON m.recording_mbid = l.recording_mbid
        JOIN read_parquet('{FEAT}/item_ids.parquet') it ON it.canonical_recording_mbid = m.canonical_recording_mbid
        GROUP BY 1, 2""",
        f"interactions {sfx}",
    )

print("\nFinal identity checks")
p, s = con.execute(
    f"SELECT count(*), sum(listens) FROM read_parquet('{INT}/interactions_*.parquet')"
).fetchone()
print(
    f"matrix: {p:,} pairs, {int(s):,} listens (must equal the item_listens total exactly)"
)
i, il = con.execute(
    f"SELECT (SELECT count(*) FROM read_parquet('{FEAT}/item_ids.parquet')), "
    f"(SELECT count(*) FROM read_parquet('{FEAT}/item_listens.parquet'))"
).fetchone()
print(f"item_ids: {i:,}  item_listens items: {il:,}  (must match)")
