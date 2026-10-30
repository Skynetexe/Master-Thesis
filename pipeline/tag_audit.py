"""Tag vocabulary audit: junk scan, variant clusters, filter footprint.

Usage:
    python -m pipeline.tag_audit junk
    python -m pipeline.tag_audit variants
    python -m pipeline.tag_audit footprint
"""

import json
import contextlib
import io
import re
import sys

import duckdb

from .config import CURATED, FEATURES_DIR, MODEL_DIR
from .common import atomic_text_write

JUNK_WORDS = {
    "seen",
    "favorites",
    "favorite",
    "favourites",
    "favourite",
    "faves",
    "fixme",
    "label",
    "wishlist",
    "wish",
    "owned",
    "own",
    "todo",
    "discover",
    "discovered",
    "recommended",
    "recommendation",
    "recommendations",
    "loved",
    "liked",
    "disliked",
    "hated",
    "collection",
    "collections",
    "want",
    "wanted",
    "check",
    "buy",
    "bought",
    "listen",
    "listening",
    "albums",
    "my",
    "mine",
}

VOCAB = str(MODEL_DIR / "vocab_tags.parquet")
COVARIATES = str(MODEL_DIR / "item_covariates.parquet")
DIRECT = str(FEATURES_DIR / "item_tags.parquet")
ARTIST = str(FEATURES_DIR / "item_tags_artist.parquet")


def _connect():
    con = duckdb.connect()
    con.execute("SET threads=4")
    return con


def _filter_tags():
    path = CURATED / "tag_filter.txt"
    tags = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            tags.append(line)
    return tags


def _load_filter_table(con):
    tags = _filter_tags()
    con.execute("CREATE OR REPLACE TABLE filt (tag_name VARCHAR)")
    if tags:
        con.executemany("INSERT INTO filt VALUES (?)", [(t,) for t in tags])
    return tags


def junk(con):
    rows = con.execute(f"""
        SELECT tag_name, df_union FROM read_parquet('{VOCAB}')
        ORDER BY df_union DESC
    """).fetchall()
    hits = []
    for name, df in rows:
        tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
        if tokens & JUNK_WORDS:
            hits.append((df, name))
    print(f"junk candidates: {len(hits)} of {len(rows)} vocabulary tags")
    for df, name in hits:
        print(f"{df:8d}  {name}")
    print("")
    print("This is a review list, not a verdict. Expect few hits:")
    print("MusicBrainz tags come from editors, not from scrobblers.")
    print("The Last.fm personal-tag plague never entered this vocabulary.")


def variants(con):
    rows = con.execute(f"""
        SELECT count(DISTINCT tag_name) AS forms,
               string_agg(tag_name, ' | ' ORDER BY tag_name) AS spellings,
               sum(df_union) AS df
        FROM read_parquet('{VOCAB}')
        GROUP BY regexp_replace(
                 regexp_replace(lower(tag_name), '&', ' and ', 'g'),
                 '[^a-z0-9]', '', 'g')
        HAVING count(DISTINCT tag_name) > 1
        ORDER BY sum(df_union) DESC
        LIMIT 60
    """).fetchall()
    print(f"variant clusters: {len(rows)} shown, ranked by coverage")
    for forms, spellings, df in rows:
        print(f"{df:8d}  ({forms} forms)  {spellings}")
    print("")
    print("Folding rule: lowercase, ampersand to 'and', punctuation gone.")
    print("An alias map could merge each cluster into one column.")


def footprint(con):
    tags = _load_filter_table(con)
    params = json.loads((MODEL_DIR / "params.json").read_text())
    floor = int(params.get("tag_min_df", 50))
    row = con.execute(f"""
        WITH d AS (SELECT DISTINCT item_id, tag_name
                   FROM read_parquet('{DIRECT}')),
             a AS (SELECT DISTINCT item_id, tag_name
                   FROM read_parquet('{ARTIST}')),
             un AS (SELECT item_id, tag_name FROM d
                    UNION SELECT item_id, tag_name FROM a),
             prefilter AS (
                 SELECT tag_name FROM un GROUP BY tag_name
                 HAVING count(DISTINCT item_id) >= {floor}),
             fd AS (SELECT item_id FROM d
                    WHERE tag_name IN (SELECT tag_name FROM filt)),
             fa AS (SELECT item_id FROM a
                    WHERE tag_name IN (SELECT tag_name FROM filt)),
             touched AS (SELECT item_id FROM fd
                         UNION SELECT item_id FROM fa),
             anytag AS (SELECT DISTINCT item_id FROM un),
             kept AS (SELECT DISTINCT item_id FROM un
                      WHERE tag_name NOT IN (SELECT tag_name FROM filt)),
             covt AS (SELECT item_id, train_listens
                      FROM read_parquet('{COVARIATES}'))
        SELECT (SELECT count(*) FROM filt
                WHERE tag_name IN (SELECT tag_name FROM prefilter)),
               (SELECT count(*) FROM prefilter),
               (SELECT count(*) FROM d) + (SELECT count(*) FROM a),
               (SELECT count(*) FROM fd) + (SELECT count(*) FROM fa),
               (SELECT count(*) FROM anytag),
               (SELECT count(*) FROM touched),
               (SELECT count(*) FROM anytag
                WHERE item_id NOT IN (SELECT item_id FROM kept)),
               (SELECT coalesce(sum(train_listens), 0) FROM covt),
               (SELECT coalesce(sum(train_listens), 0) FROM covt
                WHERE item_id IN (SELECT item_id FROM touched)),
               (SELECT coalesce(sum(train_listens), 0) FROM covt
                WHERE item_id IN (SELECT item_id FROM anytag)
                  AND item_id NOT IN (SELECT item_id FROM kept))
    """).fetchone()
    (
        n_filtered,
        n_prefilter,
        pairs_total,
        pairs_filtered,
        items_any,
        items_touched,
        items_orphaned,
        listens_total,
        listens_touched,
        listens_orphaned,
    ) = row
    print(f"filter file: {len(tags)} tags, {n_filtered} of them in the " f"vocabulary")
    print(
        f"vocabulary share: {n_filtered} of {n_prefilter} tags "
        f"({100.0 * n_filtered / n_prefilter:.2f} percent)"
    )
    print(
        f"tag pair share: {pairs_filtered} of {pairs_total} pairs "
        f"({100.0 * pairs_filtered / pairs_total:.2f} percent)"
    )
    print(
        f"items touched: {items_touched} of {items_any} "
        f"({100.0 * items_touched / items_any:.2f} percent) "
        "- these items keep their other tags"
    )
    print(
        f"items that lose every tag: {items_orphaned} of {items_any} "
        f"({100.0 * items_orphaned / items_any:.3f} percent)"
    )
    print(
        f"train listens on touched items: "
        f"{100.0 * listens_touched / listens_total:.2f} percent"
    )
    print(
        f"train listens on items with no tag left: "
        f"{100.0 * listens_orphaned / listens_total:.3f} percent"
    )


MODES = {"junk": junk, "variants": variants, "footprint": footprint}


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in MODES:
        print("usage: python -m pipeline.tag_audit " "<junk|variants|footprint>")
        return 2
    con = _connect()
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            MODES[sys.argv[1]](con)
        atomic_text_write(
            buf.getvalue(), MODEL_DIR / "reports" / ("tag_audit_" + sys.argv[1] + ".md")
        )
        print(buf.getvalue(), end="")
    finally:
        con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
