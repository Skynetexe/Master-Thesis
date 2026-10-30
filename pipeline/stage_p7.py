"""Stage P7: the item-item relational graph.

Outputs:
- item_edges.parquet: item_id_a, item_id_b, edge_type, source_count.
  item_id_a < item_id_b always (one row per unordered pair, never both
  directions). edge_type is 'shared_artist' for artist_id-based edges, or
  the literal relation string from item_works (for example 'cover') for
  work-based edges. source_count is the number of distinct artists or
  works behind the edge. It is almost always 1, but it stays, so that a
  pair with more than one real-world reason is not confused with a pair
  that rests on a single coincidence.
- p7_report.md

Measured against the real fan-out:

Artist-side edges are capped by degree (ARTIST_DEGREE_CAP), because
artist_id is already a real MusicBrainz entity ID, not a string, so the
problem there is only scale. p999 items-per-artist was 515 against a
maximum of 12,256. A cap just under p999 removes a small cluster of
outlier "hub" artists (a classical composer, a shared "Various Artists"
credit) without touching the signal for anyone else.

Work-side edges are a different problem, and a cap alone does not solve
it. item_works.work_name joins two different MusicBrainz Work entities
that share a name ("Intro", a generic track title on unrelated albums)
in the same way as one Work covered many times ("Yesterday", "Silent
Night"), and both cases sit in the same tail of the degree distribution,
so no threshold separates them. The real key is work_id. This stage
checks at run time whether work_id is present in item_works.parquet. If
it is, the stage joins on the entity ID (no collision is possible, so no
stopword list is needed, and the cap is only a second guard). If it is
not, the stage falls back to work_name plus a curated exclusion list of
generic titles (GENERIC_WORK_TITLES_PATH, created with a starter list on
the first run, in the form of stage_p4's tag_filter.txt). That keeps the
"Yesterday" edges and drops the worst collisions.

edge_type for work-based edges is the literal relation string, which is
measured to be 'performance' throughout. The live, cover, instrumental
and medley distinction lives in item_works.attributes, and this stage
does not use it, by measurement: across items that share a work_id,
88,255 groups disagree on their attributes (one item untagged, another
tagged 'cover', on the same work). attributes describes one recording's
relation to a work, not a property of the work, so no single edge_type
can hold both sides. A 'performance' edge says "these two items realize
the same work". To ask whether an item is the live version, join
item_works.parquet by that item's own item_id.

No later stage reads this graph. The final model does not use it.
"""

from pathlib import Path

from .common import (
    atomic_parquet_write,
    check,
    connect,
    features_table,
    file_sha256,
    log,
    marker_path,
    with_thread_ladder,
)
from .config import (
    ARTIST_DEGREE_CAP,
    FROZEN_MODEL_ROWS,
    GENERIC_WORK_TITLES_PATH,
    MODEL_DIR,
    WORK_DEGREE_CAP,
)

DEFAULT_GENERIC_WORK_TITLES = [
    "Intro",
    "Outro",
    "Interlude",
    "Prelude",
    "Overture",
    "Prologue",
    "Epilogue",
    "Reprise",
    "Opening",
    "Closing",
    "Finale",
    "Theme",
    "Main Title",
    "Main Theme",
    "Untitled",
    "Instrumental",
]


def _load_generic_work_titles():
    path = Path(GENERIC_WORK_TITLES_PATH)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "# One generic/placeholder work title per line, case-sensitive\n"
            "# as it appears in work_name. This is a starting list, not a\n"
            "# measured one. Check the work titles with the most items\n"
            "# for titles that belong here and aren't yet.\n"
            + "\n".join(DEFAULT_GENERIC_WORK_TITLES)
            + "\n"
        )
        log.info(
            f"created {path} with a starter list of "
            f"{len(DEFAULT_GENERIC_WORK_TITLES)} generic titles"
        )
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]


class StageP7:
    name = "p7_graph"
    upstream = ["p1_audit"]

    def outputs(self):
        return {
            "edges": MODEL_DIR / "item_edges.parquet",
            "report": MODEL_DIR / "reports" / "p7_report.md",
        }

    def current_inputs(self, files):
        inputs = {}
        for stage in self.upstream:
            try:
                inputs[f"marker:{stage}"] = file_sha256(marker_path(stage))
            except FileNotFoundError:
                inputs[f"marker:{stage}"] = None
        return inputs

    def run(self, params, files):
        outputs = self.outputs()
        artist_cap = params.get("artist_degree_cap", ARTIST_DEGREE_CAP)
        work_cap = params.get("work_degree_cap", WORK_DEGREE_CAP)

        def work(con):
            artists = features_table("item_artists")
            works = features_table("item_works")

            has_work_id = "work_id" in {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{works}') LIMIT 0"
                ).fetchall()
            }

            # Shared-artist edges. artist_id is a real entity ID, so this
            # is a pure scale cap, not a collision guard.
            con.execute(f"""
                CREATE OR REPLACE TEMP TABLE artist_edges AS
                WITH capped AS (
                    SELECT artist_id
                    FROM (SELECT artist_id, count(DISTINCT item_id) AS n
                          FROM read_parquet('{artists}')
                          WHERE artist_id IS NOT NULL
                          GROUP BY artist_id)
                    WHERE n <= {artist_cap}
                )
                SELECT a.item_id AS item_id_a, b.item_id AS item_id_b,
                       'shared_artist' AS edge_type,
                       count(DISTINCT a.artist_id) AS source_count
                FROM read_parquet('{artists}') a
                JOIN read_parquet('{artists}') b
                  ON a.artist_id = b.artist_id AND a.item_id < b.item_id
                JOIN capped c ON c.artist_id = a.artist_id
                GROUP BY 1, 2, 3
            """)

            if has_work_id:
                log.info(
                    "item_works.parquet has work_id: using the "
                    "precise entity-ID join, no stopword list needed"
                )
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE work_edges AS
                    WITH capped AS (
                        SELECT work_id
                        FROM (SELECT work_id, count(DISTINCT item_id) AS n
                              FROM read_parquet('{works}')
                              WHERE work_id IS NOT NULL
                              GROUP BY work_id)
                        WHERE n <= {work_cap}
                    )
                    SELECT a.item_id AS item_id_a, b.item_id AS item_id_b,
                           a.relation AS edge_type,
                           count(DISTINCT a.work_id) AS source_count
                    FROM read_parquet('{works}') a
                    JOIN read_parquet('{works}') b
                      ON a.work_id = b.work_id AND a.item_id < b.item_id
                         AND a.relation = b.relation
                    JOIN capped c ON c.work_id = a.work_id
                    GROUP BY 1, 2, 3
                """)
            else:
                log.info(
                    "item_works.parquet has no work_id yet: falling "
                    "back to work_name plus the generic-title "
                    "exclusion list. Run BuildFeatures.py again "
                    "to remove the need for this branch."
                )
                generic = _load_generic_work_titles()
                con.execute(
                    "CREATE OR REPLACE TEMP TABLE generic_titles "
                    "AS SELECT * FROM (VALUES "
                    + ", ".join(f"('{t}')" for t in generic)
                    + ") AS t(work_name)"
                )
                con.execute(f"""
                    CREATE OR REPLACE TEMP TABLE work_edges AS
                    WITH capped AS (
                        SELECT work_name
                        FROM (SELECT work_name, count(DISTINCT item_id) AS n
                              FROM read_parquet('{works}')
                              WHERE work_name IS NOT NULL
                              GROUP BY work_name)
                        WHERE n <= {work_cap}
                    )
                    SELECT a.item_id AS item_id_a, b.item_id AS item_id_b,
                           a.relation AS edge_type,
                           count(DISTINCT a.work_name) AS source_count
                    FROM read_parquet('{works}') a
                    JOIN read_parquet('{works}') b
                      ON a.work_name = b.work_name AND a.item_id < b.item_id
                         AND a.relation = b.relation
                    JOIN capped c ON c.work_name = a.work_name
                    LEFT JOIN generic_titles g ON g.work_name = a.work_name
                    WHERE g.work_name IS NULL
                    GROUP BY 1, 2, 3
                """)

            atomic_parquet_write(
                con,
                """SELECT * FROM artist_edges
                   UNION ALL
                   SELECT * FROM work_edges
                   ORDER BY item_id_a, item_id_b, edge_type""",
                outputs["edges"],
                expected_mb=200,
            )

        with_thread_ladder(work, params["ladder"])

    def verify(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["edges"])
            item_ids = features_table("item_ids")

            rows, ordered, self_loops = con.execute(f"""
                SELECT count(*),
                       sum(CASE WHEN item_id_a < item_id_b THEN 1 ELSE 0 END),
                       sum(CASE WHEN item_id_a = item_id_b THEN 1 ELSE 0 END)
                FROM read_parquet('{path}')
            """).fetchone()
            check(
                rows == ordered,
                f"{rows - ordered} rows do not satisfy item_id_a < item_id_b",
            )
            check(self_loops == 0, f"{self_loops} self-loop rows")

            n_bad = con.execute(f"""
                SELECT count(*) FROM read_parquet('{path}') e
                LEFT JOIN read_parquet('{item_ids}') ia ON ia.item_id = e.item_id_a
                LEFT JOIN read_parquet('{item_ids}') ib ON ib.item_id = e.item_id_b
                WHERE ia.item_id IS NULL OR ib.item_id IS NULL
            """).fetchone()[0]
            check(
                n_bad == 0,
                f"{n_bad} edges reference an item_id absent from item_ids.parquet",
            )

            n_dup = con.execute(f"""
                SELECT count(*) - count(DISTINCT (item_id_a, item_id_b, edge_type))
                FROM read_parquet('{path}')
            """).fetchone()[0]
            check(
                n_dup == 0, f"{n_dup} duplicate (item_id_a, item_id_b, edge_type) rows"
            )

            expected = FROZEN_MODEL_ROWS["item_edges.parquet"]
            check(
                rows == expected,
                f"item_edges.parquet: measured {rows} rows, frozen expects {expected}",
            )
        finally:
            con.close()

    def report(self, params):
        outputs = self.outputs()
        con = connect(params["ladder"][0])
        try:
            path = str(outputs["edges"])
            total = con.execute(
                f"SELECT count(*) FROM read_parquet('{path}')"
            ).fetchone()[0]
            by_type = con.execute(f"""
                SELECT edge_type, count(*) FROM read_parquet('{path}')
                GROUP BY 1 ORDER BY 2 DESC
            """).fetchall()
            lines = ["# P7 report", "", f"Total edges: {total:,}", "", "By edge type:"]
            for edge_type, n in by_type:
                lines.append(f"- {edge_type}: {n:,}")
            outputs["report"].parent.mkdir(parents=True, exist_ok=True)
            outputs["report"].write_text("\n".join(lines) + "\n")
            log.info("\n".join(lines))
        finally:
            con.close()
