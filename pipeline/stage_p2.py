"""Stage P2: split construction. Leave-last-out on both timestamp orders.

Correctness of the per-file top-2 merge: a global top-2 row of a user has at
most one row above it in the whole matrix, so at most one row above it in its
own file. It is therefore a top-2 row of that file. The union of the per-file
picks contains the global top-2 for every user, and the re-rank over the union
returns the global ranks.
"""

from pathlib import Path

from .common import (
    atomic_json_write,
    atomic_parquet_write,
    atomic_text_write,
    check,
    connect,
    content_hash,
    features_table,
    file_fingerprint,
    load_json,
    log,
    marker_path,
    file_sha256,
    window_pred,
    year_expr,
    with_thread_ladder,
)
from .config import FROZEN, MODEL_DIR, SCRATCH_DIR


def _top2_sql(path: str) -> str:
    return f"""
SELECT user_id, item_id, listens, first_ts, last_ts
FROM (
    SELECT src.user_id AS user_id,
           src.item_id AS item_id,
           src.listens AS listens,
           src.first_ts AS first_ts,
           src.last_ts AS last_ts,
           row_number() OVER (PARTITION BY src.user_id
                              ORDER BY src.last_ts DESC, src.item_id DESC) AS rn_last,
           row_number() OVER (PARTITION BY src.user_id
                              ORDER BY src.first_ts DESC, src.item_id DESC) AS rn_first
    FROM read_parquet('{path}') AS src
) AS ranked
WHERE rn_last <= 2 OR rn_first <= 2
"""


class StageP2:
    name = "p2_split"
    upstream = ["p1_audit"]

    def outputs(self):
        return {
            "splits": MODEL_DIR / "splits.parquet",
            "report": MODEL_DIR / "reports" / "p2_report.md",
        }

    def current_inputs(self, files):
        try:
            return {"marker:p1_audit": file_sha256(marker_path("p1_audit"))}
        except FileNotFoundError:
            return {"marker:p1_audit": None}

    def run(self, params, files):
        scratch = SCRATCH_DIR / "p2"
        scratch.mkdir(parents=True, exist_ok=True)
        outputs = self.outputs()
        min_train = int(params["eligibility_min_train_pairs"])
        user_stats = MODEL_DIR / "stats_user_full.parquet"
        cache = MODEL_DIR / "fingerprint_cache.json"
        fingerprints = {f: file_fingerprint(Path(f), cache) for f in files}
        self._double_check = "SKIPPED by params"

        def work(con):
            state = load_json(scratch / "state.json") or {}
            for index, path in enumerate(files):
                out = scratch / f"top2_{index:02d}.parquet"
                if state.get(path) == fingerprints[path] and out.exists():
                    log.info("p2: skip partial %d/%d (resume)", index + 1, len(files))
                    continue
                atomic_parquet_write(con, _top2_sql(path), out, expected_mb=128)
                state[path] = fingerprints[path]
                atomic_json_write(state, scratch / "state.json")
                log.info("p2: partial %d/%d done", index + 1, len(files))
            atomic_parquet_write(
                con,
                self._assign_sql(scratch, user_stats, min_train),
                outputs["splits"],
                expected_mb=128,
            )

        with_thread_ladder(work, params["ladder"])

        if params.get("double_check_split", True):
            self._double_check_run(params, files, user_stats, min_train)

        con = connect(params["ladder"][0])
        try:
            rows = con.execute(
                f"SELECT count(*) FROM read_parquet('{outputs['splits']}')"
            ).fetchone()[0]
            digest = content_hash(
                con, f"read_parquet('{outputs['splits']}')", "user_id, item_id"
            )
            info = {
                "splits": {
                    "path": str(outputs["splits"]),
                    "rows": int(rows),
                    "sha256": digest,
                }
            }
        finally:
            con.close()
        return info

    def _assign_sql(self, scratch, user_stats, min_train):
        return f"""
WITH merged AS (
    SELECT user_id, item_id, listens, first_ts, last_ts
    FROM read_parquet('{scratch}/top2_*.parquet')
),
ranked AS (
    SELECT user_id, item_id, listens, first_ts, last_ts,
           row_number() OVER (PARTITION BY user_id
                              ORDER BY last_ts DESC, item_id DESC) AS rn_last,
           row_number() OVER (PARTITION BY user_id
                              ORDER BY first_ts DESC, item_id DESC) AS rn_first
    FROM merged
),
picks AS (
    SELECT user_id, item_id, listens, first_ts, last_ts,
           (rn_last = 1) AS is_test_last,
           (rn_last = 2) AS is_val_last,
           (rn_first = 1) AS is_test_first,
           (rn_first = 2) AS is_val_first
    FROM ranked
    WHERE rn_last <= 2 OR rn_first <= 2
),
held AS (
    SELECT user_id, count(*) AS held_pairs FROM picks GROUP BY user_id
)
SELECT p.user_id, p.item_id, p.listens, p.first_ts, p.last_ts,
       p.is_test_last, p.is_val_last, p.is_test_first, p.is_val_first,
       CAST(u.pairs - h.held_pairs AS BIGINT) AS train_pairs
FROM picks p
JOIN held h ON p.user_id = h.user_id
JOIN read_parquet('{user_stats}') u ON p.user_id = u.user_id
WHERE u.pairs - h.held_pairs >= {int(min_train)}
ORDER BY p.user_id, p.item_id
"""

    def _double_check_run(self, params, files, user_stats, min_train):
        """Recompute the split on a different thread path and compare content.

        This check catches order instability that file bytes cannot prove.
        """
        scratch = SCRATCH_DIR / "p2_check"
        scratch.mkdir(parents=True, exist_ok=True)
        dest = scratch / "splits_check.parquet"

        def work(con):
            for index, path in enumerate(files):
                out = scratch / f"top2_{index:02d}.parquet"
                atomic_parquet_write(con, _top2_sql(path), out, expected_mb=128)
                log.info("p2 check: partial %d/%d done", index + 1, len(files))
            atomic_parquet_write(
                con,
                self._assign_sql(scratch, user_stats, min_train),
                dest,
                expected_mb=128,
            )

        check_ladder = params["ladder"][1:] or params["ladder"]
        with_thread_ladder(work, check_ladder)

        con = connect(check_ladder[0])
        try:
            original = content_hash(
                con, f"read_parquet('{self.outputs()['splits']}')", "user_id, item_id"
            )
            repeated = content_hash(con, f"read_parquet('{dest}')", "user_id, item_id")
        finally:
            con.close()
        check(
            original == repeated,
            "double-check failed: the split content changed between thread paths",
        )
        self._double_check = "PASS"
        log.info("p2 double-check passed")

    def verify(self, params):
        con = connect(4)
        splits = str(self.outputs()["splits"])
        min_train = int(params["eligibility_min_train_pairs"])
        try:
            duplicates = con.execute(f"""
                SELECT count(*) FROM (
                    SELECT user_id, item_id FROM read_parquet('{splits}')
                    GROUP BY user_id, item_id HAVING count(*) > 1)
            """).fetchone()[0]
            check(
                duplicates == 0, f"duplicate (user, item) rows in splits: {duplicates}"
            )

            bad_roles = con.execute(f"""
                SELECT count(*) FROM (
                    SELECT user_id,
                           count(*) FILTER (WHERE is_test_last) AS a,
                           count(*) FILTER (WHERE is_val_last) AS b,
                           count(*) FILTER (WHERE is_test_first) AS c,
                           count(*) FILTER (WHERE is_val_first) AS d,
                           min(train_pairs) AS tp
                    FROM read_parquet('{splits}') GROUP BY user_id)
                WHERE a != 1 OR b != 1 OR c != 1 OR d != 1 OR tp < {min_train}
            """).fetchone()[0]
            check(
                bad_roles == 0, f"role or eligibility violation for {bad_roles} users"
            )

            orphans = con.execute(f"""
                SELECT count(*) FROM read_parquet('{splits}') s
                LEFT JOIN read_parquet('{MODEL_DIR}/stats_item_full.parquet') i
                       ON s.item_id = i.item_id
                WHERE i.item_id IS NULL
            """).fetchone()[0]
            check(orphans == 0, f"split items missing from item stats: {orphans}")

            fuzzy = features_table("item_spotify_fuzzy")
            observed = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{fuzzy}')"
                ).fetchall()
            }
            check(
                "basis" in observed,
                f"item_spotify_fuzzy columns are {sorted(observed)}",
            )
            n_fuzzy = con.execute(
                f"SELECT count(*) FROM read_parquet('{fuzzy}') "
                f"WHERE basis = 'name+duration'"
            ).fetchone()[0]
            check(
                n_fuzzy == FROZEN["fuzzy_name_duration"],
                f"fuzzy name+duration rows {n_fuzzy} do not match frozen "
                f"{FROZEN['fuzzy_name_duration']}",
            )
        finally:
            con.close()

    def report(self, params):
        schema = load_json(MODEL_DIR / "schema_interactions.json") or {}
        ts_mode = schema.get("ts_mode")
        con = connect(4)
        splits = str(self.outputs()["splits"])
        item_stats = str(MODEL_DIR / "stats_item_full.parquet")
        lines = ["# P2 split report", ""]
        add = lines.append
        try:
            eligible, rows = con.execute(
                f"SELECT count(DISTINCT user_id), count(*) "
                f"FROM read_parquet('{splits}')"
            ).fetchone()
            excluded = FROZEN["users"] - eligible
            add(f"- eligible users: {eligible}")
            add(f"- excluded users (train pairs below the floor): {excluded}")
            add(f"- held-out rows: {rows}, at most 4 per user")
            add(f"- double-check: {getattr(self, '_double_check', 'NOT RUN')}")
            same, total = con.execute(f"""
                SELECT count(*) FILTER (WHERE a.item_id = b.item_id), count(*)
                FROM (SELECT user_id, item_id FROM read_parquet('{splits}')
                      WHERE is_test_last) a
                JOIN (SELECT user_id, item_id FROM read_parquet('{splits}')
                      WHERE is_test_first) b ON a.user_id = b.user_id
            """).fetchone()
            add(
                f"- users where both variants pick the same test item: "
                f"{same} of {total}"
            )
            add("")
            add("Test picks by year of last_ts (pair counts, not listen counts):")
            for yr, count in con.execute(f"""
                    SELECT {year_expr('last_ts', ts_mode)} AS yr, count(*) AS picks
                    FROM read_parquet('{splits}') WHERE is_test_last
                    GROUP BY yr ORDER BY yr""").fetchall():
                add(f"- {yr}: {count}")
            artifact = con.execute(f"""
                SELECT count(*) FROM read_parquet('{splits}')
                WHERE is_test_last AND NOT ({window_pred('last_ts', ts_mode)})
            """).fetchone()[0]
            add(
                f"- test picks with last_ts outside 2005-2013: {artifact} "
                f"(policy: keep and flag)"
            )
            add("")
            add("Tier of test items:")
            map_table = features_table("item_spotify_map")
            fuzzy_table = features_table("item_spotify_fuzzy")
            map_cols = {
                row[0]
                for row in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{map_table}')"
                ).fetchall()
            }
            check(
                "match_method" in map_cols,
                f"item_spotify_map columns are {sorted(map_cols)}",
            )
            overlap = con.execute(f"""
                SELECT count(*) FROM read_parquet('{map_table}') m
                JOIN read_parquet('{fuzzy_table}') z ON m.item_id = z.item_id
            """).fetchone()[0]
            add(
                f"- items present in both map and fuzzy tables: {overlap} "
                f"(expected 0, the map wins on any conflict)"
            )
            for tier, count in con.execute(f"""
                    WITH test AS (SELECT item_id FROM read_parquet('{splits}')
                                  WHERE is_test_last),
                         m AS (SELECT item_id, match_method
                               FROM read_parquet('{map_table}')),
                         z AS (SELECT item_id, basis
                               FROM read_parquet('{fuzzy_table}'))
                    SELECT COALESCE(m.match_method,
                           CASE WHEN z.basis = 'name+duration' THEN 'fuzzy_dur'
                                WHEN z.basis = 'name_only' THEN 'fuzzy_name'
                                ELSE 'unmatched' END) AS tier,
                           count(*) AS n
                    FROM test
                    LEFT JOIN m ON test.item_id = m.item_id
                    LEFT JOIN z ON test.item_id = z.item_id
                    GROUP BY tier ORDER BY n DESC""").fetchall():
                add(f"- {tier}: {count}")
            add("")
            add(
                "Train pairs of test items, exact (full pairs minus held-out "
                "holders). This sizes the cold slice:"
            )
            cold = con.execute(f"""
                WITH removed AS (SELECT item_id, count(*) AS holders
                                 FROM read_parquet('{splits}') GROUP BY item_id),
                     test AS (SELECT DISTINCT item_id FROM read_parquet('{splits}')
                              WHERE is_test_last),
                     base AS (SELECT t.item_id,
                                     s.pairs - COALESCE(r.holders, 0) AS train_pairs
                              FROM test t
                              JOIN read_parquet('{item_stats}') s
                                   ON t.item_id = s.item_id
                              LEFT JOIN removed r ON t.item_id = r.item_id)
                SELECT count(*),
                       count(*) FILTER (WHERE train_pairs = 0),
                       count(*) FILTER (WHERE train_pairs <= 5),
                       quantile_cont(train_pairs, 0.5),
                       quantile_cont(train_pairs, 0.9)
                FROM base
            """).fetchone()
            add(
                f"- test items: {cold[0]}, with 0 train pairs: {cold[1]}, "
                f"with 5 or fewer: {cold[2]}"
            )
            add(f"- train pairs p50/p90: {cold[3]:.0f} / {cold[4]:.0f}")
            add("")
            add(
                "Note: both holdout variants share one train matrix. The "
                "secondary variant loses one extra item per user, at most "
                "0.02 percent of a profile."
            )
        finally:
            con.close()
        atomic_text_write("\n".join(lines) + "\n", self.outputs()["report"])
        log.info("p2 report written")
