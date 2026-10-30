"""Stage P1: corpus audit. Measure the matrix before any filter decision."""

from pathlib import Path

from .common import (
    VerificationError,
    atomic_json_write,
    atomic_parquet_write,
    atomic_text_write,
    check,
    connect,
    content_hash,
    features_table,
    file_fingerprint,
    file_sha256,
    load_json,
    log,
    window_pred,
    with_thread_ladder,
)
from .config import FEATURES_DIR, FROZEN, MODEL_DIR, SCRATCH_DIR


def _user_partial_sql(path: str, ts_mode: str) -> str:
    wp_last = window_pred("src.last_ts", ts_mode)
    wp_first = window_pred("src.first_ts", ts_mode)
    return f"""
SELECT src.user_id AS user_id,
       count(*) AS pairs,
       CAST(sum(src.listens) AS BIGINT) AS listens,
       max(src.listens) AS max_pair_listens,
       min(src.first_ts) AS min_first_ts,
       max(src.last_ts) AS max_last_ts,
       CAST(count(*) FILTER (WHERE src.listens IS NULL) AS BIGINT) AS null_listens_rows,
       CAST(count(*) FILTER (WHERE src.first_ts IS NULL OR src.last_ts IS NULL) AS BIGINT) AS null_ts_rows,
       CAST(count(*) FILTER (WHERE src.first_ts > src.last_ts) AS BIGINT) AS ts_order_bad,
       CAST(count(*) FILTER (WHERE NOT ({wp_last})) AS BIGINT) AS last_out_window,
       CAST(count(*) FILTER (WHERE NOT ({wp_first})) AS BIGINT) AS first_out_window
FROM read_parquet('{path}') AS src
GROUP BY src.user_id
"""


def _item_partial_sql(path: str) -> str:
    return f"""
SELECT src.item_id AS item_id,
       count(*) AS pairs,
       CAST(sum(src.listens) AS BIGINT) AS listens,
       min(src.first_ts) AS min_first_ts,
       max(src.last_ts) AS max_last_ts
FROM read_parquet('{path}') AS src
GROUP BY src.item_id
"""


class StageP1:
    name = "p1_audit"
    upstream = []

    def outputs(self):
        return {
            "user_stats": MODEL_DIR / "stats_user_full.parquet",
            "item_stats": MODEL_DIR / "stats_item_full.parquet",
            "retention": MODEL_DIR / "p1_retention.parquet",
            "schema": MODEL_DIR / "schema_interactions.json",
            "feature_schemas": MODEL_DIR / "feature_schemas.json",
            "report": MODEL_DIR / "reports" / "p1_report.md",
        }

    def current_inputs(self, files):
        cache = MODEL_DIR / "fingerprint_cache.json"
        return {"interactions": [file_fingerprint(Path(f), cache) for f in files]}

    def run(self, params, files):
        scratch = SCRATCH_DIR / "p1"
        scratch.mkdir(parents=True, exist_ok=True)
        outputs = self.outputs()
        cache = MODEL_DIR / "fingerprint_cache.json"
        fingerprints = {f: file_fingerprint(Path(f), cache) for f in files}

        def work(con):
            schema = self._schema_and_mode(con, files, outputs["schema"])
            ts_mode = schema["ts_mode"]
            state = load_json(scratch / "state.json") or {}
            for index, path in enumerate(files):
                user_out = scratch / f"user_{index:02d}.parquet"
                item_out = scratch / f"item_{index:02d}.parquet"
                if (
                    state.get(path) == fingerprints[path]
                    and user_out.exists()
                    and item_out.exists()
                ):
                    log.info("p1: skip partial %d/%d (resume)", index + 1, len(files))
                    continue
                atomic_parquet_write(
                    con, _user_partial_sql(path, ts_mode), user_out, expected_mb=64
                )
                atomic_parquet_write(
                    con, _item_partial_sql(path), item_out, expected_mb=400
                )
                state[path] = fingerprints[path]
                atomic_json_write(state, scratch / "state.json")
                log.info("p1: partial %d/%d done", index + 1, len(files))

            atomic_parquet_write(
                con,
                self._merge_user_sql(scratch),
                outputs["user_stats"],
                expected_mb=128,
            )
            atomic_parquet_write(
                con,
                self._merge_item_sql(scratch),
                outputs["item_stats"],
                expected_mb=400,
            )
            atomic_parquet_write(
                con, self._retention_sql(params), outputs["retention"], expected_mb=8
            )
            self._dump_feature_schemas(con, outputs["feature_schemas"])

        with_thread_ladder(work, params["ladder"])

        info = {}
        con = connect(params["ladder"][0])
        try:
            for key, order in [
                ("user_stats", "user_id"),
                ("item_stats", "item_id"),
                ("retention", "dimension, floor"),
            ]:
                path = outputs[key]
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}')"
                ).fetchone()[0]
                digest = content_hash(con, f"read_parquet('{path}')", order)
                info[key] = {"path": str(path), "rows": int(rows), "sha256": digest}
            info["schema"] = {
                "path": str(outputs["schema"]),
                "rows": 0,
                "sha256": file_sha256(outputs["schema"]),
            }
            info["feature_schemas"] = {
                "path": str(outputs["feature_schemas"]),
                "rows": 0,
                "sha256": file_sha256(outputs["feature_schemas"]),
            }
        finally:
            con.close()
        return info

    def _schema_and_mode(self, con, files, dest):
        base = None
        for path in files:
            described = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
            signature = [(row[0], row[1]) for row in described]
            if base is None:
                base = signature
            else:
                check(
                    signature == base,
                    f"schema differs between interaction files: {path}",
                )
        types = dict(base)
        for column in ("user_id", "item_id", "listens", "first_ts", "last_ts"):
            check(
                column in types,
                f"column {column} missing, observed columns: {list(types)}",
            )
        ts_type = str(types["last_ts"]).upper()
        lo, hi = con.execute(
            f"SELECT min(first_ts), max(last_ts) FROM read_parquet('{files[0]}')"
        ).fetchone()
        if ts_type.startswith("TIMESTAMP"):
            mode = "timestamp_naive_utc"
        elif ts_type in ("BIGINT", "INTEGER", "HUGEINT"):
            check(hi is not None, "last_ts is empty in the first file")
            if hi < 10**10:
                mode = "epoch_s"
            elif hi < 10**13:
                mode = "epoch_ms"
            else:
                raise VerificationError(
                    f"last_ts magnitude {hi} matches no known epoch scale"
                )
        else:
            raise VerificationError(f"unexpected timestamp type {ts_type}")
        doc = {
            "columns": [{"name": name, "type": ctype} for name, ctype in base],
            "ts_mode": mode,
            "note": "timestamps are read as UTC. The local clock never enters the data.",
        }
        atomic_json_write(doc, dest)
        log.info(
            "p1: ts mode = %s, min(first_ts) = %s, max(last_ts) = %s", mode, lo, hi
        )
        return doc

    def _dump_feature_schemas(self, con, dest):
        """Record schema and row count of every table under features/.

        The key of a table is a per-table fact. item_listens proved this:
        it is keyed by UUID, not by integer item_id. This dump gives the
        ground truth for Stages P2 and P4 before they join anything.
        """
        doc = {}
        for path in sorted(FEATURES_DIR.glob("*.parquet")):
            described = con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{path}')"
            ).fetchall()
            rows = con.execute(
                f"SELECT count(*) FROM read_parquet('{path}')"
            ).fetchone()[0]
            doc[path.name] = {
                "rows": int(rows),
                "columns": [{"name": row[0], "type": row[1]} for row in described],
            }
        atomic_json_write(doc, dest)
        log.info("p1: dumped schemas of %d feature tables", len(doc))

    def _identity_row(self, con):
        """Cross-check matrix item stats against item_listens through item_ids.

        item_listens is keyed by canonical_recording_mbid. The matrix is
        keyed by integer item_id. The item_ids table carries the mapping.
        """
        item_stats = str(self.outputs()["item_stats"])
        truth = features_table("item_listens")
        ids = features_table("item_ids")
        ids_cols = [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{ids}')"
            ).fetchall()
        ]
        if "item_id" in ids_cols:
            item_col = "item_id"
        elif "id" in ids_cols:
            item_col = "id"
        else:
            raise VerificationError(
                f"item_ids columns are {ids_cols}, no integer key found"
            )
        others = [col for col in ids_cols if col != item_col]
        if "canonical_recording_mbid" in others:
            uuid_col = "canonical_recording_mbid"
        elif len(others) == 1:
            uuid_col = others[0]
        else:
            raise VerificationError(
                f"item_ids columns are {ids_cols}, cannot pick the UUID column"
            )
        truth_cols = {
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{truth}')"
            ).fetchall()
        }
        check(
            uuid_col in truth_cols and "listens" in truth_cols,
            f"item_listens columns are {sorted(truth_cols)}, expected "
            f"the UUID column {uuid_col} plus listens",
        )

        parts = [
            ("ids_rows", "(SELECT count(*) FROM ids)"),
            ("ids_distinct_uuid", "(SELECT count(DISTINCT mbid) FROM ids)"),
            ("ids_distinct_item", "(SELECT count(DISTINCT iid) FROM ids)"),
            ("truth_rows", "(SELECT count(*) FROM truth)"),
            (
                "mine_no_id",
                "(SELECT count(*) FROM mine LEFT JOIN ids ON mine.item_id = ids.iid"
                " WHERE ids.iid IS NULL)",
            ),
            (
                "ids_no_mine",
                "(SELECT count(*) FROM ids LEFT JOIN mine ON ids.iid = mine.item_id"
                " WHERE mine.item_id IS NULL)",
            ),
            (
                "truth_no_id",
                "(SELECT count(*) FROM truth LEFT JOIN ids ON truth.mbid = ids.mbid"
                " WHERE ids.iid IS NULL)",
            ),
            ("chain_rows", "(SELECT count(*) FROM chain)"),
            (
                "bad",
                "(SELECT count(*) FROM chain WHERE truth_listens IS NULL"
                " OR listens != truth_listens)",
            ),
        ]
        select_list = ", ".join(sql for _, sql in parts)
        query = f"""
            WITH mine AS (SELECT item_id, CAST(listens AS BIGINT) AS listens
                          FROM read_parquet('{item_stats}')),
                 ids AS (SELECT {item_col} AS iid, {uuid_col} AS mbid
                         FROM read_parquet('{ids}')),
                 truth AS (SELECT {uuid_col} AS mbid,
                                  CAST(listens AS BIGINT) AS listens
                           FROM read_parquet('{truth}')),
                 chain AS (SELECT mine.item_id, mine.listens,
                                  truth.listens AS truth_listens
                           FROM mine
                           JOIN ids ON mine.item_id = ids.iid
                           LEFT JOIN truth ON ids.mbid = truth.mbid)
            SELECT {select_list}
        """
        row = con.execute(query).fetchone()
        return {key: value for (key, _), value in zip(parts, row)}

    def _merge_user_sql(self, scratch):
        return f"""
SELECT user_id,
       CAST(sum(pairs) AS BIGINT) AS pairs,
       CAST(sum(listens) AS BIGINT) AS listens,
       max(max_pair_listens) AS max_pair_listens,
       min(min_first_ts) AS min_first_ts,
       max(max_last_ts) AS max_last_ts,
       CAST(sum(null_listens_rows) AS BIGINT) AS null_listens_rows,
       CAST(sum(null_ts_rows) AS BIGINT) AS null_ts_rows,
       CAST(sum(ts_order_bad) AS BIGINT) AS ts_order_bad,
       CAST(sum(last_out_window) AS BIGINT) AS last_out_window,
       CAST(sum(first_out_window) AS BIGINT) AS first_out_window
FROM read_parquet('{scratch}/user_*.parquet')
GROUP BY user_id
"""

    def _merge_item_sql(self, scratch):
        return f"""
SELECT item_id,
       CAST(sum(pairs) AS BIGINT) AS pairs,
       CAST(sum(listens) AS BIGINT) AS listens,
       min(min_first_ts) AS min_first_ts,
       max(max_last_ts) AS max_last_ts
FROM read_parquet('{scratch}/item_*.parquet')
GROUP BY item_id
"""

    def _retention_sql(self, params):
        item_stats = str(self.outputs()["item_stats"])
        user_stats = str(self.outputs()["user_stats"])
        parts = []
        for floor in params["item_pair_floors"]:
            parts.append(
                f"SELECT 'item' AS dimension, {int(floor)} AS floor, "
                f"count(*) FILTER (WHERE pairs >= {int(floor)}) AS survivors, "
                f"CAST(sum(pairs) FILTER (WHERE pairs >= {int(floor)}) AS BIGINT)"
                f" AS pairs_retained "
                f"FROM read_parquet('{item_stats}')"
            )
        for floor in params["user_pair_floors"]:
            parts.append(
                f"SELECT 'user' AS dimension, {int(floor)} AS floor, "
                f"count(*) FILTER (WHERE pairs >= {int(floor)}) AS survivors, "
                f"CAST(sum(pairs) FILTER (WHERE pairs >= {int(floor)}) AS BIGINT)"
                f" AS pairs_retained "
                f"FROM read_parquet('{user_stats}')"
            )
        return " UNION ALL ".join(parts) + " ORDER BY dimension, floor"

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        us, it = str(outputs["user_stats"]), str(outputs["item_stats"])
        try:
            users, pairs, listens = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{us}')"
            ).fetchone()
            check(
                users == FROZEN["users"],
                f"user count {users} does not match frozen {FROZEN['users']}",
            )
            check(
                pairs == FROZEN["pairs"],
                f"pair count {pairs} does not match frozen {FROZEN['pairs']}",
            )
            check(
                listens == FROZEN["matrix_listens"],
                f"listen total {listens} does not match frozen "
                f"{FROZEN['matrix_listens']}",
            )

            items, item_pairs, item_total = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{it}')"
            ).fetchone()
            check(
                items == FROZEN["items"],
                f"item count {items} does not match frozen {FROZEN['items']}",
            )
            check(
                item_pairs == FROZEN["pairs"],
                f"item-side pair total {item_pairs} does not match frozen",
            )
            check(
                item_total == FROZEN["matrix_listens"],
                f"item-side listen total {item_total} does not match frozen",
            )

            nulls = con.execute(
                f"SELECT sum(null_listens_rows), sum(null_ts_rows), "
                f"sum(ts_order_bad) FROM read_parquet('{us}')"
            ).fetchone()
            check(
                nulls == (0, 0, 0),
                f"null or bad-order rows exist: null_listens={nulls[0]}, "
                f"null_ts={nulls[1]}, first_gt_last={nulls[2]}",
            )
            check(
                con.execute(
                    f"SELECT count(*) FROM read_parquet('{us}') "
                    f"WHERE user_id IS NULL"
                ).fetchone()[0]
                == 0,
                "NULL user_id rows exist",
            )
            check(
                con.execute(
                    f"SELECT count(*) FROM read_parquet('{it}') "
                    f"WHERE item_id IS NULL"
                ).fetchone()[0]
                == 0,
                "NULL item_id rows exist",
            )

            identity = self._identity_row(con)
            check(
                identity["ids_distinct_uuid"] == identity["ids_rows"],
                f"item_ids maps one UUID to several rows: "
                f"{identity['ids_rows']} rows, "
                f"{identity['ids_distinct_uuid']} distinct UUIDs",
            )
            check(
                identity["ids_distinct_item"] == identity["ids_rows"],
                f"item_ids maps one item to several rows: "
                f"{identity['ids_rows']} rows, "
                f"{identity['ids_distinct_item']} distinct item ids",
            )
            check(
                identity["truth_rows"] == FROZEN["items"],
                f"item_listens has {identity['truth_rows']} rows, "
                f"frozen expects {FROZEN['items']}",
            )
            check(
                identity["mine_no_id"] == 0,
                f"{identity['mine_no_id']} matrix items have no item_ids row",
            )
            check(
                identity["ids_no_mine"] == 0,
                f"{identity['ids_no_mine']} item_ids rows never appear "
                f"in the matrix",
            )
            check(
                identity["truth_no_id"] == 0,
                f"{identity['truth_no_id']} item_listens rows have no " f"item_ids row",
            )
            check(
                identity["chain_rows"] == FROZEN["items"],
                f"the identity join fanned out: {identity['chain_rows']} "
                f"rows for {FROZEN['items']} items",
            )
            check(
                identity["bad"] == 0,
                f"listen sums differ from item_listens on "
                f"{identity['bad']} items. Stop and investigate.",
            )
            _reconcile_frozen_tables(outputs)
        finally:
            con.close()

    def report(self, params):
        con = connect(4)
        outputs = self.outputs()
        us, it = str(outputs["user_stats"]), str(outputs["item_stats"])
        retention = str(outputs["retention"])
        lines = ["# P1 audit report", ""]
        add = lines.append
        try:
            users, pairs, listens = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{us}')"
            ).fetchone()
            add(f"- users: {users} (frozen {FROZEN['users']})")
            add(f"- pairs: {pairs} (frozen {FROZEN['pairs']})")
            add(f"- listens: {listens} (frozen {FROZEN['matrix_listens']})")
            schema = load_json(outputs["schema"]) or {}
            add(f"- timestamp mode: {schema.get('ts_mode')}")
            flags = con.execute(
                f"SELECT sum(last_out_window), sum(first_out_window) "
                f"FROM read_parquet('{us}')"
            ).fetchone()
            add(
                f"- pairs with last_ts outside 2005-2013: {flags[0]} "
                f"(pair counts, not listen counts)"
            )
            add(
                f"- pairs with first_ts outside 2005-2013: {flags[1]} "
                f"(pair counts, not listen counts)"
            )
            multi = con.execute(
                f"SELECT count(*) FROM (SELECT user_id FROM read_parquet("
                f"'{SCRATCH_DIR}/p1/user_*.parquet') "
                f"GROUP BY user_id HAVING count(*) > 1)"
            ).fetchone()[0]
            add(f"- users present in more than one file: {multi}")
            up = con.execute(
                f"SELECT quantile_cont(pairs, 0.5), quantile_cont(pairs, 0.9), "
                f"quantile_cont(pairs, 0.99) FROM read_parquet('{us}')"
            ).fetchone()
            add(f"- user pairs p50/p90/p99: {up[0]:.0f} / {up[1]:.0f} / {up[2]:.0f}")
            ip = con.execute(
                f"SELECT quantile_cont(pairs, 0.5), quantile_cont(pairs, 0.9), "
                f"quantile_cont(pairs, 0.99) FROM read_parquet('{it}')"
            ).fetchone()
            add(f"- item pairs p50/p90/p99: {ip[0]:.0f} / {ip[1]:.0f} / {ip[2]:.0f}")
            biggest = con.execute(
                f"SELECT max(max_pair_listens) FROM read_parquet('{us}')"
            ).fetchone()[0]
            add(f"- largest single-pair listen count: {biggest}")
            add("")
            add("Retention (choose filter floors from these numbers):")
            for dim, floor, survivors, kept in con.execute(
                f"SELECT dimension, floor, survivors, pairs_retained "
                f"FROM read_parquet('{retention}') "
                f"ORDER BY dimension, floor"
            ).fetchall():
                add(f"- {dim} floor {floor}: {survivors} rows, {kept} pairs retained")
            add("")
            identity = self._identity_row(con)
            add(
                "item_listens identity through the item_ids map "
                "(every gap count must be zero):"
            )
            add(f"- item_ids rows: {identity['ids_rows']}")
            add(f"- item_listens rows: {identity['truth_rows']}")
            add(f"- matrix items with no item_ids row: {identity['mine_no_id']}")
            add(f"- item_ids rows absent from the matrix: {identity['ids_no_mine']}")
            add(
                f"- item_listens rows with no item_ids row: "
                f"{identity['truth_no_id']}"
            )
            add(
                f"- join fan-out rows: {identity['chain_rows']} "
                f"(must equal {FROZEN['items']})"
            )
            add(f"- items with different listen sums: {identity['bad']}")
            add("")
            add(
                "Feature table schemas and row counts "
                "(ground truth for Stages P2 and P4):"
            )
            schemas = load_json(outputs["feature_schemas"]) or {}
            for name in sorted(schemas):
                entry = schemas[name]
                cols = [col["name"] for col in entry["columns"]]
                add(f"- {name} ({entry['rows']} rows): {cols}")
        finally:
            con.close()
        atomic_text_write("\n".join(lines) + "\n", outputs["report"])
        log.info("p1 report written")


def _reconcile_frozen_tables(outputs):
    """Assert every feature-table row count against the frozen layer."""
    from .config import FROZEN_TABLE_ROWS

    schemas = load_json(outputs["feature_schemas"]) or {}
    check(
        set(schemas) >= set(FROZEN_TABLE_ROWS),
        "frozen tables missing from the schema dump: "
        + str(sorted(set(FROZEN_TABLE_ROWS) - set(schemas))),
    )
    for name in sorted(FROZEN_TABLE_ROWS):
        expected = FROZEN_TABLE_ROWS[name]
        entry = schemas.get(name) or {}
        rows = entry.get("rows")
        check(name in schemas, name + " is missing from the schema dump")
        check(
            rows == expected,
            name + ": measured " + str(rows) + " rows, frozen expects " + str(expected),
        )
