"""Stage P2.5: the training table and the train-only statistics.

The anti-join removes every pair named in splits.parquet.
The original 16 interaction files stay untouched.
Train row order carries no contract. The CSR build sorts later.
"""

from pathlib import Path

from .common import (
    atomic_json_write,
    atomic_parquet_write,
    atomic_text_write,
    check,
    connect,
    content_hash,
    file_fingerprint,
    file_sha256,
    load_json,
    log,
    marker_path,
    require_space,
    with_thread_ladder,
)
from .config import FROZEN, MODEL_DIR, SCRATCH_DIR


def _train_file_sql(path: str, splits: str) -> str:
    return f"""
SELECT src.user_id AS user_id,
       src.item_id AS item_id,
       src.listens AS listens,
       src.first_ts AS first_ts,
       src.last_ts AS last_ts,
       CAST(ln(1 + src.listens) AS FLOAT) AS w_log,
       least(src.listens, 255)::UTINYINT AS listens_u8
FROM read_parquet('{path}') AS src
LEFT JOIN read_parquet('{splits}') AS held
       ON src.user_id = held.user_id AND src.item_id = held.item_id
WHERE held.user_id IS NULL
"""


def _train_user_partial_sql(path: str) -> str:
    return f"""
SELECT src.user_id AS user_id,
       count(*) AS pairs,
       CAST(sum(src.listens) AS BIGINT) AS listens,
       min(src.first_ts) AS min_first_ts,
       max(src.last_ts) AS max_last_ts,
       max(src.listens) AS max_pair_listens
FROM read_parquet('{path}') AS src
GROUP BY src.user_id
"""


def _train_item_partial_sql(path: str) -> str:
    return f"""
SELECT src.item_id AS item_id,
       count(*) AS pairs,
       CAST(sum(src.listens) AS BIGINT) AS listens
FROM read_parquet('{path}') AS src
GROUP BY src.item_id
"""


class StageP25:
    name = "p25_train"
    upstream = ["p2_split"]

    def outputs(self):
        return {
            "train_manifest": MODEL_DIR / "train_manifest.json",
            "stats_user_train": MODEL_DIR / "stats_user_train.parquet",
            "stats_item_train": MODEL_DIR / "stats_item_train.parquet",
            "report": MODEL_DIR / "reports" / "p25_report.md",
        }

    def current_inputs(self, files):
        try:
            return {"marker:p2_split": file_sha256(marker_path("p2_split"))}
        except FileNotFoundError:
            return {"marker:p2_split": None}

    def run(self, params, files):
        scratch = SCRATCH_DIR / "p25"
        scratch.mkdir(parents=True, exist_ok=True)
        outputs = self.outputs()
        splits = str(MODEL_DIR / "splits.parquet")
        cache = MODEL_DIR / "fingerprint_cache.json"
        fingerprints = {f: file_fingerprint(Path(f), cache) for f in files}
        in_out = [
            (
                f,
                str(
                    MODEL_DIR / f"interactions_train_"
                    f"{Path(f).stem.removeprefix('interactions_')}.parquet"
                ),
            )
            for f in files
        ]

        need = int(sum(Path(f).stat().st_size for f in files) * 1.15)
        require_space(MODEL_DIR, need)
        log.info("p25: the train table needs about %.0f GB free", need / 1e9)

        def work(con):
            splits_sha = file_sha256(marker_path("p2_split"))
            state = load_json(scratch / "state.json") or {}
            if state.get("splits_sha") != splits_sha:
                log.info("p25: the split changed, all train partials rebuild")
                state = {"splits_sha": splits_sha}
            manifest = state.get("manifest") or {}

            for index, (src_path, dest_path) in enumerate(in_out):
                dest = Path(dest_path)
                entry = manifest.get(dest.name)
                if (
                    entry
                    and entry.get("input") == fingerprints[src_path]
                    and dest.exists()
                ):
                    log.info("p25: skip partial %d/%d (resume)", index + 1, len(files))
                    continue
                atomic_parquet_write(
                    con, _train_file_sql(src_path, splits), dest, expected_mb=3400
                )
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{dest}')"
                ).fetchone()[0]
                manifest[dest.name] = {
                    "input": fingerprints[src_path],
                    "rows": int(rows),
                    "bytes": int(dest.stat().st_size),
                }
                state["manifest"] = manifest
                atomic_json_write(state, scratch / "state.json")
                log.info(
                    "p25: train file %d/%d done, %d rows", index + 1, len(files), rows
                )

            atomic_json_write(manifest, outputs["train_manifest"])
            stats_epoch = file_sha256(outputs["train_manifest"])
            stats_ok = state.get("stats_epoch") == stats_epoch and all(
                (scratch / f"user_{i:02d}.parquet").exists()
                and (scratch / f"item_{i:02d}.parquet").exists()
                for i in range(len(files))
            )
            if not stats_ok:
                for index, (_, dest_path) in enumerate(in_out):
                    atomic_parquet_write(
                        con,
                        _train_user_partial_sql(dest_path),
                        scratch / f"user_{index:02d}.parquet",
                        expected_mb=64,
                    )
                    atomic_parquet_write(
                        con,
                        _train_item_partial_sql(dest_path),
                        scratch / f"item_{index:02d}.parquet",
                        expected_mb=400,
                    )
                    log.info("p25: stats partial %d/%d done", index + 1, len(files))
                state["stats_epoch"] = stats_epoch
                atomic_json_write(state, scratch / "state.json")

            atomic_parquet_write(
                con,
                self._merge_user_sql(scratch),
                outputs["stats_user_train"],
                expected_mb=128,
            )
            atomic_parquet_write(
                con,
                self._merge_item_sql(scratch),
                outputs["stats_item_train"],
                expected_mb=400,
            )

        with_thread_ladder(work, params["ladder"])

        info = {}
        con = connect(params["ladder"][0])
        try:
            for key, order in [
                ("stats_user_train", "user_id"),
                ("stats_item_train", "item_id"),
            ]:
                path = outputs[key]
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{path}')"
                ).fetchone()[0]
                digest = content_hash(con, f"read_parquet('{path}')", order)
                info[key] = {"path": str(path), "rows": int(rows), "sha256": digest}
            info["train_manifest"] = {
                "path": str(outputs["train_manifest"]),
                "rows": 0,
                "sha256": file_sha256(outputs["train_manifest"]),
            }
        finally:
            con.close()
        return info

    def _merge_user_sql(self, scratch):
        return f"""
SELECT user_id,
       CAST(sum(pairs) AS BIGINT) AS pairs,
       CAST(sum(listens) AS BIGINT) AS listens,
       min(min_first_ts) AS min_first_ts,
       max(max_last_ts) AS max_last_ts,
       max(max_pair_listens) AS max_pair_listens
FROM read_parquet('{scratch}/user_*.parquet')
GROUP BY user_id
"""

    def _merge_item_sql(self, scratch):
        return f"""
SELECT item_id,
       CAST(sum(pairs) AS BIGINT) AS pairs,
       CAST(sum(listens) AS BIGINT) AS listens
FROM read_parquet('{scratch}/item_*.parquet')
GROUP BY item_id
"""

    def verify(self, params):
        con = connect(4)
        outputs = self.outputs()
        splits = str(MODEL_DIR / "splits.parquet")
        utrain = str(outputs["stats_user_train"])
        itrain = str(outputs["stats_item_train"])
        try:
            held_rows, held_listens = con.execute(
                f"SELECT count(*), CAST(sum(listens) AS BIGINT) "
                f"FROM read_parquet('{splits}')"
            ).fetchone()
            exp_pairs = FROZEN["pairs"] - held_rows
            exp_listens = FROZEN["matrix_listens"] - held_listens

            users, upairs, ulistens = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{utrain}')"
            ).fetchone()
            check(
                users == FROZEN["users"],
                f"train user count {users} does not match " f"frozen {FROZEN['users']}",
            )
            check(
                upairs == exp_pairs,
                f"train pair total {upairs} does not match {exp_pairs}",
            )
            check(
                ulistens == exp_listens,
                f"train listen total {ulistens} does not match {exp_listens}",
            )

            items, ipairs, ilistens = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{itrain}')"
            ).fetchone()
            check(
                ipairs == exp_pairs,
                f"item-side train pair total {ipairs} does not match {exp_pairs}",
            )
            check(
                ilistens == exp_listens,
                f"item-side train listen total {ilistens} does not "
                f"match {exp_listens}",
            )

            zero_train = con.execute(f"""
                WITH held AS (SELECT item_id, count(*) AS held_pairs
                              FROM read_parquet('{splits}') GROUP BY item_id),
                     fstats AS (SELECT item_id, pairs
                                FROM read_parquet(
                                  '{MODEL_DIR}/stats_item_full.parquet'))
                SELECT count(*) FROM fstats JOIN held USING (item_id)
                WHERE fstats.pairs = held.held_pairs
            """).fetchone()[0]
            check(
                items == FROZEN["items"] - zero_train,
                f"train item count {items} does not match the expected "
                f"{FROZEN['items'] - zero_train}. {zero_train} items lost "
                f"every pair to the holdout",
            )

            manifest = load_json(outputs["train_manifest"]) or {}
            check(len(manifest) == 16, f"train manifest lists {len(manifest)} files")
            for name in sorted(manifest):
                tpath = MODEL_DIR / name
                rows = con.execute(
                    f"SELECT count(*) FROM read_parquet('{tpath}')"
                ).fetchone()[0]
                check(
                    rows == manifest[name]["rows"],
                    f"{name}: {rows} rows on disk, the manifest says "
                    f"{manifest[name]['rows']}",
                )
                hits = con.execute(f"""
                    SELECT count(*) FROM read_parquet('{tpath}') t
                    JOIN read_parquet('{splits}') s
                      ON t.user_id = s.user_id AND t.item_id = s.item_id
                """).fetchone()[0]
                check(hits == 0, f"{name} still contains {hits} held-out pairs")
                bad = con.execute(f"""
                    SELECT count(*) FROM read_parquet('{tpath}')
                    WHERE listens IS NULL OR w_log IS NULL OR listens_u8 IS NULL
                       OR w_log < 0 OR listens_u8 > 255
                       OR listens_u8 != least(listens, 255)
                       OR w_log != CAST(ln(1 + listens) AS FLOAT)
                """).fetchone()[0]
                check(bad == 0, f"{name}: {bad} rows carry bad weight values")
        finally:
            con.close()

    def report(self, params):
        con = connect(4)
        outputs = self.outputs()
        utrain = str(outputs["stats_user_train"])
        itrain = str(outputs["stats_item_train"])
        splits = str(MODEL_DIR / "splits.parquet")
        lines = ["# P2.5 train table report", ""]
        add = lines.append
        try:
            held_rows, held_listens = con.execute(
                f"SELECT count(*), CAST(sum(listens) AS BIGINT) "
                f"FROM read_parquet('{splits}')"
            ).fetchone()
            users, pairs, listens = con.execute(
                f"SELECT count(*), sum(pairs), sum(listens) "
                f"FROM read_parquet('{utrain}')"
            ).fetchone()
            items = con.execute(
                f"SELECT count(*) FROM read_parquet('{itrain}')"
            ).fetchone()[0]
            add(f"- held-out pairs: {held_rows}, held-out listens: {held_listens}")
            add(f"- train pairs: {pairs} (expected {FROZEN['pairs'] - held_rows})")
            add(
                f"- train listens: {listens} (expected "
                f"{FROZEN['matrix_listens'] - held_listens})"
            )
            add(f"- train users: {users} (every corpus user)")
            add(
                f"- items with at least one train pair: {items} "
                f"({FROZEN['items'] - items} lost every pair to the holdout)"
            )
            add("")
            add("Per-file train table:")
            manifest = load_json(outputs["train_manifest"]) or {}
            total_gb = 0.0
            for name in sorted(manifest):
                gb = manifest[name]["bytes"] / 1e9
                total_gb += gb
                add(f"- {name}: {manifest[name]['rows']} rows, {gb:.2f} GB")
            add(f"- total on disk: {total_gb:.1f} GB")
            add("")
            up = con.execute(
                f"SELECT quantile_cont(pairs, 0.5), quantile_cont(pairs, 0.9), "
                f"quantile_cont(pairs, 0.99) FROM read_parquet('{utrain}')"
            ).fetchone()
            add(
                f"- train user pairs p50/p90/p99: "
                f"{up[0]:.0f} / {up[1]:.0f} / {up[2]:.0f}"
            )
            ip = con.execute(
                f"SELECT quantile_cont(pairs, 0.5), quantile_cont(pairs, 0.9), "
                f"quantile_cont(pairs, 0.99) FROM read_parquet('{itrain}')"
            ).fetchone()
            add(
                f"- train item pairs p50/p90/p99: "
                f"{ip[0]:.0f} / {ip[1]:.0f} / {ip[2]:.0f}"
            )
            big = con.execute(
                f"SELECT max(max_pair_listens) FROM read_parquet('{utrain}')"
            ).fetchone()[0]
            add(f"- largest single-pair listen count: {big}")
            add("")
            add("CSR memory plan, from the actual nnz:")
            add(f"- nonzeros: {pairs}")
            add(f"- int32 indices: {pairs * 4 / 1e9:.1f} GB")
            add(f"- uint8 values: {pairs / 1e9:.1f} GB")
            add(
                "- plan: int32 indices plus uint8 values, about 18.5 GB. "
                "A float32 copy of w_log is built only inside a training "
                "process, never on disk."
            )
        finally:
            con.close()
        atomic_text_write("\n".join(lines) + "\n", outputs["report"])
        log.info("p25 report written")
